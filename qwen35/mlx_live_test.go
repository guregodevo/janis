//go:build darwin && arm64 && cgo

package qwen35

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/guregodevo/janis/hostfit"
	"github.com/guregodevo/janis/mlxc"
	"github.com/guregodevo/janis/qwen"
	"github.com/guregodevo/janis/safetensors"
)

// smallModel loads Qwen3.5-0.8B on MLX, weights pinned. Gated like the 9B
// live test; skipped when the model is not in the cache.
func smallModel(t *testing.T) (*mlxc.Backend, *Model) {
	t.Helper()
	if os.Getenv("MLX_QWEN35_E2E") != "1" {
		t.Skip("set MLX_QWEN35_E2E=1 to run (loads Qwen3.5-0.8B-MLX-4bit)")
	}
	dirs, _ := filepath.Glob(filepath.Join(hostfit.CacheDir(), "models--mlx-community--Qwen3.5-0.8B-MLX-4bit/snapshots/*"))
	if len(dirs) == 0 {
		t.Skip("Qwen3.5-0.8B-MLX-4bit not installed")
	}
	cfg, err := qwen.LoadConfig(dirs[0])
	if err != nil {
		t.Fatal(err)
	}
	st, err := safetensors.OpenModel(dirs[0])
	if err != nil {
		t.Fatal(err)
	}
	b := mlxc.New()
	t.Cleanup(b.Close)
	m, err := LoadModel(b, st, cfg)
	if err != nil {
		t.Fatal(err)
	}
	b.PinAll()
	return b, m
}

// THE RESIDUAL STREAM STAYS IN THE MODEL'S DTYPE.
//
// The Gated DeltaNet layer handed its f32 gated norm to out_proj, and the
// residual add promoted the stream to f32 from the first layer on: every
// later matmul, norm and KV write moved twice the bytes (the 9B decoded at
// 14.6 tok/s, 20.7 in bf16). The reference casts back (precise_swiglu).
func TestTheResidualStreamStaysInTheModelDtype(t *testing.T) {
	b, m := smallModel(t)
	ids := []int32{785, 6722, 315, 9625, 374}
	h := b.Reshape(m.Embed.Forward(b, b.FromInt32(ids, len(ids))), 1, len(ids), m.Cfg.Hidden)
	want := b.DTypeOf(h)
	for i, blk := range m.Blocks[:m.Cfg.FullAttnInterval] {
		h = blk.ForwardCached(b, h, nil, 0)
		if got := b.DTypeOf(h); got != want {
			t.Fatalf("after layer %d (linear=%v) the stream is dtype %v, the embedding's is %v", i, blk.Linear != nil, got, want)
		}
	}
	if got := b.DTypeOf(m.Logits(b, ids)); got != want {
		t.Fatalf("logits are dtype %v, the embedding's is %v", got, want)
	}
}

// A PREFILL CHUNK'S INTERMEDIATES ARE FREED AS THE PASS RUNS.
//
// The session swept a chunk's arrays after its eval, so the handles kept
// every layer's activations alive through it: the 9B peaked 3.7 GB over its
// weights on a 256-token chunk. Swept before the eval, MLX frees each buffer
// once its last consumer has run.
func TestAPrefillChunkFreesItsIntermediatesAsItRuns(t *testing.T) {
	b, m := smallModel(t)
	ids := make([]int32, 1024)
	for i := range ids {
		ids[i] = int32(1000 + i%500)
	}
	s := m.NewSession()
	s.PrefillChunk = 512
	before, _, _ := mlxc.MemoryStatsMB()
	mlxc.ResetPeakMemory()
	if s.Prefill(b, ids, nil) == nil {
		t.Fatal("no logits")
	}
	_, _, peak := mlxc.MemoryStatsMB()
	t.Logf("active before %.0f MB, peak %.0f MB, +%.0f MB for 512-token chunks", before, peak, peak-before)
	if peak-before > peakBudgetMB {
		t.Fatalf("a 512-token chunk peaked %.0f MB over the weights; budget %d MB", peak-before, peakBudgetMB)
	}
}

// peakBudgetMB sits between the two orders, measured on an M4 with 512-token
// chunks: +1,185 MB swept before the eval, +2,872 MB after it (+3,568 MB with
// the stream in f32 as well).
const peakBudgetMB = 2000

// A BATCHED DECODE STEP IS EACH CONVERSATION'S OWN STEP.
//
// Three conversations of different lengths, each prefilled on its own
// session, decode two tokens in one batched forward per step; the same
// conversations decoded one at a time are the reference. The logits agree
// to bf16 rounding and pick the same token.
func TestABatchedStepIsEachConversationsOwnStep(t *testing.T) {
	b, m := smallModel(t)
	prompts := [][]int32{{785, 6722, 315, 9625, 374}, {9707, 11, 1879}, {40, 1079, 264, 2632, 315, 279, 1879, 13}}
	steps := [][]int32{{264, 30, 279}, {1879, 11, 315}}
	sessions := func() ([][]qwen.LayerCache, []int) {
		caches := make([][]qwen.LayerCache, len(prompts))
		offsets := make([]int, len(prompts))
		for j, p := range prompts {
			s := m.NewSession()
			s.Prefill(b, p, nil)
			caches[j], offsets[j] = s.Caches, len(p)
		}
		return caches, offsets
	}

	own, ownOff := sessions()
	batch, batchOff := sessions()
	for n, ids := range steps {
		got := b.Floats(m.ForwardBatch(b, ids, batchOff, batch))
		V := len(got) / len(ids)
		for j := range prompts {
			want := b.Floats(m.forwardCachedT(b, b.FromInt32(ids[j:j+1], 1), 1, ownOff[j], own[j]))
			row := got[j*V : (j+1)*V]
			if len(want) != V {
				t.Fatalf("vocab %d, batch row %d", len(want), V)
			}
			maxDiff, wantTop, gotTop := 0.0, 0, 0
			for i := range want {
				if d := math.Abs(float64(want[i] - row[i])); d > maxDiff {
					maxDiff = d
				}
				if want[i] > want[wantTop] {
					wantTop = i
				}
				if row[i] > row[gotTop] {
					gotTop = i
				}
			}
			t.Logf("step %d, conversation %d: max |diff| %.4f, top %d / %d", n, j, maxDiff, gotTop, wantTop)
			if gotTop != wantTop || maxDiff > 0.5 {
				t.Errorf("step %d, conversation %d: batched top %d, own %d, max |diff| %.4f", n, j, gotTop, wantTop, maxDiff)
			}
			ownOff[j]++
		}
		for j := range batchOff {
			batchOff[j]++
		}
	}
}
