//go:build ggml

package qwen35

import (
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/guregodevo/janis/ggml"
	"github.com/guregodevo/janis/hostfit"
	"github.com/guregodevo/janis/qwen"
	"github.com/guregodevo/janis/safetensors"
)

// ggmlModel loads JANIS_BATCH_MODEL (default Qwen3.5-0.8B-MLX-4bit) on the
// ggml backend: the GPU with JANIS_GGML_DEVICE=gpu.
//
//	JANIS_GGML_DEVICE=gpu JANIS_BATCH_E2E=1 JANIS_BATCH_MODEL=Qwen3.8-27B-4bit go test -tags ggml ./qwen35 -run Batch -v
func ggmlModel(t *testing.T) (*ggml.Backend, *Model) {
	t.Helper()
	if os.Getenv("JANIS_BATCH_E2E") != "1" {
		t.Skip("set JANIS_BATCH_E2E=1 to run")
	}
	// The device is set on the thread that opens it, as the server's engine
	// thread does: a cgo call from another thread failed cudaSetDevice.
	runtime.LockOSThread()
	t.Cleanup(runtime.UnlockOSThread)
	name := os.Getenv("JANIS_BATCH_MODEL")
	if name == "" {
		name = "Qwen3.5-0.8B-MLX-4bit"
	}
	dirs, _ := filepath.Glob(filepath.Join(hostfit.CacheDir(), "models--mlx-community--"+name, "snapshots/*"))
	if len(dirs) == 0 {
		t.Skip(name + " not installed")
	}
	cfg, err := qwen.LoadConfig(dirs[0])
	if err != nil {
		t.Fatal(err)
	}
	st, err := safetensors.OpenModel(dirs[0])
	if err != nil {
		t.Fatal(err)
	}
	b := ggml.New()
	m, err := LoadModel(b, st, cfg)
	if err != nil {
		t.Fatal(err)
	}
	b.PinAll() // the weights into their buffers, as the engine does at the load
	return b, m
}

// A BATCHED DECODE STEP IS EACH CONVERSATION'S OWN STEP, on ggml.
//
// Three conversations of different lengths decode two tokens, batched and
// one at a time. A batch of rows may take another matmul kernel than one
// row, so the logits are compared to a tolerance and must pick the same
// token.
func TestABatchedStepIsEachConversationsOwnStepOnGGML(t *testing.T) {
	b, m := ggmlModel(t)
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
			if gotTop != wantTop || maxDiff > 1 {
				t.Errorf("step %d, conversation %d: batched top %d, own %d, max |diff| %.4f", n, j, gotTop, wantTop, maxDiff)
			}
			ownOff[j]++
			batchOff[j]++
		}
	}
}

// Not a check: tokens per second of a batched decode step, B conversations
// (JANIS_BATCH_SIZES, default 1,4,8,16,32) at a short prompt.
func TestBatchedDecodeSpeedOnGGML(t *testing.T) {
	b, m := ggmlModel(t)
	sizes := []int{1, 4, 8, 16, 32}
	if v := os.Getenv("JANIS_BATCH_SIZES"); v != "" {
		sizes = nil
		for _, f := range strings.Split(v, ",") {
			if n, err := strconv.Atoi(f); err == nil {
				sizes = append(sizes, n)
			}
		}
	}
	prompt := []int32{785, 6722, 315, 9625, 374, 40, 1079, 264, 2632, 315, 279, 1879, 13}
	for _, B := range sizes {
		caches := make([][]qwen.LayerCache, B)
		offsets := make([]int, B)
		ids := make([]int32, B)
		for j := range caches {
			s := m.NewSession()
			s.Prefill(b, prompt, nil)
			caches[j], offsets[j], ids[j] = s.Caches, len(prompt), 264
		}
		b.Floats(m.ForwardBatch(b, ids, offsets, caches)) // first step: kernels and buffers
		for j := range offsets {
			offsets[j]++
		}
		const steps = 10
		start := time.Now()
		for n := 0; n < steps; n++ {
			b.Floats(m.ForwardBatch(b, ids, offsets, caches))
			for j := range offsets {
				offsets[j]++
			}
		}
		el := time.Since(start).Seconds()
		t.Logf("B=%2d: %.1f ms a step, %.0f tok/s", B, 1000*el/steps, float64(B*steps)/el)
	}
}

// A RECORDED BATCHED STEP IS EACH CONVERSATION'S OWN STEP.
//
// Three conversations decode six tokens through a BatchDecoder with
// replay on: the first step stacks the recurrent states, the second is
// recorded, the rest replay it. Each step's logits are compared with each
// conversation decoded on its own, reading the caches padded as a
// recorded step does (JANIS_GGML_PADDECODE).
func TestARecordedBatchedStepIsEachConversationsOwnStepOnGGML(t *testing.T) {
	t.Setenv("JANIS_GGML_REPLAY", "1")
	t.Setenv("JANIS_GGML_PADDECODE", "1")
	b, m := ggmlModel(t)
	prompts := [][]int32{{785, 6722, 315, 9625, 374}, {9707, 11, 1879}, {40, 1079, 264, 2632, 315, 279, 1879, 13}}
	toks := [][]int32{{264, 30, 279}, {1879, 11, 315}, {13, 264, 30}, {315, 1879, 11}, {30, 13, 264}, {11, 315, 1879}}
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
	d := m.NewBatchDecoder()
	for n, ids := range toks {
		got := d.Step(b, ids, batchOff, batch)
		V := len(got) / len(ids)
		for j := range prompts {
			want := b.Floats(m.forwardCachedT(b, b.FromInt32(ids[j:j+1], 1), 1, ownOff[j], own[j]))
			row := got[j*V : (j+1)*V]
			maxDiff, wantTop, gotTop := 0.0, 0, 0
			for i := range want {
				if dd := math.Abs(float64(want[i] - row[i])); dd > maxDiff {
					maxDiff = dd
				}
				if want[i] > want[wantTop] {
					wantTop = i
				}
				if row[i] > row[gotTop] {
					gotTop = i
				}
			}
			t.Logf("step %d (recorded: %v), conversation %d: max |diff| %.4f, top %d / %d", n, d.step != nil, j, maxDiff, gotTop, wantTop)
			if gotTop != wantTop || maxDiff > 1 {
				t.Errorf("step %d, conversation %d: batched top %d, own %d, max |diff| %.4f", n, j, gotTop, wantTop, maxDiff)
			}
			ownOff[j]++
			batchOff[j]++
		}
	}
	if d.step == nil {
		t.Error("no step was recorded")
	}
	d.Settle()
	for j, cs := range batch {
		for _, c := range cs {
			if c.Len() != batchOff[j] {
				t.Fatalf("conversation %d: a cache holds %d tokens after Settle, want %d", j, c.Len(), batchOff[j])
			}
		}
	}
}

// Not a check: tokens per second of BatchDecoder steps with replay on.
func TestRecordedBatchedDecodeSpeedOnGGML(t *testing.T) {
	t.Setenv("JANIS_GGML_REPLAY", "1")
	b, m := ggmlModel(t)
	sizes := []int{1, 4, 8, 16, 32}
	if v := os.Getenv("JANIS_BATCH_SIZES"); v != "" {
		sizes = nil
		for _, f := range strings.Split(v, ",") {
			if n, err := strconv.Atoi(f); err == nil {
				sizes = append(sizes, n)
			}
		}
	}
	prompt := []int32{785, 6722, 315, 9625, 374, 40, 1079, 264, 2632, 315, 279, 1879, 13}
	for _, B := range sizes {
		caches := make([][]qwen.LayerCache, B)
		offsets := make([]int, B)
		ids := make([]int32, B)
		for j := range caches {
			s := m.NewSession()
			s.Prefill(b, prompt, nil)
			caches[j], offsets[j], ids[j] = s.Caches, len(prompt), 264
		}
		d := m.NewBatchDecoder()
		for n := 0; n < 2; n++ { // a plain step stacks the states, the next records
			d.Step(b, ids, offsets, caches)
			for j := range offsets {
				offsets[j]++
			}
		}
		const steps = 20
		start := time.Now()
		for n := 0; n < steps; n++ {
			d.Step(b, ids, offsets, caches)
			for j := range offsets {
				offsets[j]++
			}
		}
		el := time.Since(start).Seconds()
		t.Logf("B=%2d: %.1f ms a step, %.0f tok/s (replayed: %v)", B, 1000*el/steps, float64(B*steps)/el, d.step != nil)
		d.Settle()
	}
}
