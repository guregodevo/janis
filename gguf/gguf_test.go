package gguf

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/guregodevo/janis/safetensors"
)

// The container tests run against a real model stored in TWO containers, so the
// same weights can be compared across readers. That pairing is the point: a
// reader tested only against files it wrote proves nothing.
//
// They skip unless JANIS_GGUF_DIR names a directory holding both files, and each
// is a 1.5 GB download, so they are not committed:
//
//	curl -L -o qwen3-0.6b-f16.gguf \
//	  https://huggingface.co/ggml-org/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-f16.gguf
//	curl -L -o qwen3-0.6b.safetensors \
//	  https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors
//	JANIS_GGUF_DIR=$PWD go test ./gguf/ -run Qwen3 -v
func fixtures(t *testing.T) (ggufPath, sfPath string) {
	t.Helper()
	dir := os.Getenv("JANIS_GGUF_DIR")
	if dir == "" {
		t.Skip("set JANIS_GGUF_DIR to a directory holding qwen3-0.6b-f16.gguf " +
			"and qwen3-0.6b.safetensors (see the comment on fixtures)")
	}
	ggufPath = filepath.Join(dir, "qwen3-0.6b-f16.gguf")
	sfPath = filepath.Join(dir, "qwen3-0.6b.safetensors")
	for _, p := range []string{ggufPath, sfPath} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("%s not present: %v", p, err)
		}
	}
	return ggufPath, sfPath
}

// TestQwen3GGUFMetadata checks the reader against the published model card of
// Qwen3-0.6B: values this repo knows independently of the file.
func TestQwen3GGUFMetadata(t *testing.T) {
	path, _ := fixtures(t)
	g, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer g.Close()

	m := g.Metadata()
	if got := m["general.architecture"]; got != "qwen3" {
		t.Errorf("general.architecture = %v, want qwen3", got)
	}
	for _, tc := range []struct {
		key  string
		want uint64
	}{
		{"qwen3.block_count", 28},
		{"qwen3.embedding_length", 1024},
		{"qwen3.attention.head_count", 16},
		{"qwen3.attention.head_count_kv", 8}, // the model is GQA: 8 KV heads
		{"qwen3.feed_forward_length", 3072},
		{"qwen3.attention.key_length", 128},
	} {
		if got, ok := m[tc.key].(uint64); !ok || got != tc.want {
			t.Errorf("%s = %v (%T), want %d", tc.key, m[tc.key], m[tc.key], tc.want)
		}
	}
	if got, ok := m["qwen3.rope.freq_base"].(float32); !ok || got != 1e6 {
		t.Errorf("qwen3.rope.freq_base = %v, want 1e6", m["qwen3.rope.freq_base"])
	}
	if got, ok := m["qwen3.attention.layer_norm_rms_epsilon"].(float32); !ok || got != 1e-6 {
		t.Errorf("rms epsilon = %v, want 1e-6", m["qwen3.attention.layer_norm_rms_epsilon"])
	}
	if got := g.Alignment(); got != 32 {
		t.Errorf("alignment = %d, want the 32 default when general.alignment is absent", got)
	}
	// The tokenizer arrays are the reader's largest metadata objects; a reader
	// that stops early would still report the scalar keys above.
	tok, ok := m["tokenizer.ggml.tokens"].([]any)
	if !ok || len(tok) != 151936 {
		t.Errorf("tokenizer.ggml.tokens = %d entries, want 151936", len(tok))
	}
}

// TestQwen3GGUFInventory checks the tensor table: 28 layers x 11 + 3 globals.
func TestQwen3GGUFInventory(t *testing.T) {
	path, _ := fixtures(t)
	g, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()

	names := g.Names()
	if len(names) != 311 {
		t.Errorf("tensor count = %d, want 311", len(names))
	}
	if len(g.TensorNames()) != len(names) {
		t.Error("TensorNames and Names disagree on the count")
	}
	// Types: the norms stay F32, the matrices are F16 in this conversion.
	for _, tc := range []struct {
		name string
		want TensorType
		rank int
	}{
		{"token_embd.weight", TensorF16, 2},
		{"output.weight", TensorF16, 2},
		{"output_norm.weight", TensorF32, 1},
		{"blk.0.attn_norm.weight", TensorF32, 1},
		{"blk.0.attn_q.weight", TensorF16, 2},
		{"blk.0.attn_q_norm.weight", TensorF32, 1},
		{"blk.27.ffn_down.weight", TensorF16, 2},
	} {
		name, tt, shape, err := g.Info(tc.name)
		if err != nil {
			t.Errorf("%s: %v", tc.name, err)
			continue
		}
		if tt != tc.want {
			t.Errorf("%s type = %s, want %s", tc.name, name, TypeName(tc.want))
		}
		if len(shape) != tc.rank {
			t.Errorf("%s rank = %d, want %d (%v)", tc.name, len(shape), tc.rank, shape)
		}
	}
	// Every tensor must be one the reader can actually dequantize, or the reader
	// would hand a caller raw blocks it cannot interpret.
	for _, n := range names {
		if _, tt, _, _ := g.Info(n); tt != TensorF16 && tt != TensorF32 {
			t.Errorf("unexpected type %s for %s", TypeName(tt), n)
		}
	}
	if g.Has("does.not.exist") {
		t.Error("Has reported a tensor that is not in the file")
	}
}

// TestQwen3GGUFMatchesSafetensors is the decisive check: the same weights in two
// INDEPENDENT containers must come out the same through this repo's two readers.
//
// It cross-checks three things a self-consistent round-trip cannot: the GGUF
// container offsets (a wrong data start shows up as garbage), the F16
// dequantizer, and the dimension-order convention. The last one is worth stating
// because the obvious assumption is wrong: GGUF reports shapes in GGML order
// (Shape[0] contiguous) while safetensors is row-major, so the two agree once
// the DIMS ARE REVERSED and the values are read flat — no transpose of the data.
// This test found that: it was first written to transpose, and the numbers
// disagreed until measuring both containers showed the flat order identical.
//
// Tolerance: the safetensors file is BF16 and the GGUF is F16, so the two agree
// to within the F16 representation; measured, the difference is ~3e-8 absolute.
// A shape mismatch, or anything past this tolerance, is a defect.
func TestQwen3GGUFMatchesSafetensors(t *testing.T) {
	ggufPath, sfPath := fixtures(t)
	g, err := Open(ggufPath)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	sf, err := safetensors.Open(sfPath)
	if err != nil {
		t.Fatal(err)
	}
	defer sf.Close()

	// gguf name -> safetensors name, for tensors covering every shape class.
	pairs := [][2]string{
		{"token_embd.weight", "model.embed_tokens.weight"},
		{"output.weight", "lm_head.weight"},
		{"output_norm.weight", "model.norm.weight"},
		{"blk.0.attn_norm.weight", "model.layers.0.input_layernorm.weight"},
		{"blk.0.attn_q.weight", "model.layers.0.self_attn.q_proj.weight"},
		{"blk.0.attn_k.weight", "model.layers.0.self_attn.k_proj.weight"},
		{"blk.0.attn_q_norm.weight", "model.layers.0.self_attn.q_norm.weight"},
		{"blk.0.ffn_gate.weight", "model.layers.0.mlp.gate_proj.weight"},
		{"blk.0.ffn_up.weight", "model.layers.0.mlp.up_proj.weight"},
		{"blk.0.ffn_down.weight", "model.layers.0.mlp.down_proj.weight"},
		{"blk.13.ffn_down.weight", "model.layers.13.mlp.down_proj.weight"},
		{"blk.27.attn_output.weight", "model.layers.27.self_attn.o_proj.weight"},
	}
	const relTol = 0.01 // 1%: BF16 rounding is ~0.4% and F16 is finer
	for _, p := range pairs {
		gotVals, gotShape, err := g.F32(p[0])
		if err != nil {
			t.Errorf("%s: %v", p[0], err)
			continue
		}
		dt, sfShape, raw, err := sf.Get(p[1])
		if err != nil {
			t.Errorf("%s: %v", p[1], err)
			continue
		}
		// Same contiguous data; the dims are in opposite orders.
		wantShape := RowMajorShape(gotShape)
		if !sameShape(sfShape, wantShape) {
			t.Errorf("%s: safetensors shape %v, want %v (gguf %v reversed)",
				p[1], sfShape, wantShape, gotShape)
			continue
		}
		sfVals, err := decodeFloats(dt, raw)
		if err != nil {
			t.Errorf("%s: %v", p[1], err)
			continue
		}
		if len(sfVals) != len(gotVals) {
			t.Errorf("%s: %d values vs %d in %s", p[0], len(gotVals), len(sfVals), p[1])
			continue
		}
		// Both are now in the same storage order, so compare value by value. The
		// relative metric is guarded by a floor: a weight near zero has no
		// meaningful relative error and would otherwise dominate the maximum.
		maxRel, maxAbs, worst := 0.0, 0.0, 0
		for i := range gotVals {
			a, b := float64(gotVals[i]), float64(sfVals[i])
			abs := math.Abs(a - b)
			if abs > maxAbs {
				maxAbs, worst = abs, i
			}
			if denom := math.Abs(b); denom > 1e-2 {
				maxRel = math.Max(maxRel, abs/denom)
			}
		}
		if maxRel > relTol {
			t.Errorf("%s vs %s: max relative difference %.4f > %.3f (abs %.3g) — "+
				"the gguf reader or its dequantizer is wrong",
				p[0], p[1], maxRel, relTol, maxAbs)
		}
		t.Logf("%-26s %-42s max rel %.6f  abs %.2e (at %d)",
			p[0], p[1], maxRel, maxAbs, worst)
	}

	// And prove the comparison discriminates: reading the same tensor with the
	// dims the WRONG way round (as if row-major) must differ substantially, or
	// this test would pass on a reader that ignored the dimension order.
	vals, shape, err := g.F32("blk.0.ffn_gate.weight")
	if err != nil {
		t.Fatal(err)
	}
	_, sfShape, raw, err := sf.Get("model.layers.0.mlp.gate_proj.weight")
	if err != nil {
		t.Fatal(err)
	}
	sfVals, err := decodeFloats("BF16", raw)
	if err != nil {
		t.Fatal(err)
	}
	// The wrong view: take gguf's dims as row-major and index transposed.
	rows, cols := sfShape[0], sfShape[1] // row-major [out, in]
	wrongSame, wrongDiff, maxWrong := 0, 0, 0.0
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			a := float64(vals[j*rows+i]) // read as if the dims were swapped
			bb := float64(sfVals[i*cols+j])
			if abs := math.Abs(a - bb); abs < 1e-2 {
				wrongSame++
			} else {
				wrongDiff++
				maxWrong = math.Max(maxWrong, abs)
			}
		}
	}
	if wrongDiff == 0 {
		t.Errorf("the swapped-dimension read agrees on all %d values, so the shape "+
			"assertion above proves nothing (gguf %v, safetensors %v)",
			rows*cols, shape, sfShape)
	}
	t.Logf("swapped dims: only %d/%d values agree (max off %.3g) — the dimension "+
		"order is load-bearing", wrongSame, rows*cols, maxWrong)
}

func sameShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// decodeFloats turns a safetensors tensor's bytes into float32 values.
func decodeFloats(dtype string, raw []byte) ([]float32, error) {
	switch dtype {
	case "F32":
		out := make([]float32, len(raw)/4)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
		}
		return out, nil
	case "F16", "BF16":
		out := make([]float32, len(raw)/2)
		for i := range out {
			h := binary.LittleEndian.Uint16(raw[2*i:])
			if dtype == "F16" {
				out[i] = f16(h)
			} else {
				out[i] = math.Float32frombits(uint32(h) << 16)
			}
		}
		return out, nil
	}
	return nil, &unsupportedDtype{dtype}
}

type unsupportedDtype struct{ d string }

func (e *unsupportedDtype) Error() string { return "unsupported dtype " + e.d }
