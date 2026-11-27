//go:build ggml && darwin && arm64 && cgo

package inkling2

import (
	"math"
	"math/rand"
	"path/filepath"
	"testing"

	"github.com/guregodevo/janis/engine"
	"github.com/guregodevo/janis/ggml"
	"github.com/guregodevo/janis/mlxc"
	"github.com/guregodevo/janis/qwen"
	"github.com/guregodevo/janis/safetensors"
)

// EACH LOOP OPERATION ON GGML GIVES WHAT MLX GIVES.
//
// The same random inputs through both backends: the five-axis shapes the
// attention uses, the clip, the router's top k and its gather, the bias
// gather over the heads, and the tiny model's expert stacks. An nvfp4
// stack is ggml's NVFP4, whose matmul rounds its input to 8 bits (3% of
// the largest output allowed); an mxfp8 one is decoded to f16 (0.2%).
func TestGGMLLoopOpsMatchMLX(t *testing.T) {
	g := ggml.New()
	defer g.Close()
	m := mlxc.New()
	defer m.Close()
	r := rand.New(rand.NewSource(1))
	rnd := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(r.NormFloat64())
		}
		return v
	}
	type B interface {
		engine.Backend
		engine.LoopOps
	}
	both := func(name string, tol float64, f func(b B) engine.Tensor) {
		t.Helper()
		got, want := g.Floats(f(g)), m.Floats(f(m))
		if len(got) != len(want) {
			t.Fatalf("%s: %d values on ggml, %d on MLX", name, len(got), len(want))
		}
		if d, s := maxDiff(got, want); d > tol*s || math.IsNaN(d) {
			t.Errorf("%s: off by %.3g (largest value %.3g)", name, d, s)
		}
	}
	x := rnd(1 * 3 * 4 * 5 * 6)
	y := rnd(1 * 3 * 1 * 6 * 7)
	both("matmul5d", 1e-5, func(b B) engine.Tensor {
		return b.MatMul(b.FromFloats(x, 1, 3, 4, 5, 6), b.FromFloats(y, 1, 3, 1, 6, 7))
	})
	both("softmax5d+concat+slice", 1e-5, func(b B) engine.Tensor {
		a := b.FromFloats(x, 1, 3, 4, 5, 6)
		s := b.FromFloats(x[:60], 1, 3, 4, 5, 1)
		return b.Slice(b.Softmax(b.Concat(a, s, 4), 4), 4, 0, 6)
	})
	both("add5d bias", 1e-5, func(b B) engine.Tensor {
		return b.Add(b.ScalarMul(b.FromFloats(x, 1, 3, 4, 5, 6), 0.5), b.Reshape(b.FromFloats(x, 1, 12, 5, 6), 1, 3, 4, 5, 6))
	})
	both("transpose5d", 1e-5, func(b B) engine.Tensor {
		return b.Transpose(b.ExpandDims(b.FromFloats(x[:120], 1, 4, 5, 6), 2), 0, 1, 2, 4, 3)
	})
	both("l2clip-ish", 1e-5, func(b B) engine.Tensor {
		s := b.FromFloats(x, 1, 5, 12, 6)
		n := b.Sqrt(b.ExpandDims(b.ScalarMul(b.Mean(b.Mul(s, s), 3), 6), 3))
		f := b.Minimum(b.Div(b.FromFloats([]float32{2}, 1), b.ScalarAdd(n, 1e-6)), b.FromFloats([]float32{1}, 1))
		return b.Mul(s, f)
	})
	both("topk+takealong", 1e-5, func(b B) engine.Tensor {
		s := b.FromFloats(x[:5*16], 1, 5, 16)
		idx := b.TopK(s, 4)
		w := b.TakeAlong(s, idx, 2)
		return b.ExpandDims(b.ScalarMul(b.Mean(w, 2), 4), 2)
	})
	idx := make([]int32, 5*7)
	for i := range idx {
		idx[i] = int32(r.Intn(6))
	}
	both("takealong wide", 1e-5, func(b B) engine.Tensor {
		return b.TakeAlong(b.FromFloats(x[:3*5*6], 1, 3, 5, 6), b.FromInt32(idx, 1, 1, 5, 7), 3)
	})
	both("conv", 1e-5, func(b B) engine.Tensor {
		return b.Conv1dDepthwise(b.FromFloats(x[:9*8], 1, 9, 8), b.FromFloats(y[:8*4], 8, 4, 1))
	})
	both("rmsnorm", 1e-5, func(b B) engine.Tensor {
		return b.RMSNorm(b.FromFloats(x[:40], 1, 5, 8), b.FromFloats(y[:8], 8), 1e-6)
	})

	// The tiny model's expert stacks.
	dir := filepath.Join("testdata", "tiny-f32")
	st, err := safetensors.OpenModel(dir)
	if err != nil {
		t.Fatal(err)
	}
	stacks := 0
	for _, name := range st.Names() {
		if len(name) < 7 || name[len(name)-7:] != ".weight" {
			continue
		}
		base := name[:len(name)-7]
		wdt, ws, _ := st.Info(name)
		sdt, ss, err := st.Info(base + ".scales")
		if err != nil || wdt != "U32" || len(ws) != 3 || sdt != "U8" {
			continue
		}
		bits := 0
		in := ss[2] * 16
		if ws[2]*32/in == 4 {
			bits = 4
		} else {
			in = ss[2] * 32
			bits = ws[2] * 32 / in
		}
		gs := in / ss[2]
		mode := expertModes[[2]int{bits, gs}]
		xi := rnd(3 * in)
		ids := []int32{0, 1, int32(ws[0] - 1), 1, 0, 2}
		tol := 3e-2
		if mode != "nvfp4" {
			tol = 2e-3
		}
		stacks++
		both(base+" "+mode, tol, func(b B) engine.Tensor {
			w, _ := qwen.LoadTensor(b, st, name)
			sc, _ := qwen.LoadTensor(b, st, base+".scales")
			return b.GatherModeMatmul(b.FromFloats(xi, 1, 3, 1, 1, in), w, sc, b.FromInt32(ids, 1, 3, 2), gs, bits, mode)
		})
	}
	if stacks != 16 {
		t.Errorf("%d expert stacks compared, the tiny model has 16", stacks)
	}
}
