//go:build ggml

package ggml

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/guregodevo/janis/cpu"
	"github.com/guregodevo/janis/engine"
)

// The CPU backend is the oracle (it is proven against MLX by
// cpu/cpu_mlx_diff_test.go); ggml must match it op by op on the shapes the
// models use.

func rnd(n int, seed int64) []float32 {
	r := rand.New(rand.NewSource(seed))
	d := make([]float32, n)
	for i := range d {
		d[i] = r.Float32()*2 - 1
	}
	return d
}

func rndPackedU32(n int, seed int64) []byte {
	r := rand.New(rand.NewSource(seed))
	raw := make([]byte, n*4)
	for i := 0; i < n; i++ {
		binary.LittleEndian.PutUint32(raw[i*4:], r.Uint32())
	}
	return raw
}

func diffOK(t *testing.T, name string, tol float64, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Errorf("%s: %d values, the oracle has %d", name, len(got), len(want))
		return
	}
	var m float64
	for i := range got {
		if d := math.Abs(float64(got[i]) - float64(want[i])); d > m {
			m = d
		}
	}
	if m > tol {
		t.Errorf("%s: max|ggml-cpu| = %g > tol %g (first: ggml %v cpu %v)", name, m, tol, got[:min(4, len(got))], want[:min(4, len(want))])
	} else {
		t.Logf("%s: max|ggml-cpu| = %g", name, m)
	}
}

// relOK compares with a tolerance relative to the largest value the oracle
// produced: a quantized kernel's error scales with the output.
func relOK(t *testing.T, name string, rel float64, got, want []float32) {
	t.Helper()
	var scale float64
	for _, w := range want {
		scale = math.Max(scale, math.Abs(float64(w)))
	}
	diffOK(t, name, rel*scale, got, want)
}

func TestGGMLMatchesCPU(t *testing.T) {
	g := New()
	defer g.Close()
	c := cpu.New()
	const tol = 2e-3
	t.Log(g.Version())

	a, w := rnd(2*3*4, 1), rnd(2*4*5, 2)
	diffOK(t, "MatMul", tol,
		g.Floats(g.MatMul(g.FromFloats(a, 2, 3, 4), g.FromFloats(w, 2, 4, 5))),
		c.Floats(c.MatMul(c.FromFloats(a, 2, 3, 4), c.FromFloats(w, 2, 4, 5))))

	s := rnd(4*7, 3)
	diffOK(t, "Softmax", tol, g.Floats(g.Softmax(g.FromFloats(s, 4, 7), -1)), c.Floats(c.Softmax(c.FromFloats(s, 4, 7), -1)))
	diffOK(t, "Add(broadcast)", tol,
		g.Floats(g.Add(g.FromFloats(s, 4, 7), g.FromFloats(s[:7], 7))),
		c.Floats(c.Add(c.FromFloats(s, 4, 7), c.FromFloats(s[:7], 7))))
	diffOK(t, "Mul+SiLU", tol,
		g.Floats(g.Mul(g.SiLU(g.FromFloats(s, 4, 7)), g.FromFloats(s, 4, 7))),
		c.Floats(c.Mul(c.SiLU(c.FromFloats(s, 4, 7)), c.FromFloats(s, 4, 7))))

	x, rw := rnd(3*8, 4), rnd(8, 5)
	diffOK(t, "RMSNorm", tol,
		g.Floats(g.RMSNorm(g.FromFloats(x, 3, 8), g.FromFloats(rw, 8), 1e-5)),
		c.Floats(c.RMSNorm(c.FromFloats(x, 3, 8), c.FromFloats(rw, 8), 1e-5)))
	lb := rnd(8, 6)
	diffOK(t, "LayerNorm", tol,
		g.Floats(g.LayerNorm(g.FromFloats(x, 3, 8), g.FromFloats(rw, 8), g.FromFloats(lb, 8), 1e-5)),
		c.Floats(c.LayerNorm(c.FromFloats(x, 3, 8), c.FromFloats(rw, 8), c.FromFloats(lb, 8), 1e-5)))

	diffOK(t, "Transpose(0,2,1,3)", 0,
		g.Floats(g.Transpose(g.FromFloats(rnd(2*3*4*5, 7), 2, 3, 4, 5), 0, 2, 1, 3)),
		c.Floats(c.Transpose(c.FromFloats(rnd(2*3*4*5, 7), 2, 3, 4, 5), 0, 2, 1, 3)))
	diffOK(t, "Reshape", 0,
		g.Floats(g.Reshape(g.FromFloats(rnd(24, 8), 2, 12), 2, 3, 4)),
		c.Floats(c.Reshape(c.FromFloats(rnd(24, 8), 2, 12), 2, 3, 4)))
	diffOK(t, "Concat(axis 2)", 0,
		g.Floats(g.Concat(g.FromFloats(rnd(1*2*3*4, 9), 1, 2, 3, 4), g.FromFloats(rnd(1*2*1*4, 10), 1, 2, 1, 4), 2)),
		c.Floats(c.Concat(c.FromFloats(rnd(1*2*3*4, 9), 1, 2, 3, 4), c.FromFloats(rnd(1*2*1*4, 10), 1, 2, 1, 4), 2)))
	diffOK(t, "Slice(axis 2)", 0,
		g.Floats(g.Slice(g.FromFloats(rnd(1*2*5*4, 11), 1, 2, 5, 4), 2, 1, 4)),
		c.Floats(c.Slice(c.FromFloats(rnd(1*2*5*4, 11), 1, 2, 5, 4), 2, 1, 4)))

	rp := rnd(1*2*4*8, 12)
	for _, trad := range []bool{false, true} {
		name := "RoPE(neox)"
		if trad {
			name = "RoPE(traditional)"
		}
		diffOK(t, name, tol,
			g.Floats(g.RoPE(g.FromFloats(rp, 1, 2, 4, 8), 8, trad, 10000, 1, 0)),
			c.Floats(c.RoPE(c.FromFloats(rp, 1, 2, 4, 8), 8, trad, 10000, 1, 0)))
	}
	half := 4
	fr := make([]float32, half)
	for i := range fr {
		fr[i] = float32(math.Pow(10000, float64(2*i)/8))
	}
	diffOK(t, "RoPEFreqs(offset=3)", tol,
		g.Floats(g.RoPEFreqs(g.FromFloats(rp, 1, 2, 4, 8), 8, false, 1.0, 3, g.FromFloats(fr, half))),
		c.Floats(c.RoPEFreqs(c.FromFloats(rp, 1, 2, 4, 8), 8, false, 1.0, 3, c.FromFloats(fr, half))))

	// The Qwen3.5 hybrid's two ops: the causal depthwise conv over the packed
	// projection (K=4, as the model's), and the gated delta-rule recurrence
	// with key heads serving adjacent value heads (Hk=2, Hv=4, Dk=Dv=16).
	cx, cw := rnd(1*7*6, 20), rnd(6*4*1, 21)
	diffOK(t, "Conv1dDepthwise", tol,
		g.Floats(g.Conv1dDepthwise(g.FromFloats(cx, 1, 7, 6), g.FromFloats(cw, 6, 4, 1))),
		c.Floats(c.Conv1dDepthwise(c.FromFloats(cx, 1, 7, 6), c.FromFloats(cw, 6, 4, 1))))
	dq, dk, dv := rnd(1*5*2*16, 22), rnd(1*5*2*16, 23), rnd(1*5*4*16, 24)
	dg, db, ds := rnd(1*5*4, 25), rnd(1*5*4, 26), rnd(1*4*16*16, 27)
	for i := range dg {
		dg[i] = 0.5 + 0.5*float32(math.Abs(float64(dg[i]))) // a gate in (0, 1]
		db[i] = 0.5 + 0.5*float32(math.Abs(float64(db[i])))
	}
	dy, dst := g.GatedDeltaScan(g.FromFloats(dq, 1, 5, 2, 16), g.FromFloats(dk, 1, 5, 2, 16), g.FromFloats(dv, 1, 5, 4, 16),
		g.FromFloats(dg, 1, 5, 4), g.FromFloats(db, 1, 5, 4), g.FromFloats(ds, 1, 4, 16, 16))
	cy, cs := c.GatedDeltaScan(c.FromFloats(dq, 1, 5, 2, 16), c.FromFloats(dk, 1, 5, 2, 16), c.FromFloats(dv, 1, 5, 4, 16),
		c.FromFloats(dg, 1, 5, 4), c.FromFloats(db, 1, 5, 4), c.FromFloats(ds, 1, 4, 16, 16))
	diffOK(t, "GatedDeltaScan(y)", tol, g.Floats(dy), c.Floats(cy))
	diffOK(t, "GatedDeltaScan(state)", tol, g.Floats(dst), c.Floats(cs))

	q, k, v := rnd(1*4*5*8, 13), rnd(1*2*5*8, 14), rnd(1*2*5*8, 15)
	scale := float32(1.0 / math.Sqrt(8))
	diffOK(t, "SDPA(causal,GQA)", tol,
		g.Floats(g.SDPA(g.FromFloats(q, 1, 4, 5, 8), g.FromFloats(k, 1, 2, 5, 8), g.FromFloats(v, 1, 2, 5, 8), scale, true)),
		c.Floats(c.SDPA(c.FromFloats(q, 1, 4, 5, 8), c.FromFloats(k, 1, 2, 5, 8), c.FromFloats(v, 1, 2, 5, 8), scale, true)))
	q1 := rnd(1*4*1*8, 16)
	diffOK(t, "SDPA(decode: 1 query over 5 keys)", tol,
		g.Floats(g.SDPA(g.FromFloats(q1, 1, 4, 1, 8), g.FromFloats(k, 1, 2, 5, 8), g.FromFloats(v, 1, 2, 5, 8), scale, true)),
		c.Floats(c.SDPA(c.FromFloats(q1, 1, 4, 1, 8), c.FromFloats(k, 1, 2, 5, 8), c.FromFloats(v, 1, 2, 5, 8), scale, true)))

	const out, in, gs, bits = 8, 128, 64, 4
	P, G := in*bits/32, in/gs
	raw := rndPackedU32(out*P, 17)
	sc, bi := rnd(out*G, 18), rnd(out*G, 19)
	gW, cW := g.FromRaw(engine.U32, raw, out, P), c.FromRaw(engine.U32, raw, out, P)
	gS, gB := g.FromFloats(sc, out, G), g.FromFloats(bi, out, G)
	cS, cB := c.FromFloats(sc, out, G), c.FromFloats(bi, out, G)
	diffOK(t, "Dequantize", tol, g.Floats(g.Dequantize(gW, gS, gB, gs, bits)), c.Floats(c.Dequantize(cW, cS, cB, gs, bits)))
	// The embedding is gathered before it is ever multiplied (the model's
	// order); a weight that is only multiplied keeps its Q4_1 copy alone.
	ids := []int32{0, 3, 5, 2}
	gIdx, cIdx := g.FromInt32(ids, len(ids)), c.FromInt32(ids, len(ids))
	diffOK(t, "QuantEmbed(gather+dequant)", tol,
		g.Floats(g.Dequantize(g.TakeAxis(gW, gIdx, 0), g.TakeAxis(gS, gIdx, 0), g.TakeAxis(gB, gIdx, 0), gs, bits)),
		c.Floats(c.Dequantize(c.TakeAxis(cW, cIdx, 0), c.TakeAxis(cS, cIdx, 0), c.TakeAxis(cB, cIdx, 0), gs, bits)))
	// ggml's Q4_1 kernel rounds the activations to 8 bits per block of 32
	// (q8_1) before the dot, and keeps scale and min in f16: llama.cpp's
	// precision, about half a percent of the output's magnitude, not the
	// oracle's f32 affine. Asserted relative to the largest output.
	xq := rnd(3*in, 20)
	relOK(t, "QuantMatmul(M=3,prefill) on Q4_1", 1e-2,
		g.Floats(g.QuantMatmul(g.FromFloats(xq, 3, in), gW, gS, gB, true, gs, bits)),
		c.Floats(c.QuantMatmul(c.FromFloats(xq, 3, in), cW, cS, cB, true, gs, bits)))
	xq1 := rnd(in, 21)
	relOK(t, "QuantMatmul(M=1,decode) on Q4_1", 1e-2,
		g.Floats(g.QuantMatmul(g.FromFloats(xq1, 1, in), gW, gS, gB, true, gs, bits)),
		c.Floats(c.QuantMatmul(c.FromFloats(xq1, 1, in), cW, cS, cB, true, gs, bits)))

	am := rnd(3*9, 22)
	gi, ci := g.Ints(g.Argmax(g.FromFloats(am, 3, 9), -1)), c.Ints(c.Argmax(c.FromFloats(am, 3, 9), -1))
	for i := range ci {
		if gi[i] != ci[i] {
			t.Errorf("Argmax row %d: ggml %d cpu %d", i, gi[i], ci[i])
		}
	}
}
