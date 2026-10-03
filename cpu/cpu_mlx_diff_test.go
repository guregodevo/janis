//go:build darwin && arm64

// This harness is the oracle for the pure-Go backend: it runs identical f32
// inputs through BOTH the CPU backend and the MLX backend and asserts the
// outputs match. MLX is the reference the CPU impl must reproduce — especially
// for RoPE and SDPA, whose conventions are easy to get subtly wrong. Runs only
// on Apple Silicon (where MLX links). On Linux the CPU backend stands alone.
package cpu

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/guregodevo/janis/engine"
	"github.com/guregodevo/janis/mlxc"
)

func rnd(n int) []float32 {
	r := rand.New(rand.NewSource(1)) // fixed seed → both backends get identical data
	d := make([]float32, n)
	for i := range d {
		d[i] = r.Float32()*2 - 1
	}
	return d
}

// rndSeed is rnd with a chosen seed, so independent operands (e.g. scales vs
// biases) get genuinely different data and a swap would be caught.
func rndSeed(n int, seed int64) []float32 {
	r := rand.New(rand.NewSource(seed))
	d := make([]float32, n)
	for i := range d {
		d[i] = r.Float32()*2 - 1
	}
	return d
}

// rndPackedU32 returns n random packed words as little-endian bytes — the wire
// form of an MLX-quantized weight, fed verbatim to both backends' FromRaw.
func rndPackedU32(n int, seed int64) []byte {
	r := rand.New(rand.NewSource(seed))
	raw := make([]byte, n*4)
	for i := 0; i < n; i++ {
		binary.LittleEndian.PutUint32(raw[i*4:], r.Uint32())
	}
	return raw
}

func maxAbsDiff(a, b []float32) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}
	var m float64
	for i := range a {
		if d := math.Abs(float64(a[i]) - float64(b[i])); d > m {
			m = d
		}
	}
	return m
}

// runs op(b) on both backends from the same float input(s) and compares.
func diffOK(t *testing.T, name string, tol float64, cpuOut, mlxOut []float32) {
	t.Helper()
	if d := maxAbsDiff(cpuOut, mlxOut); d > tol {
		t.Errorf("%s: max|cpu-mlx| = %g > tol %g", name, d, tol)
	} else {
		t.Logf("%s: max|cpu-mlx| = %g (ok)", name, d)
	}
}

func TestCPUMatchesMLX(t *testing.T) {
	c := New()
	m := mlxc.New()
	defer m.Close()
	const tol = 2e-3 // f32 both sides; small accumulation-order differences

	// MatMul: [2,3,4] @ [2,4,5]
	a, w := rnd(2*3*4), rnd(2*4*5)
	diffOK(t, "MatMul", tol,
		c.Floats(c.MatMul(c.FromFloats(a, 2, 3, 4), c.FromFloats(w, 2, 4, 5))),
		m.Floats(m.MatMul(m.FromFloats(a, 2, 3, 4), m.FromFloats(w, 2, 4, 5))))

	// Softmax along last axis of [4,7]
	s := rnd(4 * 7)
	diffOK(t, "Softmax", tol,
		c.Floats(c.Softmax(c.FromFloats(s, 4, 7), -1)),
		m.Floats(m.Softmax(m.FromFloats(s, 4, 7), -1)))

	// RMSNorm over last dim 8
	x, rw := rnd(3*8), rnd(8)
	diffOK(t, "RMSNorm", tol,
		c.Floats(c.RMSNorm(c.FromFloats(x, 3, 8), c.FromFloats(rw, 8), 1e-5)),
		m.Floats(m.RMSNorm(m.FromFloats(x, 3, 8), m.FromFloats(rw, 8), 1e-5)))

	// LayerNorm over last dim 8
	lw, lb := rnd(8), rnd(8)
	diffOK(t, "LayerNorm", tol,
		c.Floats(c.LayerNorm(c.FromFloats(x, 3, 8), c.FromFloats(lw, 8), c.FromFloats(lb, 8), 1e-5)),
		m.Floats(m.LayerNorm(m.FromFloats(x, 3, 8), m.FromFloats(lw, 8), m.FromFloats(lb, 8), 1e-5)))

	// RoPE — the convention check. [B=1, H=2, L=4, Dh=8], rotate all 8 dims.
	rp := rnd(1 * 2 * 4 * 8)
	for _, trad := range []bool{false, true} {
		name := "RoPE(neox)"
		if trad {
			name = "RoPE(traditional)"
		}
		diffOK(t, name, tol,
			c.Floats(c.RoPE(c.FromFloats(rp, 1, 2, 4, 8), 8, trad, 10000, 1, 0)),
			m.Floats(m.RoPE(m.FromFloats(rp, 1, 2, 4, 8), 8, trad, 10000, 1, 0)))
	}

	// SDPA causal, GQA (Hq=4, Hk=2), [B=1, H, L=5, Dh=8]
	q := rnd(1 * 4 * 5 * 8)
	k := rnd(1 * 2 * 5 * 8)
	v := rnd(1 * 2 * 5 * 8)
	scale := float32(1.0 / math.Sqrt(8))
	diffOK(t, "SDPA(causal,GQA)", tol,
		c.Floats(c.SDPA(c.FromFloats(q, 1, 4, 5, 8), c.FromFloats(k, 1, 2, 5, 8), c.FromFloats(v, 1, 2, 5, 8), scale, true)),
		m.Floats(m.SDPA(m.FromFloats(q, 1, 4, 5, 8), m.FromFloats(k, 1, 2, 5, 8), m.FromFloats(v, 1, 2, 5, 8), scale, true)))

	// 4-bit affine quantization — the convention that lets a Qwen3-8B-4bit MLX
	// checkpoint run on the CPU backend. weight [out=8, P=16] packed = [8,128]
	// dequantized; groupSize 64, 4 bits → 2 groups/row. Identical packed bytes +
	// f32 scales/biases go to both backends.
	const out, in, gs, bits = 8, 128, 64, 4
	P, G := in*bits/32, in/gs
	raw := rndPackedU32(out*P, 7)
	sc, bi := rndSeed(out*G, 2), rndSeed(out*G, 3)
	cW, mW := c.FromRaw(engine.U32, raw, out, P), m.FromRaw(engine.U32, raw, out, P)
	cS, cB := c.FromFloats(sc, out, G), c.FromFloats(bi, out, G)
	mS, mB := m.FromFloats(sc, out, G), m.FromFloats(bi, out, G)

	diffOK(t, "Dequantize", tol,
		c.Floats(c.Dequantize(cW, cS, cB, gs, bits)),
		m.Floats(m.Dequantize(mW, mS, mB, gs, bits)))

	// QuantMatmul transpose=true (QuantizedLinear): x[3,128] @ Wᵀ -> [3,8].
	// rows=3 exercises the general/prefill path.
	xq := rnd(3 * in)
	diffOK(t, "QuantMatmul(M=3,prefill)", tol,
		c.Floats(c.QuantMatmul(c.FromFloats(xq, 3, in), cW, cS, cB, true, gs, bits)),
		m.Floats(m.QuantMatmul(m.FromFloats(xq, 3, in), mW, mS, mB, true, gs, bits)))

	// rows=1 exercises the DECODE fast path (the fused, word-at-a-time loop) —
	// the hot path at generation time. Must match MLX too.
	xq1 := rnd(in)
	diffOK(t, "QuantMatmul(M=1,decode)", tol,
		c.Floats(c.QuantMatmul(c.FromFloats(xq1, 1, in), cW, cS, cB, true, gs, bits)),
		m.Floats(m.QuantMatmul(m.FromFloats(xq1, 1, in), mW, mS, mB, true, gs, bits)))

	// Quantized embedding lookup: gather packed rows (TakeAxis over U32) +
	// per-row scales/biases, then dequantize — the QuantEmbedding.Forward path.
	ids := []int32{0, 3, 5, 2}
	cIdx, mIdx := c.FromInt32(ids, len(ids)), m.FromInt32(ids, len(ids))
	diffOK(t, "QuantEmbed(gather+dequant)", tol,
		c.Floats(c.Dequantize(c.TakeAxis(cW, cIdx, 0), c.TakeAxis(cS, cIdx, 0), c.TakeAxis(cB, cIdx, 0), gs, bits)),
		m.Floats(m.Dequantize(m.TakeAxis(mW, mIdx, 0), m.TakeAxis(mS, mIdx, 0), m.TakeAxis(mB, mIdx, 0), gs, bits)))

	// RoPEFreqs — the llama3-scaling path (explicit denominators base^(2i/d),
	// nonzero offset). This is what the plain-RoPE unit test missed.
	rf := rnd(1 * 2 * 4 * 8)
	half := 8 / 2
	fr := make([]float32, half)
	for i := range fr {
		fr[i] = float32(math.Pow(10000, float64(2*i)/8))
	}
	diffOK(t, "RoPEFreqs(offset=3)", tol,
		c.Floats(c.RoPEFreqs(c.FromFloats(rf, 1, 2, 4, 8), 8, false, 1.0, 3, c.FromFloats(fr, half))),
		m.Floats(m.RoPEFreqs(m.FromFloats(rf, 1, 2, 4, 8), 8, false, 1.0, 3, m.FromFloats(fr, half))))
}

var _ = engine.F32 // keep the engine import even if assertions change

// The Gated DeltaNet ops (Qwen3.5 hybrid): elementwise gates and the depthwise
// causal-conv front-end must match MLX exactly, or the recurrence drifts.
func TestCPUMatchesMLXDeltaNetOps(t *testing.T) {
	c := New()
	m := mlxc.New()
	defer m.Close()
	const tol = 2e-3

	u := rnd(3 * 5)
	for _, op := range []struct {
		name string
		f    func(b engine.Backend, x engine.Tensor) engine.Tensor
	}{
		{"Exp", func(b engine.Backend, x engine.Tensor) engine.Tensor { return b.Exp(x) }},
		{"Sigmoid", func(b engine.Backend, x engine.Tensor) engine.Tensor { return b.Sigmoid(x) }},
		{"Softplus", func(b engine.Backend, x engine.Tensor) engine.Tensor { return b.Softplus(x) }},
	} {
		diffOK(t, op.name, tol,
			c.Floats(op.f(c, c.FromFloats(u, 3, 5))),
			m.Floats(op.f(m, m.FromFloats(u, 3, 5))))
	}

	// Depthwise conv: x [B=2, L=7, C=6], w [C=6, K=4, 1] -> [2, 4, 6]
	xc, wc := rnd(2*7*6), rndSeed(6*4, 7)
	cOut := c.Floats(c.Conv1dDepthwise(c.FromFloats(xc, 2, 7, 6), c.FromFloats(wc, 6, 4, 1)))
	mOut := m.Floats(m.Conv1dDepthwise(m.FromFloats(xc, 2, 7, 6), m.FromFloats(wc, 6, 4, 1)))
	diffOK(t, "Conv1dDepthwise", tol, cOut, mOut)
	if got := len(cOut); got != 2*4*6 {
		t.Errorf("Conv1dDepthwise output size = %d, want %d", got, 2*4*6)
	}
}

// The fused metal GatedDeltaScan must reproduce the sequential pure-Go
// recurrence: same y, same final state, including a warm (nonzero) incoming
// state and Hv > Hk head mapping.
func TestCPUMatchesMLXGatedDeltaScan(t *testing.T) {
	c := New()
	m := mlxc.New()
	defer m.Close()
	const B, T, Hk, Dk, Hv, Dv = 1, 5, 2, 32, 4, 32
	const tol = 2e-3

	q, k := rndSeed(B*T*Hk*Dk, 1), rndSeed(B*T*Hk*Dk, 2)
	v := rndSeed(B*T*Hv*Dv, 3)
	// g in (0,1) like a real decay, beta in (0,1) like a real sigmoid.
	g, beta := rndSeed(B*T*Hv, 4), rndSeed(B*T*Hv, 5)
	for i := range g {
		g[i] = 0.5 + g[i]/4
		beta[i] = 0.5 + beta[i]/4
	}
	st := rndSeed(B*Hv*Dv*Dk, 6)

	run := func(b engine.Backend) ([]float32, []float32) {
		y, ns := b.GatedDeltaScan(
			b.FromFloats(q, B, T, Hk, Dk), b.FromFloats(k, B, T, Hk, Dk),
			b.FromFloats(v, B, T, Hv, Dv), b.FromFloats(g, B, T, Hv),
			b.FromFloats(beta, B, T, Hv), b.FromFloats(st, B, Hv, Dv, Dk))
		return b.Floats(y), b.Floats(ns)
	}
	cy, cs := run(c)
	my, ms := run(m)
	diffOK(t, "GatedDeltaScan.y", tol, cy, my)
	diffOK(t, "GatedDeltaScan.state", tol, cs, ms)
}
