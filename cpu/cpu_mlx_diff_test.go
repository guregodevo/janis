//go:build darwin && arm64

// This harness is the oracle for the pure-Go backend: it runs identical f32
// inputs through BOTH the CPU backend and the MLX backend and asserts the
// outputs match. MLX is the reference the CPU impl must reproduce — especially
// for RoPE and SDPA, whose conventions are easy to get subtly wrong. Runs only
// on Apple Silicon (where MLX links). On Linux the CPU backend stands alone.
package cpu

import (
	"math"
	"math/rand"
	"testing"

	"memdoor/llm/engine"
	"memdoor/llm/mlxc"
)

func rnd(n int) []float32 {
	r := rand.New(rand.NewSource(1)) // fixed seed → both backends get identical data
	d := make([]float32, n)
	for i := range d {
		d[i] = r.Float32()*2 - 1
	}
	return d
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
}

var _ = engine.F32 // keep the engine import even if assertions change
