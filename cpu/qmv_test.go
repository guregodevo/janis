package cpu

import (
	"math"
	"math/rand"
	"testing"
)

// qChannel4Ref is the scalar reference the NEON kernel must reproduce: one 4-bit
// quantized output channel. Mirrors QuantMatmul's per-group math exactly.
func qChannel4Ref(w []uint32, x, scales, biases []float32, groups, wordsPerGroup int) float32 {
	var acc float32
	xi, wp := 0, 0
	for g := 0; g < groups; g++ {
		var sumXQ, sumX float32
		for wc := 0; wc < wordsPerGroup; wc++ {
			word := w[wp]
			wp++
			for v := 0; v < 8; v++ {
				xv := x[xi]
				xi++
				sumXQ += xv * float32((word>>uint(v*4))&0xF)
				sumX += xv
			}
		}
		acc += scales[g]*sumXQ + biases[g]*sumX
	}
	return acc
}

// TestQChannel4Accel checks the SIMD kernel (NEON on arm64) against the scalar
// reference over a range of shapes. On non-arm64 qChannel4Accel returns
// ok=false and the test is a no-op (the portable Go path is used there).
func TestQChannel4Accel(t *testing.T) {
	r := rand.New(rand.NewSource(42))
	shapes := []struct{ groups, wordsPerGroup int }{
		{1, 1}, {1, 8}, {2, 8}, {4, 8}, {32, 8}, {16, 2}, {8, 16}, {3, 4},
	}
	for _, s := range shapes {
		nWords := s.groups * s.wordsPerGroup
		in := nWords * 8
		w := make([]uint32, nWords)
		for i := range w {
			w[i] = r.Uint32()
		}
		x := make([]float32, in)
		for i := range x {
			x[i] = r.Float32()*2 - 1
		}
		scales := make([]float32, s.groups)
		biases := make([]float32, s.groups)
		for i := range scales {
			scales[i] = r.Float32()*0.1 + 0.01
			biases[i] = r.Float32()*0.2 - 0.1
		}

		got, ok := qChannel4Accel(&w[0], &x[0], &scales[0], &biases[0], s.groups, s.wordsPerGroup)
		if !ok {
			t.Skip("no SIMD kernel on this arch (portable Go path used)")
		}
		want := qChannel4Ref(w, x, scales, biases, s.groups, s.wordsPerGroup)
		if d := math.Abs(float64(got - want)); d > 1e-3*(1+math.Abs(float64(want))) {
			t.Errorf("shape{g=%d,wpg=%d}: kernel=%g ref=%g |diff|=%g",
				s.groups, s.wordsPerGroup, got, want, d)
		}
	}
}
