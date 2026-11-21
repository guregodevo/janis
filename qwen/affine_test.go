package qwen

import (
	"math"
	"math/rand"
	"testing"

	"github.com/guregodevo/janis/cpu"
	"github.com/guregodevo/janis/safetensors"
)

// TestQuantizeAffineDequantizes checks the packed triple against the CPU
// backend's Dequantize, the reader of MLX's layout every backend matches.
func TestQuantizeAffineDequantizes(t *testing.T) {
	b := cpu.New()
	rows, cols, gs := 6, 128, 64
	rng := rand.New(rand.NewSource(1))
	v := make([]float32, rows*cols)
	for i := range v {
		v[i] = float32(rng.NormFloat64()) * 0.02
	}
	for i := cols; i < 2*cols; i++ {
		v[i] = 0.5 // a constant row
	}
	for _, bits := range []int{2, 4, 8} {
		packed, scales, biases, err := QuantizeAffine(v, rows, cols, gs, bits, "BF16")
		if err != nil {
			t.Fatal(err)
		}
		w, _ := TensorFrom(b, safetensors.Entry{Dtype: "U32", Shape: []int{rows, cols * bits / 32}, Raw: packed})
		s, _ := TensorFrom(b, safetensors.Entry{Dtype: "BF16", Shape: []int{rows, cols / gs}, Raw: scales})
		bi, _ := TensorFrom(b, safetensors.Entry{Dtype: "BF16", Shape: []int{rows, cols / gs}, Raw: biases})
		got := b.Floats(b.Dequantize(w, s, bi, gs, bits))
		sc, _ := safetensors.DecodeFloats("BF16", scales)
		for i, x := range v {
			step := sc[(i/cols)*(cols/gs)+(i%cols)/gs]
			// half a step, and the bias's own bf16 rounding below the minimum
			if tol := step/2 + float32(math.Abs(float64(x)))/128 + 1e-7; float32(math.Abs(float64(got[i]-x))) > tol {
				t.Fatalf("%d bits: value %d is %v, quantized %v (step %v)", bits, i, x, got[i], step)
			}
		}
		for i := cols; i < 2*cols; i++ {
			if got[i] != 0.5 {
				t.Fatalf("%d bits: the constant row came back %v", bits, got[i])
			}
		}
	}
}

func TestQuantizeAffineRefuses(t *testing.T) {
	v := make([]float32, 64)
	for _, c := range []struct {
		name                 string
		v                    []float32
		rows, cols, gs, bits int
	}{
		{"3 bits", v, 1, 64, 64, 3},
		{"columns not in groups", v, 1, 64, 48, 4},
		{"wrong length", v, 2, 64, 64, 4},
		{"NaN", append([]float32{float32(math.NaN())}, v[1:]...), 1, 64, 64, 4},
	} {
		if _, _, _, err := QuantizeAffine(c.v, c.rows, c.cols, c.gs, c.bits, "BF16"); err == nil {
			t.Errorf("%s: quantized", c.name)
		}
	}
}
