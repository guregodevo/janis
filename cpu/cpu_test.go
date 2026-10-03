package cpu

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/guregodevo/janis/engine"
)

func approx(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-4 }

func vals(t *testing.T, x interface{ Shape() []int }) []float32 {
	t.Helper()
	return x.(*tensor).data
}

func TestMatMul(t *testing.T) {
	b := New()
	// [2x3] @ [3x2] = [2x2]
	a := b.FromFloats([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	w := b.FromFloats([]float32{1, 0, 0, 1, 1, 1}, 3, 2)
	got := vals(t, b.MatMul(a, w))
	want := []float32{4, 5, 10, 11}
	for i := range want {
		if !approx(got[i], want[i]) {
			t.Fatalf("MatMul[%d] = %v, want %v (%v)", i, got[i], want[i], got)
		}
	}
}

func TestMatMulBatched(t *testing.T) {
	b := New()
	// batch=2, each [2x2] identity-ish
	a := b.FromFloats([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 2, 2, 2)
	id := b.FromFloats([]float32{1, 0, 0, 1}, 2, 2) // broadcasts over the batch
	got := vals(t, b.MatMul(a, id))
	want := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	for i := range want {
		if !approx(got[i], want[i]) {
			t.Fatalf("batched MatMul[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestSoftmax(t *testing.T) {
	b := New()
	got := vals(t, b.Softmax(b.FromFloats([]float32{1, 2, 3}, 3), 0))
	var sum float32
	for i, v := range got {
		sum += v
		if i > 0 && v <= got[i-1] {
			t.Fatalf("softmax not increasing: %v", got)
		}
	}
	if !approx(sum, 1) {
		t.Fatalf("softmax sum = %v, want 1", sum)
	}
	if !approx(got[0], 0.09003057) {
		t.Fatalf("softmax[0] = %v, want 0.09003", got[0])
	}
}

func TestRMSNorm(t *testing.T) {
	b := New()
	x := b.FromFloats([]float32{3, 4}, 2)
	w := b.FromFloats([]float32{1, 1}, 2)
	got := vals(t, b.RMSNorm(x, w, 0))
	// rms = sqrt((9+16)/2) = 3.53553; out = x/rms
	want := []float32{3.0 / 3.5355339, 4.0 / 3.5355339}
	for i := range want {
		if !approx(got[i], want[i]) {
			t.Fatalf("RMSNorm[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestAddBroadcast(t *testing.T) {
	b := New()
	// [2x3] + [3] (bias) broadcasts the bias across rows
	x := b.FromFloats([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	bias := b.FromFloats([]float32{10, 20, 30}, 3)
	got := vals(t, b.Add(x, bias))
	want := []float32{11, 22, 33, 14, 25, 36}
	for i := range want {
		if !approx(got[i], want[i]) {
			t.Fatalf("Add bcast[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestTransposeConcatSlice(t *testing.T) {
	b := New()
	x := b.FromFloats([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	tr := vals(t, b.Transpose(x, 1, 0)) // -> [3x2] = 1,4,2,5,3,6
	if want := []float32{1, 4, 2, 5, 3, 6}; !eq(tr, want) {
		t.Fatalf("Transpose = %v, want %v", tr, want)
	}
	cat := vals(t, b.Concat(b.FromFloats([]float32{1, 2}, 1, 2), b.FromFloats([]float32{3, 4}, 1, 2), 1))
	if want := []float32{1, 2, 3, 4}; !eq(cat, want) {
		t.Fatalf("Concat = %v, want %v", cat, want)
	}
	sl := vals(t, b.Slice(x, 1, 1, 3)) // cols [1:3] of [2x3] -> [2x2] = 2,3,5,6
	if want := []float32{2, 3, 5, 6}; !eq(sl, want) {
		t.Fatalf("Slice = %v, want %v", sl, want)
	}
}

func eq(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !approx(a[i], b[i]) {
			return false
		}
	}
	return true
}

// TestDequantize4bit checks the MLX-affine 4-bit path with a hand-packed word,
// so Linux (where the MLX diff harness can't run) still covers the quant path.
// nibbles q = [1..8] packed LSB-first -> 0x87654321; groupSize 4 -> 2 groups.
func TestDequantize4bit(t *testing.T) {
	b := New()
	raw := make([]byte, 4)
	binary.LittleEndian.PutUint32(raw, 0x87654321) // q0=1 (low nibble) .. q7=8
	w := b.FromRaw(engine.U32, raw, 1, 1)          // [rows=1, P=1]
	scales := b.FromFloats([]float32{2, 0.1}, 1, 2)
	biases := b.FromFloats([]float32{0.5, -1}, 1, 2)

	got := vals(t, b.Dequantize(w, scales, biases, 4 /*group*/, 4 /*bits*/))
	want := []float32{2.5, 4.5, 6.5, 8.5, -0.5, -0.4, -0.3, -0.2} // q*scale+bias per group
	if len(got) != len(want) {
		t.Fatalf("Dequantize len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if !approx(got[i], want[i]) {
			t.Fatalf("Dequantize[%d] = %v, want %v (%v)", i, got[i], want[i], got)
		}
	}

	// QuantMatmul transpose=true: x=ones[1,8] @ Wᵀ -> sum of the dequantized row.
	x := b.FromFloats([]float32{1, 1, 1, 1, 1, 1, 1, 1}, 1, 8)
	y := vals(t, b.QuantMatmul(x, w, scales, biases, true, 4, 4))
	if !approx(y[0], 20.6) { // 2.5+4.5+6.5+8.5-0.5-0.4-0.3-0.2
		t.Fatalf("QuantMatmul = %v, want 20.6", y[0])
	}
}
