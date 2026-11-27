package llm

import (
	"math"
	"testing"
)

func TestProjectRemovesTheDirection(t *testing.T) {
	// W is 3x2; u = (1, 1, 0) unnormalized.
	v := []float32{1, 2, 3, 4, 5, 6}
	if err := project(v, []int{3, 2}, []float32{1, 1, 0}, 1); err != nil {
		t.Fatal(err)
	}
	// ûᵀW = ((1+3)/√2, (2+4)/√2); û·ûᵀW subtracts (2,3) from rows 0 and 1.
	want := []float32{-1, -1, 1, 1, 5, 6}
	for i := range want {
		if math.Abs(float64(v[i]-want[i])) > 1e-5 {
			t.Fatalf("v = %v, want %v", v, want)
		}
	}
	// What the matrix writes along û is now zero, for every input.
	for c := 0; c < 2; c++ {
		if s := v[0*2+c] + v[1*2+c]; math.Abs(float64(s)) > 1e-5 {
			t.Fatalf("column %d still writes %v along u", c, s)
		}
	}
}

func TestProjectAlphaZeroChangesNothing(t *testing.T) {
	v := []float32{1, 2, 3, 4}
	if err := project(v, []int{2, 2}, []float32{3, 4}, 0); err != nil {
		t.Fatal(err)
	}
	for i, want := range []float32{1, 2, 3, 4} {
		if v[i] != want {
			t.Fatalf("v = %v", v)
		}
	}
}

func TestProjectRefuses(t *testing.T) {
	for name, c := range map[string]struct {
		shape []int
		u     []float32
		alpha float64
	}{
		"a vector":       {[]int{4}, []float32{1, 0, 0, 0}, 1},
		"u of the input": {[]int{2, 3}, []float32{1, 0, 0}, 1},
		"zero u":         {[]int{2, 2}, []float32{0, 0}, 1},
		"NaN alpha":      {[]int{2, 2}, []float32{1, 0}, math.NaN()},
		"Inf in u":       {[]int{2, 2}, []float32{float32(math.Inf(1)), 0}, 1},
	} {
		if err := project(make([]float32, 4), c.shape, c.u, c.alpha); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
}
