package safetensors

import (
	"math"
	"testing"
)

func TestHalfFloatsRoundToNearestEven(t *testing.T) {
	cases := []struct {
		dtype string
		in    float32
		want  float32
	}{
		{"BF16", 1, 1},
		{"BF16", -2.5, -2.5},
		{"BF16", 1 + 1.0/256, 1},           // a tie, to the even 1
		{"BF16", 1 + 3.0/256, 1 + 4.0/256}, // a tie, to the even 1+2^-6
		{"BF16", 1 + 1.0/256 + 1e-6, 1 + 1.0/128},
		{"F16", 1 + 1.0/2048, 1}, // a tie, to the even 1
		{"F16", 1 + 3.0/2048, 1 + 4.0/2048},
		{"F16", 65504, 65504},
		{"F16", 1e5, float32(math.Inf(1))},
		{"F16", 1.0 / (1 << 24), 1.0 / (1 << 24)}, // the smallest subnormal
		{"F16", 3.0 / (1 << 26), 1.0 / (1 << 24)},
		{"F16", 1.0 / (1 << 26), 0},
		{"F16", -0.000123, -0.00012302399},
	}
	for _, c := range cases {
		if got := RoundFloat(c.dtype, c.in); got != c.want {
			t.Errorf("RoundFloat(%s, %v) = %v, want %v", c.dtype, c.in, got, c.want)
		}
	}
}

func TestEncodeDecodeFloats(t *testing.T) {
	v := []float32{0, 1, -1, 0.1, 3.14159, -1e-3, 1234.5}
	for _, dt := range []string{"F32", "F16", "BF16"} {
		raw, err := EncodeFloats(dt, v)
		if err != nil {
			t.Fatal(err)
		}
		back, err := DecodeFloats(dt, raw)
		if err != nil {
			t.Fatal(err)
		}
		for i := range v {
			if back[i] != RoundFloat(dt, v[i]) {
				t.Errorf("%s: %v came back %v, want %v", dt, v[i], back[i], RoundFloat(dt, v[i]))
			}
		}
	}
	if _, err := EncodeFloats("U32", v); err == nil {
		t.Error("U32 is not a float dtype")
	}
}
