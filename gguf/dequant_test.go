package gguf

import (
	"math"
	"testing"
)

// f16bits are exact half-precision words for the values the block tests use, so
// the tests do not depend on a float32->float16 converter being right.
const (
	hOne     = 0x3C00 // 1.0
	hHalf    = 0x3800 // 0.5
	hQuarter = 0x3400 // 0.25
	hTwo     = 0x4000 // 2.0
	hNegOne  = 0xBC00 // -1.0
)

func le16(v uint16) []byte { return []byte{byte(v), byte(v >> 8)} }
func le32(v uint32) []byte {
	return []byte{byte(v), byte(v >> 8), byte(v >> 16), byte(v >> 24)}
}

// TestDequantizeBlockLayouts pins the bit packing of each type against values
// computed by hand. This is the test that catches a wrong nibble order or a
// missing sign: those produce numbers that still look like plausible weights, so
// nothing downstream would notice.
func TestDequantizeBlockLayouts(t *testing.T) {
	t.Run("F32", func(t *testing.T) {
		raw := append(le32(math.Float32bits(1.5)), le32(math.Float32bits(-2.25))...)
		got, err := Dequantize(TensorF32, raw)
		if err != nil {
			t.Fatal(err)
		}
		want := []float32{1.5, -2.25}
		for i := range want {
			if got[i] != want[i] {
				t.Errorf("F32[%d] = %v, want %v", i, got[i], want[i])
			}
		}
	})

	t.Run("F16 and BF16", func(t *testing.T) {
		got, err := Dequantize(TensorF16, append(le16(hOne), le16(hNegOne)...))
		if err != nil {
			t.Fatal(err)
		}
		if got[0] != 1 || got[1] != -1 {
			t.Errorf("F16 = %v, want [1 -1]", got)
		}
		// bf16 is the top half of an f32: 0x3F80 is 1.0, 0xC000 is -2.0.
		got, err = Dequantize(TensorBF16, append(le16(0x3F80), le16(0xC000)...))
		if err != nil {
			t.Fatal(err)
		}
		if got[0] != 1 || got[1] != -2 {
			t.Errorf("BF16 = %v, want [1 -2]", got)
		}
	})

	t.Run("Q8_0 signed and scaled", func(t *testing.T) {
		raw := le16(hQuarter)
		for i := 0; i < 32; i++ {
			raw = append(raw, byte(int8(i-16))) // -16..15, incl. negatives
		}
		got, err := Dequantize(TensorQ8_0, raw)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != 32 {
			t.Fatalf("got %d values, want 32", len(got))
		}
		for i := 0; i < 32; i++ {
			want := float32(0.25 * float64(i-16))
			if got[i] != want {
				t.Errorf("Q8_0[%d] = %v, want %v", i, got[i], want)
			}
		}
	})

	t.Run("Q4_0 nibble order", func(t *testing.T) {
		// 16 payload bytes: byte j holds the LOW nibble of value j and the HIGH
		// nibble of value j+16. Make every value distinct so a swapped order or a
		// reversed nibble shows up.
		raw := le16(hHalf)
		for j := 0; j < 16; j++ {
			raw = append(raw, byte(j)|byte(15-j)<<4)
		}
		got, err := Dequantize(TensorQ4_0, raw)
		if err != nil {
			t.Fatal(err)
		}
		for j := 0; j < 16; j++ {
			if want := float32(0.5) * float32(j-8); got[j] != want {
				t.Errorf("Q4_0[%d] = %v, want %v (low nibble of byte %d)", j, got[j], want, j)
			}
			if want := float32(0.5) * float32((15-j)-8); got[j+16] != want {
				t.Errorf("Q4_0[%d] = %v, want %v (high nibble of byte %d)", j+16, got[j+16], want, j)
			}
		}
	})

	t.Run("Q4_1 offset", func(t *testing.T) {
		raw := append(le16(hHalf), le16(hOne)...) // d=0.5, m=1
		for j := 0; j < 16; j++ {
			raw = append(raw, byte(j))
		}
		got, err := Dequantize(TensorQ4_1, raw)
		if err != nil {
			t.Fatal(err)
		}
		for j := 0; j < 16; j++ {
			if want := float32(0.5)*float32(j) + 1; got[j] != want {
				t.Errorf("Q4_1[%d] = %v, want %v", j, got[j], want)
			}
		}
	})

	t.Run("Q5_0 fifth bit plane", func(t *testing.T) {
		// Byte j holds the low nibble of value j and the high nibble of value
		// j+16, and qh's bit i is the 5th bit of value i. So the 5-bit code is
		// q(i) = low_nibble(i) + 16*bit_i, and the value is d*(q-16): the whole
		// field is offset by 16, so a set bit alone still yields 0.
		// Set the 5th bit at i = 0, 3, 20 and 31 only.
		qh := uint32(1) | uint32(1)<<3 | uint32(1)<<20 | uint32(1)<<31
		raw := append(le16(hOne), le32(qh)...)
		for j := 0; j < 16; j++ {
			raw = append(raw, byte(j)|byte(j)<<4)
		}
		got, err := Dequantize(TensorQ5_0, raw)
		if err != nil {
			t.Fatal(err)
		}
		// each entry: index, 5-bit code, expected value with d=1
		for _, tc := range []struct {
			i, q int
		}{
			{0, 0 + 16}, {1, 1}, {3, 3 + 16}, {15, 15},
			{16, 0}, {20, 4 + 16}, {30, 14}, {31, 15 + 16},
		} {
			if want := float32(tc.q - 16); got[tc.i] != want {
				t.Errorf("Q5_0[%d] = %v, want %v (5-bit code %d)", tc.i, got[tc.i], want, tc.q)
			}
		}
	})

	t.Run("Q5_1 min offset", func(t *testing.T) {
		// Same packing, plus a minimum: value = d*q + m. With d=0.5 and m=-1 the
		// result differs from Q5_0's (q-16) by a factor and offset, so a reader
		// that ignored the min field would be caught here.
		qh := uint32(1) << 2 // 5th bit on value 2 only
		raw := append(le16(hHalf), le16(hNegOne)...)
		raw = append(raw, le32(qh)...)
		for j := 0; j < 16; j++ {
			raw = append(raw, byte(j))
		}
		got, err := Dequantize(TensorQ5_1, raw)
		if err != nil {
			t.Fatal(err)
		}
		for _, tc := range []struct{ i, q int }{{0, 0}, {2, 2 + 16}, {16, 0}} {
			if want := float32(0.5)*float32(tc.q) - 1; got[tc.i] != want {
				t.Errorf("Q5_1[%d] = %v, want %v (code %d)", tc.i, got[tc.i], want, tc.q)
			}
		}
	})

	t.Run("Q8_1 skips the sum field", func(t *testing.T) {
		raw := append(le16(hQuarter), le16(hTwo)...) // scale, sum (unused)
		for i := 0; i < 32; i++ {
			raw = append(raw, byte(int8(i)))
		}
		got, err := Dequantize(TensorQ8_1, raw)
		if err != nil {
			t.Fatal(err)
		}
		if got[0] != 0 || got[1] != 0.25 {
			t.Errorf("Q8_1 = %v…, want 0 then 0.25 (payload starts after the sum)", got[:2])
		}
	})

	t.Run("rejects block-misaligned and unknown types", func(t *testing.T) {
		if _, err := Dequantize(TensorQ4_0, make([]byte, 17)); err == nil {
			t.Error("17 bytes is not a Q4_0 block; expected an error")
		}
		if _, err := Dequantize(TensorQ8_1+1, make([]byte, 8)); err == nil {
			t.Error("unknown type; expected an error")
		}
	})
}

// TestF16Widening covers the half-precision path, including the two cases a
// naive implementation gets wrong: subnormals and the infinities.
func TestF16Widening(t *testing.T) {
	for _, tc := range []struct {
		bits uint16
		want float32
	}{
		{0x0000, 0},
		{0x8000, float32(math.Copysign(0, -1))},
		{hOne, 1},
		{hNegOne, -1},
		{hHalf, 0.5},
		{0x0001, float32(math.Pow(2, -24))}, // smallest subnormal
		{0x03FF, float32(1023) * float32(math.Pow(2, -24))}, // largest subnormal
		{0x0400, float32(math.Pow(2, -14))},                 // smallest normal
		{0x7BFF, 65504},                                     // largest finite half
		{0x7C00, float32(math.Inf(1))},
	} {
		got := f16(tc.bits)
		if math.Float32bits(got) != math.Float32bits(tc.want) {
			t.Errorf("f16(%#04x) = %v (%#x), want %v (%#x)", tc.bits, got,
				math.Float32bits(got), tc.want, math.Float32bits(tc.want))
		}
	}
	if got := f16(0x7E00); !math.IsNaN(float64(got)) {
		t.Errorf("f16(0x7E00) = %v, want NaN", got)
	}
}
