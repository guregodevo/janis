package safetensors

import (
	"encoding/binary"
	"fmt"
	"math"
)

// EncodeFloats packs v as little-endian dtype ("F32", "F16" or "BF16"), each
// value rounded to the nearest representable one, ties to even.
func EncodeFloats(dtype string, v []float32) ([]byte, error) {
	switch dtype {
	case "F32":
		out := make([]byte, 4*len(v))
		for i, f := range v {
			binary.LittleEndian.PutUint32(out[4*i:], math.Float32bits(f))
		}
		return out, nil
	case "F16", "BF16":
		enc := f32ToBF16
		if dtype == "F16" {
			enc = f32ToF16
		}
		out := make([]byte, 2*len(v))
		for i, f := range v {
			binary.LittleEndian.PutUint16(out[2*i:], enc(f))
		}
		return out, nil
	}
	return nil, fmt.Errorf("safetensors: cannot encode floats as %q", dtype)
}

// DecodeFloats unpacks little-endian dtype ("F32", "F16" or "BF16") bytes.
func DecodeFloats(dtype string, raw []byte) ([]float32, error) {
	switch dtype {
	case "F32":
		out := make([]float32, len(raw)/4)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
		}
		return out, nil
	case "F16", "BF16":
		dec := bf16ToF32
		if dtype == "F16" {
			dec = f16ToF32
		}
		out := make([]float32, len(raw)/2)
		for i := range out {
			out[i] = dec(binary.LittleEndian.Uint16(raw[2*i:]))
		}
		return out, nil
	}
	return nil, fmt.Errorf("safetensors: cannot decode floats from %q", dtype)
}

// RoundFloat is f as dtype stores it ("F32", "F16" or "BF16").
func RoundFloat(dtype string, f float32) float32 {
	switch dtype {
	case "BF16":
		return bf16ToF32(f32ToBF16(f))
	case "F16":
		return f16ToF32(f32ToF16(f))
	}
	return f
}

func f32ToBF16(f float32) uint16 {
	b := math.Float32bits(f)
	if b&0x7fffffff > 0x7f800000 {
		return uint16(b>>16) | 0x40
	}
	b += 0x7fff + (b>>16)&1
	return uint16(b >> 16)
}

func bf16ToF32(h uint16) float32 { return math.Float32frombits(uint32(h) << 16) }

func f32ToF16(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16(b>>16) & 0x8000
	man := b & 0x7fffff
	if (b>>23)&0xff == 0xff {
		if man != 0 {
			return sign | 0x7e00
		}
		return sign | 0x7c00
	}
	exp := int((b>>23)&0xff) - 127 + 15
	if exp >= 0x1f {
		return sign | 0x7c00
	}
	if exp <= 0 {
		if exp < -10 {
			return sign
		}
		man |= 0x800000
		shift := uint(14 - exp)
		r := man >> shift
		rem, half := man&(1<<shift-1), uint32(1)<<(shift-1)
		if rem > half || (rem == half && r&1 == 1) {
			r++
		}
		return sign | uint16(r)
	}
	r := uint32(exp)<<10 | man>>13
	if rem := man & 0x1fff; rem > 0x1000 || (rem == 0x1000 && r&1 == 1) {
		r++
	}
	return sign | uint16(r)
}

func f16ToF32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1f
	man := uint32(h & 0x3ff)
	switch {
	case exp == 0x1f:
		return math.Float32frombits(sign | 0x7f800000 | man<<13)
	case exp == 0 && man == 0:
		return math.Float32frombits(sign)
	case exp == 0:
		f := float32(man) / (1 << 24)
		if sign != 0 {
			return -f
		}
		return f
	}
	return math.Float32frombits(sign | (exp+127-15)<<23 | man<<13)
}
