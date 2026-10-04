package gguf

import (
	"encoding/binary"
	"fmt"
	"math"
)

// Dequantize turns a tensor's stored blocks into float32 values in logical
// (row-major) order.
//
// Every GGML type is a block code: a block of `values` numbers is stored in
// `bytes` bytes, headed by an f16 scale — and for the Q4_1/Q5_1/Q8_1 families a
// second f16 minimum or sum. The quantized payload follows as bit-packed
// unsigned or two's-complement integers.
//
// The exact layouts, which this implements from the GGML block definitions:
//
//	Q4_0: d(f16), 16 bytes = 32 nibbles. value[i] = d * (nibble_i - 8)
//	Q4_1: d(f16), m(f16), 16 bytes.        value[i] = d * nibble_i + m
//	Q5_0: d(f16), qh(u32), 16 bytes.       value[i] = d * ((nibble_i | (qh bit i << 4)) - 16)
//	Q5_1: d(f16), m(f16), qh(u32), 16 bytes. value[i] = d * (nibble_i | (qh bit i << 4)) + m
//	Q8_0: d(f16), 32 int8.                 value[i] = d * int8_i
//	Q8_1: d(f16), s(f16), 32 int8.         value[i] = d * int8_i
//
// In the 4/5-bit types the first 16 payload bytes hold the LOW nibbles of
// values 0..15 and the HIGH nibbles of values 16..31 (in that byte order) —
// that is GGML's packing, and getting it backwards yields values that still
// look like plausible weights, which is why the tests compare against a
// second container rather than eyeballing statistics.
func Dequantize(t TensorType, raw []byte) ([]float32, error) {
	blk, ok := typeBlock(t)
	if !ok {
		return nil, fmt.Errorf("%s is not a type this reader dequantizes", TypeName(t))
	}
	if len(raw)%blk.bytes != 0 {
		return nil, fmt.Errorf("%s: %d bytes is not a multiple of the block size %d",
			TypeName(t), len(raw), blk.bytes)
	}
	nBlocks := len(raw) / blk.bytes
	out := make([]float32, 0, nBlocks*blk.values)
	for b := 0; b < nBlocks; b++ {
		p := raw[b*blk.bytes : (b+1)*blk.bytes]
		switch t {
		case TensorF32:
			out = append(out, math.Float32frombits(binary.LittleEndian.Uint32(p)))
		case TensorF16:
			out = append(out, f16(uint16(binary.LittleEndian.Uint16(p))))
		case TensorBF16:
			// bf16 is the top 16 bits of an f32: widen by shifting the mantissa
			// into place rather than converting through f16.
			out = append(out, math.Float32frombits(uint32(binary.LittleEndian.Uint16(p))<<16))
		case TensorQ8_0, TensorQ8_1:
			d := f16(binary.LittleEndian.Uint16(p[0:2]))
			// Q8_1 carries a second f16 (the block sum) we do not need; the
			// int8 payload starts after it.
			q := p[2:34]
			if t == TensorQ8_1 {
				q = p[4:36]
			}
			for i := 0; i < 32; i++ {
				out = append(out, d*float32(int8(q[i])))
			}
		case TensorQ4_0:
			d := f16(binary.LittleEndian.Uint16(p[0:2]))
			nib := p[2:18]
			for i := 0; i < 16; i++ {
				out = append(out, d*float32(int(nib[i]&0x0F)-8))
			}
			for i := 0; i < 16; i++ {
				out = append(out, d*float32(int(nib[i]>>4)-8))
			}
		case TensorQ4_1:
			d := f16(binary.LittleEndian.Uint16(p[0:2]))
			m := f16(binary.LittleEndian.Uint16(p[2:4]))
			nib := p[4:20]
			for i := 0; i < 16; i++ {
				out = append(out, d*float32(nib[i]&0x0F)+m)
			}
			for i := 0; i < 16; i++ {
				out = append(out, d*float32(nib[i]>>4)+m)
			}
		case TensorQ5_0, TensorQ5_1:
			// Layout: scale [+ min for Q5_1], then a u32 holding the 5th bit of
			// each of the 32 values, then the 16 nibble bytes.
			off := 2
			d := f16(binary.LittleEndian.Uint16(p[0:2]))
			var m float32
			if t == TensorQ5_1 {
				m = f16(binary.LittleEndian.Uint16(p[2:4]))
				off = 4
			}
			qh := binary.LittleEndian.Uint32(p[off : off+4])
			nib := p[off+4 : off+20]
			for i := 0; i < 32; i++ {
				v := uint32(nib[i%16])
				if i < 16 {
					v &= 0x0F
				} else {
					v >>= 4
				}
				if qh&(1<<uint(i)) != 0 {
					v |= 0x10
				}
				if t == TensorQ5_0 {
					out = append(out, d*float32(int(v)-16))
				} else {
					out = append(out, d*float32(v)+m)
				}
			}
		}
	}
	return out, nil
}

// f16 widens an IEEE 754 half to float32 (GGML stores its scales this way).
func f16(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1F
	mant := uint32(h) & 0x3FF
	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign << 31)
		}
		// subnormal: renormalize into an f32 exponent
		e := uint32(127 - 15 + 1)
		for mant&0x400 == 0 {
			mant <<= 1
			e--
		}
		mant &= 0x3FF
		return math.Float32frombits(sign<<31 | e<<23 | mant<<13)
	case 0x1F:
		if mant == 0 {
			return inf(sign)
		}
		return nan(sign)
	}
	return math.Float32frombits(sign<<31 | (exp+127-15)<<23 | mant<<13)
}

func inf(sign uint32) float32 { return math.Float32frombits(sign<<31 | 0xFF<<23) }
func nan(sign uint32) float32 { return math.Float32frombits(sign<<31 | 0xFF<<23 | 1) }
