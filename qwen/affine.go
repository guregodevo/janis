package qwen

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/guregodevo/janis/safetensors"
)

// QuantizeAffine quantizes a row-major [rows, cols] matrix in MLX's affine
// layout, the one every backend's Dequantize reads: 32/bits values per uint32,
// least significant first, and per group of groupSize values one scale and
// one bias, value = scale*q + bias. The scales and biases are stored as
// scaleDtype and rounded to it before q is chosen, so the stored triple
// dequantizes to within half a step of v. Bits is 2, 4 or 8.
func QuantizeAffine(v []float32, rows, cols, groupSize, bits int, scaleDtype string) (packed, scales, biases []byte, err error) {
	if bits != 2 && bits != 4 && bits != 8 {
		return nil, nil, nil, fmt.Errorf("affine quantization: %d bits (2, 4 or 8)", bits)
	}
	if groupSize <= 0 || cols%groupSize != 0 || (cols*bits)%32 != 0 {
		return nil, nil, nil, fmt.Errorf("affine quantization: %d columns in groups of %d at %d bits", cols, groupSize, bits)
	}
	if len(v) != rows*cols {
		return nil, nil, nil, fmt.Errorf("affine quantization: %d values for [%d, %d]", len(v), rows, cols)
	}
	levels := float32(int(1)<<bits - 1)
	perWord := 32 / bits
	groups := cols / groupSize
	words := make([]byte, 4*rows*cols/perWord)
	sc := make([]float32, rows*groups)
	bi := make([]float32, rows*groups)
	for r := 0; r < rows; r++ {
		for g := 0; g < groups; g++ {
			grp := v[r*cols+g*groupSize : r*cols+(g+1)*groupSize]
			lo, hi := grp[0], grp[0]
			for _, x := range grp {
				if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
					return nil, nil, nil, fmt.Errorf("affine quantization: row %d holds %v", r, x)
				}
				lo, hi = min(lo, x), max(hi, x)
			}
			bias := safetensors.RoundFloat(scaleDtype, lo)
			scale := safetensors.RoundFloat(scaleDtype, (hi-bias)/levels)
			sc[r*groups+g], bi[r*groups+g] = scale, bias
			for j, x := range grp {
				var q float32
				if scale != 0 {
					q = min(max(float32(math.Round(float64((x-bias)/scale))), 0), levels)
				}
				col := g*groupSize + j
				w := 4 * ((r*cols + col) / perWord)
				word := binary.LittleEndian.Uint32(words[w:])
				word |= uint32(q) << uint((col%perWord)*bits)
				binary.LittleEndian.PutUint32(words[w:], word)
			}
		}
	}
	if scales, err = safetensors.EncodeFloats(scaleDtype, sc); err != nil {
		return nil, nil, nil, err
	}
	if biases, err = safetensors.EncodeFloats(scaleDtype, bi); err != nil {
		return nil, nil, nil, err
	}
	return words, scales, biases, nil
}

// DequantizeAffine is QuantizeAffine's inverse, on the stored triple: w is
// U32 [rows, cols*bits/32], scales and biases [rows, cols/groupSize] in a
// float dtype. It returns the row-major [rows, cols] values.
func DequantizeAffine(w, scales, biases safetensors.Entry, groupSize, bits int) ([]float32, error) {
	if bits != 2 && bits != 4 && bits != 8 {
		return nil, fmt.Errorf("affine dequantization: %d bits (2, 4 or 8)", bits)
	}
	if w.Dtype != "U32" || len(w.Shape) != 2 || len(w.Raw) != 4*w.Shape[0]*w.Shape[1] {
		return nil, fmt.Errorf("affine dequantization: weight %s %v", w.Dtype, w.Shape)
	}
	rows, cols := w.Shape[0], w.Shape[1]*32/bits
	if groupSize <= 0 || cols%groupSize != 0 {
		return nil, fmt.Errorf("affine dequantization: %d columns in groups of %d", cols, groupSize)
	}
	sc, err := safetensors.DecodeFloats(scales.Dtype, scales.Raw)
	if err != nil {
		return nil, err
	}
	bi, err := safetensors.DecodeFloats(biases.Dtype, biases.Raw)
	if err != nil {
		return nil, err
	}
	groups := cols / groupSize
	if len(sc) != rows*groups || len(bi) != rows*groups {
		return nil, fmt.Errorf("affine dequantization: %d scales and %d biases for [%d, %d] in groups of %d", len(sc), len(bi), rows, cols, groupSize)
	}
	perWord, mask := 32/bits, uint32(1)<<bits-1
	out := make([]float32, rows*cols)
	for r := 0; r < rows; r++ {
		for j := 0; j < cols; j++ {
			i := r*cols + j
			word := binary.LittleEndian.Uint32(w.Raw[4*(i/perWord):])
			q := (word >> uint((j%perWord)*bits)) & mask
			g := r*groups + j/groupSize
			out[i] = float32(q)*sc[g] + bi[g]
		}
	}
	return out, nil
}
