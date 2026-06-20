package qwen

import (
	"memdoor/llm/engine"
	"memdoor/llm/safetensors"
)

// QuantLinear is an affine-quantized linear layer (QuantizedLinear): a packed
// 4-bit weight with per-group scales/biases, plus an optional additive bias.
// Note the two distinct "bias" tensors: `Biases` are the quantization zero
// points; `Bias` is the layer's additive bias.
type QuantLinear struct {
	Weight, Scales, Biases engine.Tensor
	Bias                   engine.Tensor // nil when the layer has no bias
	GroupSize, Bits        int
}

// LoadQuantLinear loads the quantized weight triplet under prefix, and the
// additive bias when withBias is set.
func LoadQuantLinear(b engine.Backend, st *safetensors.File, prefix string, groupSize, bits int, withBias bool) (*QuantLinear, error) {
	w, err := loadTensor(b, st, prefix+".weight")
	if err != nil {
		return nil, err
	}
	s, err := loadTensor(b, st, prefix+".scales")
	if err != nil {
		return nil, err
	}
	qb, err := loadTensor(b, st, prefix+".biases")
	if err != nil {
		return nil, err
	}
	l := &QuantLinear{Weight: w, Scales: s, Biases: qb, GroupSize: groupSize, Bits: bits}
	if withBias {
		if l.Bias, err = loadTensor(b, st, prefix+".bias"); err != nil {
			return nil, err
		}
	}
	return l, nil
}

// Forward computes x @ Wᵀ (+ bias) via quantized matmul.
func (l *QuantLinear) Forward(b engine.Backend, x engine.Tensor) engine.Tensor {
	y := b.QuantMatmul(x, l.Weight, l.Scales, l.Biases, true, l.GroupSize, l.Bits)
	if l.Bias != nil {
		y = b.Add(y, l.Bias)
	}
	return y
}
