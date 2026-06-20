// Package qwen implements the Qwen2 model forward pass over engine.Backend,
// one rung at a time. Everything here is backend-agnostic — it never imports
// mlxc or calls C.
package qwen

import (
	"fmt"

	"memdoor/llm/engine"
	"memdoor/llm/safetensors"
)

// stDType maps a safetensors dtype string to an engine.DType.
func stDType(s string) (engine.DType, error) {
	switch s {
	case "F32":
		return engine.F32, nil
	case "F16":
		return engine.F16, nil
	case "BF16":
		return engine.BF16, nil
	case "I32":
		return engine.I32, nil
	case "U32":
		return engine.U32, nil
	default:
		return 0, fmt.Errorf("unsupported safetensors dtype %q", s)
	}
}

// LoadTensor reads a named tensor from the safetensors file into the backend
// (exported for sibling architecture packages).
func LoadTensor(b engine.Backend, st *safetensors.File, name string) (engine.Tensor, error) {
	return loadTensor(b, st, name)
}

// loadTensor reads a named tensor from the safetensors file into the backend.
func loadTensor(b engine.Backend, st *safetensors.File, name string) (engine.Tensor, error) {
	dt, shape, raw, err := st.Get(name)
	if err != nil {
		return nil, err
	}
	edt, err := stDType(dt)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", name, err)
	}
	return b.FromRaw(edt, raw, shape...), nil
}

// QuantEmbedding is an affine-quantized embedding table (weight/scales/biases).
type QuantEmbedding struct {
	Weight, Scales, Biases engine.Tensor
	GroupSize, Bits        int
}

// LoadQuantEmbedding loads the three quantized tensors under prefix
// (e.g. "model.embed_tokens").
func LoadQuantEmbedding(b engine.Backend, st *safetensors.File, prefix string, groupSize, bits int) (*QuantEmbedding, error) {
	w, err := loadTensor(b, st, prefix+".weight")
	if err != nil {
		return nil, err
	}
	s, err := loadTensor(b, st, prefix+".scales")
	if err != nil {
		return nil, err
	}
	bi, err := loadTensor(b, st, prefix+".biases")
	if err != nil {
		return nil, err
	}
	return &QuantEmbedding{Weight: w, Scales: s, Biases: bi, GroupSize: groupSize, Bits: bits}, nil
}

// Forward gathers the quantized rows for ids and dequantizes them, yielding
// [len(ids), hidden] — matching mlx-lm's QuantizedEmbedding.
func (e *QuantEmbedding) Forward(b engine.Backend, ids engine.Tensor) engine.Tensor {
	w := b.TakeAxis(e.Weight, ids, 0)
	s := b.TakeAxis(e.Scales, ids, 0)
	bi := b.TakeAxis(e.Biases, ids, 0)
	return b.Dequantize(w, s, bi, e.GroupSize, e.Bits)
}

// AsLinear uses the embedding table as the (tied) output projection:
// x @ Wᵀ -> logits over the vocabulary.
func (e *QuantEmbedding) AsLinear(b engine.Backend, x engine.Tensor) engine.Tensor {
	return b.QuantMatmul(x, e.Weight, e.Scales, e.Biases, true, e.GroupSize, e.Bits)
}
