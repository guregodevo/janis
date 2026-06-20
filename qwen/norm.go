package qwen

import (
	"memdoor/llm/engine"
	"memdoor/llm/safetensors"
)

// RMSNorm is a Qwen2 RMS normalization layer (a learned per-channel weight
// applied after normalizing by RMS, with epsilon for stability).
type RMSNorm struct {
	Weight engine.Tensor
	Eps    float32
}

// LoadRMSNorm loads the norm weight at name (e.g. "model.layers.0.input_layernorm").
func LoadRMSNorm(b engine.Backend, st *safetensors.File, name string, eps float32) (*RMSNorm, error) {
	w, err := loadTensor(b, st, name+".weight")
	if err != nil {
		return nil, err
	}
	return &RMSNorm{Weight: w, Eps: eps}, nil
}

// Forward normalizes x and scales by the learned weight.
func (n *RMSNorm) Forward(b engine.Backend, x engine.Tensor) engine.Tensor {
	return b.RMSNorm(x, n.Weight, n.Eps)
}
