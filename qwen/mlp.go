package qwen

import (
	"github.com/guregodevo/janis/engine"
	"github.com/guregodevo/janis/safetensors"
)

// MLP is the Qwen2 SwiGLU feed-forward: down(silu(gate(x)) * up(x)). All three
// projections are quantized and biasless.
type MLP struct {
	Gate, Up, Down *QuantLinear
}

// LoadMLP loads the gate/up/down projections under prefix (e.g. "model.layers.0.mlp").
func LoadMLP(b engine.Backend, st *safetensors.File, prefix string, groupSize, bits int) (*MLP, error) {
	g, err := LoadQuantLinear(b, st, prefix+".gate_proj", groupSize, bits, false)
	if err != nil {
		return nil, err
	}
	u, err := LoadQuantLinear(b, st, prefix+".up_proj", groupSize, bits, false)
	if err != nil {
		return nil, err
	}
	d, err := LoadQuantLinear(b, st, prefix+".down_proj", groupSize, bits, false)
	if err != nil {
		return nil, err
	}
	return &MLP{Gate: g, Up: u, Down: d}, nil
}

// Forward computes the SwiGLU feed-forward.
func (m *MLP) Forward(b engine.Backend, x engine.Tensor) engine.Tensor {
	gate := b.SiLU(m.Gate.Forward(b, x))
	up := m.Up.Forward(b, x)
	return m.Down.Forward(b, b.Mul(gate, up))
}
