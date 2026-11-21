package qwen

import "github.com/guregodevo/janis/engine"

// Param is one weight an edit may replace, by its checkpoint name: Quant is
// set for an affine-quantized matrix (named by its ".weight"), Plain for a
// tensor the model uses as stored.
type Param struct {
	Name  string
	Quant *QuantLinear
	Plain *engine.Tensor
}

// QuantParam is the Param of the quantized matrix l stored under prefix.
func QuantParam(prefix string, l *QuantLinear) Param {
	return Param{Name: prefix + ".weight", Quant: l}
}

// NormParam is the Param of the norm weight stored under prefix.
func NormParam(prefix string, n *RMSNorm) Param {
	return Param{Name: prefix + ".weight", Plain: &n.Weight}
}

// Reset drops what the session holds, so the next prompt is read from its
// first token: after a weight edit the caches were computed by other weights.
func (s *Session) Reset(b engine.Backend) {
	if sw, ok := b.(engine.Sweeper); ok {
		sw.Unpin(cacheTensors(s.Caches)...)
	}
	s.IDs, s.Caches, s.marks, s.uses = nil, nil, nil, 0
}
