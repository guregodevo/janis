package inkling2

import (
	"fmt"

	"github.com/guregodevo/janis/engine"
	"github.com/guregodevo/janis/safetensors"
)

// nvfp4Unit is what MLX divides an nvfp4 global scale by: the largest
// 4-bit value (6) times the largest 8-bit scale (448).
const nvfp4Unit = 2688

// expertStack is one projection of every expert of a layer, stacked
// [experts, out, in] and quantized in one of MLX's floating-point modes.
type expertStack struct {
	W, Scales engine.Tensor
	// Global is each expert's own scale over its nvfp4 block scales, already
	// divided by nvfp4Unit: float32 [experts]. Nil when the stack has none.
	Global          engine.Tensor
	GroupSize, Bits int
	Mode            string
}

// expertModes are the formats by (bits, group size), as the reference tells
// them apart: the checkpoint's quantization config does not name them.
var expertModes = map[[2]int]string{{4, 16}: "nvfp4", {4, 32}: "mxfp4", {8, 32}: "mxfp8"}

func (ld *loader) experts(prefix string, in int) *expertStack {
	e := &expertStack{W: ld.tensor(prefix + ".weight"), Scales: ld.tensor(prefix + ".scales")}
	if ld.err != nil {
		return nil
	}
	if _, _, _, err := ld.st.Get(prefix + ".biases"); err == nil {
		ld.err = fmt.Errorf("%s: experts quantized with zero points (affine) are not supported", prefix)
		return nil
	}
	ws, ss := e.W.Shape(), e.Scales.Shape()
	e.GroupSize = in / ss[len(ss)-1]
	e.Bits = ws[len(ws)-1] * 32 / in
	mode, ok := expertModes[[2]int{e.Bits, e.GroupSize}]
	if !ok {
		ld.err = fmt.Errorf("%s: %d-bit experts in groups of %d are in no format janis reads", prefix, e.Bits, e.GroupSize)
		return nil
	}
	e.Mode = mode
	if _, _, _, err := ld.st.Get(prefix + ".global_scale"); err == nil {
		e.Global = ld.c.ScalarMul(ld.f32(prefix+".global_scale"), 1.0/nvfp4Unit)
	}
	return e
}

// forward applies to each row of x [1, L, k, 1, in] the expert idx
// [1, L, k] names for it -> [1, L, k, 1, out], in the model dtype.
func (e *expertStack) forward(c *ctx, x, idx engine.Tensor) engine.Tensor {
	y := c.ops.GatherModeMatmul(x, e.W, e.Scales, idx, e.GroupSize, e.Bits, e.Mode)
	if e.Global == nil {
		return y
	}
	s := idx.Shape()
	g := c.Reshape(c.TakeAxis(e.Global, idx, 0), s[0], s[1], s[2], 1, 1)
	return c.model(c.Mul(c.f32(y), g))
}

// MoE is a layer's MLP: a router scores every expert for each token, the
// TopK best run, and their outputs are summed by score.
type MoE struct {
	RouterT, RouterBias engine.Tensor // float32 [hidden, experts], [experts]
	GateUp, Down        *expertStack
	TopK, Intermediate  int
	GateCap, UpCap      float32
	RouteScale          float32
}

func loadMoE(c *ctx, st *safetensors.File, prefix string, cfg Config) (*MoE, error) {
	ld := loader{c: c, st: st, cfg: cfg}
	m := &MoE{
		TopK: cfg.TopK, Intermediate: cfg.Intermediate,
		GateCap: cfg.GateCap, UpCap: cfg.UpCap, RouteScale: cfg.RouteScale,
	}
	if w := ld.f32(prefix + ".gate.weight"); w != nil {
		m.RouterT = c.Transpose(w, 1, 0)
	}
	m.RouterBias = ld.f32(prefix + ".gate.e_score_correction_bias")
	m.GateUp = ld.experts(prefix+".experts.gate_up_proj", cfg.Hidden)
	m.Down = ld.experts(prefix+".experts.down_proj", cfg.Intermediate)
	return m, ld.err
}

// Forward runs the block on x [1, L, hidden] in the model dtype.
func (m *MoE) Forward(c *ctx, x engine.Tensor) engine.Tensor {
	L, hidden, k, I := x.Shape()[1], x.Shape()[2], m.TopK, m.Intermediate

	// The bias only chooses the experts; the weights are the plain scores
	// of the chosen, normalized to sum to RouteScale.
	scores := c.Sigmoid(c.MatMul(c.f32(x), m.RouterT)) // [1, L, experts]
	idx := c.ops.TopK(c.Add(scores, m.RouterBias), k)  // [1, L, k]
	weights := c.ops.TakeAlong(scores, idx, 2)
	weights = c.ops.Div(weights, c.ScalarAdd(c.sumLast(weights), 1e-20))
	weights = c.ScalarMul(weights, m.RouteScale)

	// gate_up's output rows alternate gate, up.
	gu := c.Reshape(m.GateUp.forward(c, c.Reshape(x, 1, L, 1, 1, hidden), idx), 1, L, k, I, 2)
	gate := c.f32(c.Reshape(c.Slice(gu, 4, 0, 1), 1, L, k, 1, I))
	up := c.f32(c.Reshape(c.Slice(gu, 4, 1, 2), 1, L, k, 1, I))
	// SwiGLU with both branches softly capped: tanh(x/cap)*cap.
	act := c.Mul(c.ScalarMul(c.Tanh(c.ScalarMul(gate, 1/m.GateCap)), m.GateCap), c.Sigmoid(gate))
	act = c.Mul(act, c.ScalarMul(c.Tanh(c.ScalarMul(up, 1/m.UpCap)), m.UpCap))

	out := c.f32(c.Reshape(m.Down.forward(c, c.model(act), idx), 1, L, k, hidden))
	out = c.Mul(out, c.Reshape(weights, 1, L, k, 1))
	return c.model(c.ScalarMul(c.Mean(out, 2), float32(k)))
}
