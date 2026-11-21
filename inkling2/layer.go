package inkling2

import (
	"fmt"

	"github.com/guregodevo/janis/engine"
	"github.com/guregodevo/janis/qwen"
	"github.com/guregodevo/janis/safetensors"
)

// loader reads a model's tensors and keeps the first error, so a layer's
// load reads as a list.
type loader struct {
	c   *ctx
	st  *safetensors.File
	cfg Config
	err error
}

func (ld *loader) tensor(name string) engine.Tensor {
	if ld.err != nil {
		return nil
	}
	t, err := qwen.LoadTensor(ld.c, ld.st, name)
	if err != nil {
		ld.err = err
		return nil
	}
	return t
}

func (ld *loader) f32(name string) engine.Tensor {
	t := ld.tensor(name)
	if t == nil {
		return nil
	}
	return ld.c.f32(t)
}

func (ld *loader) linear(prefix string) *qwen.QuantLinear {
	if ld.err != nil {
		return nil
	}
	l, err := qwen.LoadQuantLinear(ld.c, ld.st, prefix, ld.cfg.GroupSize, ld.cfg.Bits, false)
	if err != nil {
		ld.err = err
		return nil
	}
	return l
}

func (ld *loader) norm(prefix string) *qwen.RMSNorm {
	w := ld.tensor(prefix + ".weight")
	if w == nil {
		return nil
	}
	return &qwen.RMSNorm{Weight: w, Eps: ld.cfg.Eps}
}

// queries loads a depth attention's queries, [dims] or [n, dims], as the
// [dims, n] float32 the scores are a product with.
func (ld *loader) queries(name string) engine.Tensor {
	q := ld.f32(name)
	if q == nil {
		return nil
	}
	dims := ld.cfg.Hidden
	n := 1
	if s := q.Shape(); len(s) == 2 {
		n = s[0]
	}
	return ld.c.Transpose(ld.c.Reshape(q, n, dims), 1, 0)
}

// dense loads an unquantized linear layer's weight [out, in] as [in, out].
func (ld *loader) dense(name string) engine.Tensor {
	w := ld.tensor(name)
	if w == nil {
		return nil
	}
	return ld.c.Transpose(w, 1, 0)
}

// MessageGate mixes a looped layer's input with the message from the pass
// before: a low-rank projection of both decides, per channel, how much of
// the input to forget and how much of the message to let in.
type MessageGate struct {
	DownT           engine.Tensor // [2*hidden, rank], model dtype
	ForgetT, AdmitT engine.Tensor // [rank, hidden], float32
}

func (ld *loader) gate(prefix string) *MessageGate {
	return &MessageGate{
		DownT:   ld.dense(prefix + ".down_proj.weight"),
		ForgetT: ld.dense(prefix + ".forget_gate_proj.weight"),
		AdmitT:  ld.dense(prefix + ".message_gate_proj.weight"),
	}
}

// Forward mixes h and message, both [1, L, hidden], in float32.
func (g *MessageGate) Forward(c *ctx, h, message engine.Tensor) engine.Tensor {
	h = c.f32(h)
	s := c.Concat(c.rms(h, defaultEps), c.rms(message, defaultEps), 2)
	s = c.f32(c.MatMul(c.model(s), g.DownT))
	forget := c.Tanh(c.MatMul(s, c.f32(g.ForgetT)))
	admit := c.Tanh(c.MatMul(s, c.f32(g.AdmitT)))
	keep := c.ScalarAdd(c.ScalarMul(forget, -1), 1)
	return c.Add(c.Mul(h, keep), c.Mul(message, admit))
}

// Layer is one decoder layer. Its attention and its MLP each read a mix of
// the block outputs so far and the current state (the depth attention), not
// the residual stream itself; a layer that opens a block adds its input to
// those outputs and starts the stream again from its attention's output.
type Layer struct {
	Sliding  bool
	Boundary bool // opens a block

	// AttnQueryT is nil on layer 0, which reads the embedding as it is. A
	// sliding layer has one query; a full one has three, for its query, key
	// and value inputs.
	AttnQueryT, MLPQueryT engine.Tensor

	InputNorm           *qwen.RMSNorm // sliding
	QNorm, KNorm, VNorm *qwen.RMSNorm // full
	Attn                *Attention
	AttnConv            engine.Tensor
	PostNorm            *qwen.RMSNorm
	MLP                 *MoE
	MLPConv             engine.Tensor

	// The gates of a looped layer, nil outside the core.
	QGate, KGate, VGate, MLPGate *MessageGate
}

func loadLayer(c *ctx, st *safetensors.File, i int, cfg Config) (*Layer, error) {
	p := fmt.Sprintf("model.language_model.layers.%d", i)
	ld := loader{c: c, st: st, cfg: cfg}
	l := &Layer{Sliding: cfg.Local[i], Boundary: i%cfg.BlockSize == 0}
	if i > 0 {
		l.AttnQueryT = ld.queries(p + ".attention_input.query")
	}
	l.MLPQueryT = ld.queries(p + ".mlp_input.query")
	if l.Sliding {
		l.InputNorm = ld.norm(p + ".input_layernorm")
	} else {
		l.QNorm = ld.norm(p + ".q_input_layernorm")
		l.KNorm = ld.norm(p + ".k_input_layernorm")
		l.VNorm = ld.norm(p + ".v_input_layernorm")
	}
	l.AttnConv = ld.tensor(p + ".post_attention_convolution.conv1d.weight")
	l.PostNorm = ld.norm(p + ".post_attention_layernorm")
	l.MLPConv = ld.tensor(p + ".post_mlp_convolution.conv1d.weight")
	if i >= cfg.CoreStart && i < cfg.CoreEnd {
		l.QGate = ld.gate(p + ".q_message_gate")
		l.KGate = ld.gate(p + ".k_message_gate")
		l.VGate = ld.gate(p + ".v_message_gate")
		l.MLPGate = ld.gate(p + ".mlp_message_gate")
	}
	if ld.err != nil {
		return nil, ld.err
	}
	var err error
	if l.Attn, err = loadAttention(c, st, p+".self_attn", cfg, l.Sliding); err != nil {
		return nil, err
	}
	if l.MLP, err = loadMoE(c, st, p+".mlp", cfg); err != nil {
		return nil, err
	}
	return l, nil
}

// Forward runs the layer on h [1, L, hidden] at positions pos.., with the
// block outputs so far and, in a loop pass after the first, the messages
// [1, L, 4, hidden] for its query, key, value and MLP inputs. It returns
// the new stream and block outputs.
func (l *Layer) Forward(c *ctx, h engine.Tensor, blocks []engine.Tensor, cache *VisitCache, pos int, messages engine.Tensor, masks *maskMaker) (engine.Tensor, []engine.Tensor) {
	eps := masks.cfg.Eps
	hf := c.f32(h)
	normed := c.rms(hf, defaultEps)

	q, k, v := h, h, h
	if l.AttnQueryT != nil {
		in := c.depthAttention(blocks, normed, l.AttnQueryT)
		q = c.pick(in, 0)
		k, v = q, q
		if !l.Sliding {
			k, v = c.pick(in, 1), c.pick(in, 2)
		}
	}
	if messages != nil {
		q = l.QGate.Forward(c, q, c.pick(messages, 0))
		k = l.KGate.Forward(c, k, c.pick(messages, 1))
		v = l.VGate.Forward(c, v, c.pick(messages, 2))
	}
	switch {
	case !l.Sliding:
		q, k, v = c.model(l.QNorm.Forward(c, q)), c.model(l.KNorm.Forward(c, k)), c.model(l.VNorm.Forward(c, v))
	case messages != nil:
		q, k, v = c.model(l.InputNorm.Forward(c, q)), c.model(l.InputNorm.Forward(c, k)), c.model(l.InputNorm.Forward(c, v))
	default:
		q = c.model(l.InputNorm.Forward(c, q))
		k, v = q, q
	}
	a := l.Attn.Forward(c, q, k, v, cache, pos, masks, eps)
	a, cache.Conv[2] = c.shortConv(a, l.AttnConv, cache.Conv[2])

	residual := c.f32(a)
	if l.Boundary {
		blocks = append(blocks[:len(blocks):len(blocks)], normed)
	} else {
		residual = c.Add(hf, residual)
	}

	x := c.pick(c.depthAttention(blocks, c.rms(residual, defaultEps), l.MLPQueryT), 0)
	if messages != nil {
		x = l.MLPGate.Forward(c, x, c.pick(messages, 3))
	}
	x = l.MLP.Forward(c, c.model(l.PostNorm.Forward(c, x)))
	x, cache.Conv[3] = c.shortConv(x, l.MLPConv, cache.Conv[3])
	cache.steps += h.Shape()[1]
	return c.model(c.Add(residual, c.f32(x))), blocks
}
