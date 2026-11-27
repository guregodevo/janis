package inkling2

import (
	"fmt"
	"strings"

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

// altNames are the tensor names of the MLX-VLM layout (the netprojects
// conversions), by the name of this package's own layout, the mlx-lm
// one: the same tensors, renamed. Applied in order to a name's parts.
var altNames = []struct{ from, to string }{
	{"model.language_model.", "language_model.model."},
	{".q_l2_clip.scale", ".q_scale"},
	{".rel_l2_clip.scale", ".rel_scale"},
	{".rel_logits_proj.proj", ".rel_proj"},
	{".conv1d.weight", ".conv.weight"},
	{".post_attention_convolution.", ".attn_sconv."},
	{".post_mlp_convolution.", ".mlp_sconv."},
	{".mlp.gate.weight", ".mlp.gate_weight"},
	{".mlp.gate.e_score_correction_bias", ".mlp.e_score_correction_bias"},
	{".q_message_gate.", ".controllers.attn."},
	{".k_message_gate.", ".controllers.k."},
	{".v_message_gate.", ".controllers.v."},
	{".mlp_message_gate.", ".controllers.mlp."},
	{".down_proj.weight", ".down_weight"},
	{".forget_gate_proj.weight", ".forget_weight"},
	{".message_gate_proj.weight", ".gain_weight"},
	{".message_input.query", ".message_queries"},
	{".mlp.experts.", ".mlp.switch_mlp."},
	{"model.vision_tower.", "vision_tower."},
	{".embeddings.position_embedding.weight", ".embeddings.position_embedding"},
	{".merger.linear_1", ".proj_in"},
	{".merger.linear_2", ".proj_out"},
}

// name is the checkpoint's name for a tensor of this package's layout.
func (ld *loader) name(name string) string {
	if !ld.cfg.AltLayout {
		return name
	}
	if strings.HasPrefix(name, "lm_head") {
		return "language_model." + name
	}
	gate := strings.Contains(name, "_message_gate.")
	for _, r := range altNames {
		if r.from == ".down_proj.weight" && !gate {
			continue // an expert's down projection keeps its name
		}
		name = strings.Replace(name, r.from, r.to, 1)
	}
	return name
}

func (ld *loader) has(name string) bool {
	_, _, err := ld.st.Info(ld.name(name))
	return err == nil
}

func (ld *loader) tensor(name string) engine.Tensor {
	if ld.err != nil {
		return nil
	}
	t, err := qwen.LoadTensor(ld.c, ld.st, ld.name(name))
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

// projection is a linear layer with no bias, quantized or kept as floats:
// the conversions differ in which they quantize.
type projection interface {
	Forward(b engine.Backend, x engine.Tensor) engine.Tensor
}

type denseProjection struct{ wT engine.Tensor }

func (d denseProjection) Forward(b engine.Backend, x engine.Tensor) engine.Tensor {
	return b.MatMul(x, d.wT)
}

// bits reads a quantized tensor's bit width off its shapes: the packed
// weight holds bits/32 of a word per input, the scales one per group.
func (ld *loader) bits(prefix string) int {
	_, w, err := ld.st.Info(ld.name(prefix + ".weight"))
	_, sc, err2 := ld.st.Info(ld.name(prefix + ".scales"))
	if err != nil || err2 != nil || len(w) == 0 || len(sc) == 0 {
		return ld.cfg.Bits
	}
	in := sc[len(sc)-1] * ld.cfg.GroupSize
	return w[len(w)-1] * 32 / in
}

func (ld *loader) linear(prefix string) projection {
	if ld.err != nil {
		return nil
	}
	if !ld.has(prefix + ".scales") {
		return denseProjection{wT: ld.dense(prefix + ".weight")}
	}
	l, err := qwen.LoadQuantLinear(ld.c, ld.st, ld.name(prefix), ld.cfg.GroupSize, ld.bits(prefix), false)
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

// dense loads an unquantized linear layer's weight [out, in] as [in, out],
// in the dtype the stream is computed in.
func (ld *loader) dense(name string) engine.Tensor {
	w := ld.tensor(name)
	if w == nil {
		return nil
	}
	return ld.c.model(ld.c.Transpose(w, 1, 0))
}

// f32T is dense in float32.
func (ld *loader) f32T(name string) engine.Tensor {
	w := ld.f32(name)
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
		ForgetT: ld.f32T(prefix + ".forget_gate_proj.weight"),
		AdmitT:  ld.f32T(prefix + ".message_gate_proj.weight"),
	}
}

// Forward mixes h and message, both [1, L, hidden], in float32.
func (g *MessageGate) Forward(c *ctx, h, message engine.Tensor) engine.Tensor {
	h = c.f32(h)
	if c.rel != nil && h.Shape()[1] == 1 {
		if out := c.rel.GateMix(h, message, g.DownT, g.ForgetT, g.AdmitT, defaultEps); out != nil {
			return out
		}
	}
	s := c.Concat(c.rms(h, defaultEps), c.rms(message, defaultEps), 2)
	s = c.f32(c.MatMul(c.model(s), g.DownT))
	forget := c.Tanh(c.MatMul(s, g.ForgetT))
	admit := c.Tanh(c.MatMul(s, g.AdmitT))
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
	// QKVNorm is the three input norms' weights of a full layer side by
	// side, [3, hidden], for a backend that mixes a token's query, key
	// and value inputs as one batch (engine.RelAttention).
	QKVNorm engine.Tensor
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
	if c.rel != nil && !l.Sliding && l.QGate != nil {
		rows := func(n *qwen.RMSNorm) engine.Tensor { return c.Reshape(c.f32(n.Weight), 1, cfg.Hidden) }
		l.QKVNorm = c.Concat(c.Concat(rows(l.QNorm), rows(l.KNorm), 0), rows(l.VNorm), 0)
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
	cache.begin(h.Shape()[1])
	hf := c.f32(h)
	normed := c.rms(hf, defaultEps)

	q, k, v := h, h, h
	var in engine.Tensor // the mix for the attention: one row, or three on a full layer
	if l.AttnQueryT != nil {
		in = c.depthAttention(blocks, normed, l.AttnQueryT)
		q = c.pick(in, 0)
		k, v = q, q
		if !l.Sliding {
			k, v = c.pick(in, 1), c.pick(in, 2)
		}
	}
	mixed := false
	if messages != nil && c.rel != nil && h.Shape()[1] == 1 && in != nil {
		// One token: the three gates as one batch, then the three norms.
		if l.Sliding {
			in = c.f32(q) // one input for the three
		}
		three := c.Slice(messages, 2, 0, 3)
		g := []*MessageGate{l.QGate, l.KGate, l.VGate}
		downs, forgets, admits := make([]engine.Tensor, 3), make([]engine.Tensor, 3), make([]engine.Tensor, 3)
		for i := range g {
			downs[i], forgets[i], admits[i] = g[i].DownT, g[i].ForgetT, g[i].AdmitT
		}
		if qkv := c.rel.GateMixRows(in, three, downs, forgets, admits, defaultEps); qkv != nil {
			if l.Sliding {
				qkv = l.InputNorm.Forward(c, qkv)
			} else {
				qkv = c.RMSNorm(qkv, l.QKVNorm, l.QNorm.Eps)
			}
			q, k, v, mixed = c.pick(qkv, 0), c.pick(qkv, 1), c.pick(qkv, 2), true
		}
	}
	if messages != nil && !mixed {
		q = l.QGate.Forward(c, q, c.pick(messages, 0))
		k = l.KGate.Forward(c, k, c.pick(messages, 1))
		v = l.VGate.Forward(c, v, c.pick(messages, 2))
	}
	switch {
	case mixed:
	case !l.Sliding:
		q, k, v = c.model(l.QNorm.Forward(c, q)), c.model(l.KNorm.Forward(c, k)), c.model(l.VNorm.Forward(c, v))
	case messages != nil:
		q, k, v = c.model(l.InputNorm.Forward(c, q)), c.model(l.InputNorm.Forward(c, k)), c.model(l.InputNorm.Forward(c, v))
	default:
		q = c.model(l.InputNorm.Forward(c, q))
		k, v = q, q
	}
	a := l.Attn.Forward(c, q, k, v, cache, pos, masks, eps)
	a = c.shortConv(a, l.AttnConv, cache, 2)

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
	x = c.shortConv(x, l.MLPConv, cache, 3)
	cache.steps += h.Shape()[1]
	return c.model(c.Add(residual, c.f32(x))), blocks
}
