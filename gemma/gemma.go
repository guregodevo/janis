// Package gemma implements the Gemma2 architecture over engine.Backend. It
// reuses qwen's quantized layers (QuantLinear/QuantEmbedding/RMSNorm/KVCache)
// but adds Gemma's specifics: (1+weight) RMSNorm, embedding scaling by
// sqrt(hidden), softcapped manual attention (no fused SDPA), GeGLU MLP, four
// norms per block, a tied head, and final-logit softcapping.
//
// Sliding-window attention is intentionally omitted: for sequences within the
// window it is identical to global attention, which is what the mlx-lm
// reference uses for all layers — so this matches the oracle on normal prompts.
package gemma

import (
	"math"

	"memdoor/llm/engine"
	"memdoor/llm/qwen"
	"memdoor/llm/safetensors"
)

// loadNorm loads a Gemma RMSNorm: its stored weight is offset by +1 so the
// standard rms_norm(x, weight) reproduces rms_norm(x, 1+weight).
func loadNorm(b engine.Backend, st *safetensors.File, name string, eps float32) (*qwen.RMSNorm, error) {
	w, err := qwen.LoadTensor(b, st, name+".weight")
	if err != nil {
		return nil, err
	}
	return &qwen.RMSNorm{Weight: b.ScalarAdd(w, 1.0), Eps: eps}, nil
}

// Attention is Gemma2 grouped-query attention with logit softcapping.
type Attention struct {
	QProj, KProj, VProj, OProj         *qwen.QuantLinear
	NHeads, NKVHeads, HeadDim, Repeats int
	RopeBase, Scale, Softcap           float32
}

func loadAttention(b engine.Backend, st *safetensors.File, prefix string, cfg qwen.Config) (*Attention, error) {
	mk := func(n string) (*qwen.QuantLinear, error) {
		return qwen.LoadQuantLinear(b, st, prefix+"."+n, cfg.GroupSize, cfg.Bits, false)
	}
	q, err := mk("q_proj")
	if err != nil {
		return nil, err
	}
	k, err := mk("k_proj")
	if err != nil {
		return nil, err
	}
	v, err := mk("v_proj")
	if err != nil {
		return nil, err
	}
	o, err := mk("o_proj")
	if err != nil {
		return nil, err
	}
	return &Attention{
		QProj: q, KProj: k, VProj: v, OProj: o,
		NHeads: cfg.NHeads, NKVHeads: cfg.NKVHeads, HeadDim: cfg.HeadDim,
		Repeats:  cfg.NHeads / cfg.NKVHeads,
		RopeBase: cfg.RopeBase, Softcap: cfg.AttnSoftcap,
		Scale: float32(1.0 / math.Sqrt(float64(cfg.QueryPreAttnScalar))),
	}, nil
}

func (a *Attention) forward(b engine.Backend, x engine.Tensor, cache *qwen.KVCache, offset int, mask engine.Tensor) engine.Tensor {
	B, L := x.Shape()[0], x.Shape()[1]

	q := b.Transpose(b.Reshape(a.QProj.Forward(b, x), B, L, a.NHeads, a.HeadDim), 0, 2, 1, 3)
	k := b.Transpose(b.Reshape(a.KProj.Forward(b, x), B, L, a.NKVHeads, a.HeadDim), 0, 2, 1, 3)
	v := b.Transpose(b.Reshape(a.VProj.Forward(b, x), B, L, a.NKVHeads, a.HeadDim), 0, 2, 1, 3)

	q = b.RoPE(q, a.HeadDim, false, a.RopeBase, 1.0, offset)
	k = b.RoPE(k, a.HeadDim, false, a.RopeBase, 1.0, offset)
	k, v = cache.Update(b, k, v)
	q = b.ScalarMul(q, a.Scale)

	// GQA via broadcasting: queries -> [B, n_kv, repeats, L, hd]; k/v expand.
	q = b.Reshape(q, B, a.NKVHeads, a.Repeats, L, a.HeadDim)
	k = b.ExpandDims(k, 2) // [B, n_kv, 1, T, hd]
	v = b.ExpandDims(v, 2)

	kT := b.Transpose(k, 0, 1, 2, 4, 3)            // [B, n_kv, 1, hd, T]
	scores := b.MatMul(q, kT)                      // [B, n_kv, repeats, L, T]
	scores = b.ScalarMul(b.Tanh(b.ScalarMul(scores, 1.0/a.Softcap)), a.Softcap)
	if mask != nil {
		scores = b.Add(scores, mask)
	}
	scores = b.Softmax(scores, len(scores.Shape())-1)

	out := b.MatMul(scores, v) // [B, n_kv, repeats, L, hd]
	out = b.Reshape(out, B, a.NHeads, L, a.HeadDim)
	out = b.Reshape(b.Transpose(out, 0, 2, 1, 3), B, L, a.NHeads*a.HeadDim)
	return a.OProj.Forward(b, out)
}

// MLP is Gemma's GeGLU feed-forward: down(gelu(gate(x)) * up(x)).
type MLP struct {
	Gate, Up, Down *qwen.QuantLinear
}

func loadMLP(b engine.Backend, st *safetensors.File, prefix string, cfg qwen.Config) (*MLP, error) {
	mk := func(n string) (*qwen.QuantLinear, error) {
		return qwen.LoadQuantLinear(b, st, prefix+"."+n, cfg.GroupSize, cfg.Bits, false)
	}
	g, err := mk("gate_proj")
	if err != nil {
		return nil, err
	}
	u, err := mk("up_proj")
	if err != nil {
		return nil, err
	}
	d, err := mk("down_proj")
	if err != nil {
		return nil, err
	}
	return &MLP{Gate: g, Up: u, Down: d}, nil
}

func (m *MLP) forward(b engine.Backend, x engine.Tensor) engine.Tensor {
	return m.Down.Forward(b, b.Mul(b.Gelu(m.Gate.Forward(b, x)), m.Up.Forward(b, x)))
}

// Block is a Gemma2 decoder block with four norms (pre/post around each
// sublayer) and post-sublayer normalization before the residual add.
type Block struct {
	InputNorm, PostAttnNorm, PreFFNorm, PostFFNorm *qwen.RMSNorm
	Attn                                           *Attention
	MLP                                            *MLP
}

func loadBlock(b engine.Backend, st *safetensors.File, i int, cfg qwen.Config) (*Block, error) {
	p := layerPrefix(i)
	in, err := loadNorm(b, st, p+".input_layernorm", cfg.RMSEps)
	if err != nil {
		return nil, err
	}
	pa, err := loadNorm(b, st, p+".post_attention_layernorm", cfg.RMSEps)
	if err != nil {
		return nil, err
	}
	pf, err := loadNorm(b, st, p+".pre_feedforward_layernorm", cfg.RMSEps)
	if err != nil {
		return nil, err
	}
	pof, err := loadNorm(b, st, p+".post_feedforward_layernorm", cfg.RMSEps)
	if err != nil {
		return nil, err
	}
	attn, err := loadAttention(b, st, p+".self_attn", cfg)
	if err != nil {
		return nil, err
	}
	mlp, err := loadMLP(b, st, p+".mlp", cfg)
	if err != nil {
		return nil, err
	}
	return &Block{InputNorm: in, PostAttnNorm: pa, PreFFNorm: pf, PostFFNorm: pof, Attn: attn, MLP: mlp}, nil
}

func (bl *Block) forward(b engine.Backend, x engine.Tensor, cache *qwen.KVCache, offset int, mask engine.Tensor) engine.Tensor {
	r := bl.Attn.forward(b, bl.InputNorm.Forward(b, x), cache, offset, mask)
	h := b.Add(x, bl.PostAttnNorm.Forward(b, r))
	r = bl.MLP.forward(b, bl.PreFFNorm.Forward(b, h))
	return b.Add(h, bl.PostFFNorm.Forward(b, r))
}

// Model is a full Gemma2 causal LM (tied embeddings).
type Model struct {
	Embed    *qwen.QuantEmbedding
	Blocks   []*Block
	Norm     *qwen.RMSNorm
	Cfg      qwen.Config
	embScale float32
}

func LoadModel(b engine.Backend, st *safetensors.File, cfg qwen.Config) (*Model, error) {
	emb, err := qwen.LoadQuantEmbedding(b, st, "model.embed_tokens", cfg.GroupSize, cfg.Bits)
	if err != nil {
		return nil, err
	}
	blocks := make([]*Block, cfg.Layers)
	for i := range blocks {
		if blocks[i], err = loadBlock(b, st, i, cfg); err != nil {
			return nil, err
		}
	}
	norm, err := loadNorm(b, st, "model.norm", cfg.RMSEps)
	if err != nil {
		return nil, err
	}
	return &Model{Embed: emb, Blocks: blocks, Norm: norm, Cfg: cfg, embScale: float32(math.Sqrt(float64(cfg.Hidden)))}, nil
}

func (m *Model) forwardCachedT(b engine.Backend, idsT engine.Tensor, seqLen, offset int, caches []*qwen.KVCache) engine.Tensor {
	h := b.ScalarMul(m.Embed.Forward(b, idsT), m.embScale)
	h = b.Reshape(h, 1, seqLen, m.Cfg.Hidden)

	var mask engine.Tensor
	if seqLen > 1 {
		mask = causalMask(b, seqLen)
	}
	for i, blk := range m.Blocks {
		h = blk.forward(b, h, caches[i], offset, mask)
	}
	h = m.Norm.Forward(b, h)
	h = b.Slice(h, 1, seqLen-1, seqLen) // last position only

	logits := m.Embed.AsLinear(b, h)
	fc := m.Cfg.FinalSoftcap
	return b.ScalarMul(b.Tanh(b.ScalarMul(logits, 1.0/fc)), fc)
}

// Generate greedily decodes nGen tokens using the shared decode loop.
func (m *Model) Generate(b engine.Backend, prompt []int32, nGen int) []int32 {
	return qwen.GreedyGenerate(b, prompt, nGen, m.Cfg.Layers, m.forwardCachedT)
}

// GenerateSampled decodes with temperature/top-p sampling.
func (m *Model) GenerateSampled(b engine.Backend, prompt []int32, nGen int, p qwen.SampleParams) []int32 {
	return qwen.SampledGenerate(b, prompt, nGen, m.Cfg.Layers, m.Cfg.Vocab, m.forwardCachedT, p)
}

// NewSession returns a prefix-reusing session for multi-turn serving.
func (m *Model) NewSession() *qwen.Session {
	return &qwen.Session{NLayers: m.Cfg.Layers, Vocab: m.Cfg.Vocab, Forward: m.forwardCachedT}
}

// causalMask builds an additive f16 causal mask [L, L] (0 on/below diagonal,
// large-negative above) to match the reference's prefill mask.
func causalMask(b engine.Backend, L int) engine.Tensor {
	data := make([]float32, L*L)
	for i := 0; i < L; i++ {
		for j := i + 1; j < L; j++ {
			data[i*L+j] = -65504.0 // ~ -inf in f16
		}
	}
	return b.Cast(b.FromFloats(data, L, L), engine.F16)
}

func layerPrefix(i int) string { return "model.layers." + itoa(i) }

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var buf [12]byte
	n := len(buf)
	for i > 0 {
		n--
		buf[n] = byte('0' + i%10)
		i /= 10
	}
	return string(buf[n:])
}
