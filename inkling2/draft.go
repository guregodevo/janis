package inkling2

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/guregodevo/janis/engine"
	"github.com/guregodevo/janis/qwen"
	"github.com/guregodevo/janis/safetensors"
)

// DraftFile is the checkpoint of the draft head, beside the model's.
const DraftFile = "mtp.safetensors"

// DraftConfig is config.json's mtp_config.
type DraftConfig struct {
	Hidden       int     `json:"hidden_size"`
	Layers       int     `json:"num_hidden_layers"`
	Heads        int     `json:"num_attention_heads"`
	KVHeads      int     `json:"num_key_value_heads"`
	HeadDim      int     `json:"head_dim"`
	Intermediate int     `json:"intermediate_size"`
	Window       int     `json:"sliding_window_size"`
	Local        []int   `json:"local_layer_ids"`
	Block        int     `json:"block_size"`
	Rank         int     `json:"markov_rank"`
	HeadType     string  `json:"markov_head_type"`
	Targets      []int   `json:"target_layer_ids"`
	RopeTheta    float64 `json:"rope_theta"`
	Eps          float64 `json:"rms_norm_eps"`
}

// LoadDraftConfig reads mtp_config; ok is false when the model has none or
// no draft checkpoint beside it.
func LoadDraftConfig(modelDir string, cfg Config) (d DraftConfig, ok bool, err error) {
	if _, err := os.Stat(filepath.Join(modelDir, DraftFile)); err != nil {
		return d, false, nil
	}
	raw, err := os.ReadFile(filepath.Join(modelDir, "config.json"))
	if err != nil {
		return d, false, err
	}
	var r struct {
		MTP  *DraftConfig `json:"mtp_config"`
		Text struct {
			MTP *DraftConfig `json:"mtp_config"`
		} `json:"text_config"`
	}
	if err := json.Unmarshal(raw, &r); err != nil {
		return d, false, err
	}
	if r.MTP == nil {
		r.MTP = r.Text.MTP
	}
	if r.MTP == nil {
		return d, false, nil
	}
	d = *r.MTP
	visits := cfg.Visits()
	bad := d.HeadType != "rnn_hidden" || d.Hidden != cfg.Hidden || d.Layers < 1 || d.Block < 2 ||
		d.Rank < 1 || d.Heads < 1 || d.KVHeads < 1 || d.Heads%d.KVHeads != 0 || d.HeadDim%2 != 0 ||
		d.Window < 1 || len(d.Targets) < 2 || d.Targets[len(d.Targets)-1] != visits
	for i, t := range d.Targets {
		bad = bad || t < 0 || t > visits || (i > 0 && t <= d.Targets[i-1])
	}
	if bad {
		return d, false, fmt.Errorf("%s: mtp_config is not one janis can run", modelDir)
	}
	return d, true, nil
}

type draftLayer struct {
	InNorm, PostNorm, CtxNorm engine.Tensor // RMSNorm weights [hidden]
	QT, KT, VT, KCtxT, VCtxT  engine.Tensor // [hidden, heads*dim]
	OT                        engine.Tensor
	QNorm, KNorm, KCtxNorm    engine.Tensor // [head dim]
	GateT, UpT, DownT         engine.Tensor
	Window                    int // 0: the whole context
}

// Draft is the model's draft head (DSpark): three small decoder layers
// that, from one token and what the model itself computed for the tokens
// before it, propose the next Block-1 tokens in one pass. Each layer
// attends over a mix of the model's states at four depths (Targets), kept
// as keys and values of its own; a small recurrent head over the proposed
// tokens corrects each one's logits by the token before it.
//
// The session verifies the proposals with the model, so a wrong draft
// costs time and never changes the reply.
type Draft struct {
	Cfg    DraftConfig
	Layers []draftLayer
	Norm   engine.Tensor
	Mask   engine.Tensor // [1, hidden]: the embedding of a place to fill
	// Mix[i] are the weights of the four depths for layer i, the last row
	// for the recurrent head: [1, targets] float32.
	Mix []engine.Tensor

	// The recurrent head.
	W1               engine.Tensor // [vocab, rank]: the token before
	StateT, ContextT engine.Tensor // rnn_proj, split: [2*rank, 3*rank], [2*hidden, 3*rank]
	RNNBias          engine.Tensor // [3*rank]
	RNNNorm          engine.Tensor
	// Count is how many tokens a proposal holds, Block-1 at most. The head
	// always runs its whole block; the first Count places are read.
	Count int

	// FoldT and FoldBias are the output head applied to the recurrent
	// head's projection back to the hidden size, computed once:
	// [rank, vocab] and [vocab].
	FoldT, FoldBias engine.Tensor

	m *Model
}

// LoadDraft loads the draft head from the model directory and turns the
// model's sessions speculative.
func (m *Model) LoadDraft(modelDir string, cfg DraftConfig) error {
	st, err := safetensors.Open(filepath.Join(modelDir, DraftFile))
	if err != nil {
		return err
	}
	defer st.Close()
	c := m.c
	ld := loader{c: c, st: st, cfg: Config{}}
	d := &Draft{Cfg: cfg, m: m, Count: cfg.Block - 1}
	local := map[int]bool{}
	for _, i := range cfg.Local {
		local[i] = true
	}
	for i := 0; i < cfg.Layers; i++ {
		p := fmt.Sprintf("layers.%d.", i)
		l := draftLayer{
			InNorm: ld.tensor(p + "input_layernorm.weight"), PostNorm: ld.tensor(p + "post_attention_layernorm.weight"),
			CtxNorm: ld.tensor(p + "dflare_norm.weight"),
			QT:      ld.dense(p + "self_attn.q_proj.weight"), KT: ld.dense(p + "self_attn.k_proj.weight"),
			VT: ld.dense(p + "self_attn.v_proj.weight"), KCtxT: ld.dense(p + "self_attn.k_proj_ctx.weight"),
			VCtxT: ld.dense(p + "self_attn.v_proj_ctx.weight"), OT: ld.dense(p + "self_attn.o_proj.weight"),
			QNorm: ld.tensor(p + "self_attn.q_norm.weight"), KNorm: ld.tensor(p + "self_attn.k_norm.weight"),
			KCtxNorm: ld.tensor(p + "self_attn.k_norm_ctx.weight"),
			GateT:    ld.dense(p + "mlp.gate_proj.weight"), UpT: ld.dense(p + "mlp.up_proj.weight"),
			DownT: ld.dense(p + "mlp.down_proj.weight"),
		}
		if local[i] {
			l.Window = cfg.Window
		}
		d.Layers = append(d.Layers, l)
	}
	d.Norm = ld.tensor("norm.weight")
	mask := ld.tensor("mask_embed")
	fuse := ld.f32("fuse_weight")
	d.W1 = ld.tensor("markov_head.markov_w1.weight")
	rnn := ld.f32("markov_head.rnn_proj.weight") // [3r, 2r+2H]
	d.RNNBias = ld.f32("markov_head.rnn_proj.bias")
	d.RNNNorm = ld.tensor("markov_head.dflare_norm.weight")
	out := ld.tensor("markov_head.hidden_out.weight") // [hidden, rank]
	outBias := ld.tensor("markov_head.hidden_out.bias")
	if ld.err != nil {
		return ld.err
	}
	r, H := cfg.Rank, cfg.Hidden
	if s := fuse.Shape(); len(s) != 2 || s[0] != cfg.Layers+1 || s[1] != len(cfg.Targets) {
		return fmt.Errorf("%s: fuse_weight is %v, want [%d %d]", DraftFile, s, cfg.Layers+1, len(cfg.Targets))
	}
	if s := rnn.Shape(); len(s) != 2 || s[0] != 3*r || s[1] != 2*r+2*H {
		return fmt.Errorf("%s: rnn_proj is %v, want [%d %d]", DraftFile, s, 3*r, 2*r+2*H)
	}
	d.Mask = c.Reshape(c.model(mask), 1, H)
	// As the reference: the mixing weights are rounded to the model dtype.
	alpha := c.f32(c.model(c.Softmax(fuse, 1)))
	for i := 0; i <= cfg.Layers; i++ {
		d.Mix = append(d.Mix, c.Slice(alpha, 0, i, i+1))
	}
	d.StateT = c.Transpose(c.Slice(rnn, 1, 0, 2*r), 1, 0)
	d.ContextT = c.Transpose(c.Slice(rnn, 1, 2*r, 2*r+2*H), 1, 0)

	// The fold, a slab of the vocabulary at a time: the head as floats is
	// 1.9 GB on the published model.
	outF, biasF := c.f32(out), c.Reshape(c.f32(outBias), H, 1)
	var fold, foldBias engine.Tensor
	const slab = 4096
	for from := 0; from < m.Cfg.Vocab; from += slab {
		to := min(from+slab, m.Cfg.Vocab)
		var rows engine.Tensor
		switch h := m.Head.(type) {
		case *qwen.QuantLinear:
			rows = c.Dequantize(c.Slice(h.Weight, 0, from, to), c.Slice(h.Scales, 0, from, to), c.Slice(h.Biases, 0, from, to), h.GroupSize, h.Bits)
		case denseProjection:
			rows = c.Slice(c.Transpose(h.wT, 1, 0), 0, from, to)
		default:
			return fmt.Errorf("the draft head cannot read the model's output head")
		}
		rows = c.f32(rows)
		w, bb := c.model(c.MatMul(rows, outF)), c.model(c.MatMul(rows, biasF))
		c.Eval(w, bb)
		if fold == nil {
			fold, foldBias = w, bb
		} else {
			fold, foldBias = c.Concat(fold, w, 0), c.Concat(foldBias, bb, 0)
		}
	}
	d.FoldT = c.Transpose(fold, 1, 0)
	d.FoldBias = c.Reshape(foldBias, 1, m.Cfg.Vocab)
	c.Eval(d.FoldT, d.FoldBias)
	m.draft = d
	return nil
}

// DraftCache is the draft head's state for a session: each layer's keys
// and values over the context read so far, and the model's states for the
// tokens read since the last proposal, which the next one takes in.
type DraftCache struct {
	K, V  []engine.Tensor // per draft layer, [1, kv heads, T, head dim]
	first []int           // the position of each layer's oldest key
	// taken is the number of tokens whose states the layers hold; pending
	// [1, n, targets*hidden] are the states of the n after them.
	taken    int
	pending  engine.Tensor
	pendingN int

	marks map[int]draftMark
}

type draftMark struct {
	k, v     []engine.Tensor
	first    []int
	taken    int
	pending  engine.Tensor
	pendingN int
}

func (c *DraftCache) Len() int { return c.taken + c.pendingN }

// CanTruncate: always. Cut back past what it kept, the head goes on with
// the context it has left, and its proposals are verified as ever.
func (c *DraftCache) CanTruncate(int) bool { return true }

func (c *DraftCache) Truncate(b engine.Backend, n int) {
	if n >= c.Len() {
		return
	}
	for at := range c.marks {
		if at > n {
			delete(c.marks, at)
		}
	}
	if m, ok := c.marks[n]; ok {
		c.K, c.V = append([]engine.Tensor{}, m.k...), append([]engine.Tensor{}, m.v...)
		c.first = append([]int{}, m.first...)
		c.taken, c.pending, c.pendingN = m.taken, m.pending, m.pendingN
		return
	}
	if n >= c.taken {
		c.pendingN = n - c.taken
		if c.pendingN == 0 {
			c.pending = nil
		} else {
			c.pending = b.Slice(c.pending, 1, 0, c.pendingN)
		}
		return
	}
	c.pending, c.pendingN, c.taken = nil, 0, n
	for i := range c.K {
		if c.K[i] == nil {
			continue
		}
		if keep := n - c.first[i]; keep > 0 {
			c.K[i], c.V[i] = b.Slice(c.K[i], 2, 0, keep), b.Slice(c.V[i], 2, 0, keep)
		} else {
			c.K[i], c.V[i], c.first[i] = nil, nil, n
		}
	}
}

func (c *DraftCache) Mark() {
	if c.Len() == 0 {
		return
	}
	if c.marks == nil {
		c.marks = map[int]draftMark{}
	}
	c.marks[c.Len()] = draftMark{
		k: append([]engine.Tensor{}, c.K...), v: append([]engine.Tensor{}, c.V...),
		first: append([]int{}, c.first...), taken: c.taken, pending: c.pending, pendingN: c.pendingN,
	}
}

func (c *DraftCache) Forget(n int) { delete(c.marks, n) }

func (c *DraftCache) Tensors() []engine.Tensor {
	var ts []engine.Tensor
	add := func(k, v []engine.Tensor, pending engine.Tensor) {
		for _, t := range append(append(append([]engine.Tensor{}, k...), v...), pending) {
			if t != nil {
				ts = append(ts, t)
			}
		}
	}
	add(c.K, c.V, c.pending)
	for _, m := range c.marks {
		add(m.k, m.v, m.pending)
	}
	return ts
}

// record adds the model's states for the tokens of one pass.
func (c *DraftCache) record(b engine.Backend, states engine.Tensor) {
	if c.pending == nil {
		c.pending = states
	} else {
		c.pending = b.Concat(c.pending, states, 1)
	}
	c.pendingN += states.Shape()[1]
}

// heads splits x [1, n, heads*dim] into [1, heads, n, dim].
func (d *Draft) heads(x engine.Tensor, heads int) engine.Tensor {
	c := d.m.c
	return c.Transpose(c.Reshape(x, 1, x.Shape()[1], heads, d.Cfg.HeadDim), 0, 2, 1, 3)
}

func (d *Draft) rope(x engine.Tensor, offset int) engine.Tensor {
	return d.m.c.RoPE(x, d.Cfg.HeadDim, false, float32(d.Cfg.RopeTheta), 1, offset)
}

// mix is the model's states [1, n, targets*hidden] as the context one part
// of the head reads: the four depths weighted, then normalized.
func (d *Draft) mix(states engine.Tensor, row int, norm engine.Tensor) engine.Tensor {
	c := d.m.c
	n, H := states.Shape()[1], d.Cfg.Hidden
	parts := c.f32(c.Reshape(states, 1, n, len(d.Cfg.Targets), H))
	mixed := c.Reshape(c.MatMul(d.Mix[row], parts), 1, n, H)
	return c.model(c.RMSNorm(mixed, norm, float32(d.Cfg.Eps)))
}

// Propose drafts the Block-1 tokens after tok (qwen.Speculator).
func (d *Draft) Propose(_ engine.Backend, tok int32, caches []qwen.LayerCache) []int32 {
	c, cfg, m := d.m.c, d.Cfg, d.m
	dc, ok := caches[len(caches)-1].(*DraftCache)
	if !ok || dc.pending == nil {
		return nil
	}
	if dc.K == nil {
		dc.K, dc.V, dc.first = make([]engine.Tensor, cfg.Layers), make([]engine.Tensor, cfg.Layers), make([]int, cfg.Layers)
	}
	states, n := dc.pending, dc.pendingN
	H, hd := cfg.Hidden, cfg.HeadDim
	eps := float32(cfg.Eps)
	scale := float32(1 / math.Sqrt(float64(hd)))
	count, places := min(max(d.Count, 1), cfg.Block-1), cfg.Block-1

	// The block: the token, then a place to fill for each of its others.
	x := m.EmbedNorm.Forward(c, c.Reshape(m.Embed.Forward(c, c.FromInt32([]int32{tok}, 1)), 1, 1, H))
	ones := make([]float32, places)
	for i := range ones {
		ones[i] = 1
	}
	x = c.Concat(x, c.Reshape(c.MatMul(c.model(c.FromFloats(ones, places, 1)), d.Mask), 1, places, H), 1)
	column := c.model(c.FromFloats(ones[:count], count, 1))

	var residual engine.Tensor
	for i, l := range d.Layers {
		summed := c.f32(x)
		if residual != nil {
			summed = c.Add(summed, c.f32(residual))
		}
		residual = c.model(summed)
		h := c.model(c.RMSNorm(summed, l.InNorm, eps))

		// The context goes into the layer's own keys and values, at the
		// positions of its tokens; a sliding layer keeps the last Window.
		ctx, at := d.mix(states, i, l.CtxNorm), dc.taken
		if l.Window > 0 && n > l.Window {
			ctx, at = c.Slice(ctx, 1, n-l.Window, n), dc.taken+n-l.Window
		}
		k := d.rope(c.RMSNorm(d.heads(c.MatMul(ctx, l.KCtxT), cfg.KVHeads), l.KCtxNorm, eps), at)
		v := d.heads(c.MatMul(ctx, l.VCtxT), cfg.KVHeads)
		if dc.K[i] == nil {
			dc.K[i], dc.V[i], dc.first[i] = k, v, at
		} else {
			dc.K[i], dc.V[i] = c.Concat(dc.K[i], k, 2), c.Concat(dc.V[i], v, 2)
		}
		if T := dc.K[i].Shape()[2]; l.Window > 0 && T > l.Window {
			dc.K[i], dc.V[i] = c.Slice(dc.K[i], 2, T-l.Window, T), c.Slice(dc.V[i], 2, T-l.Window, T)
			dc.first[i] += T - l.Window
		}

		// Every place of the block sees the context and the whole block.
		end := dc.taken + n
		q := d.rope(c.RMSNorm(d.heads(c.MatMul(h, l.QT), cfg.Heads), l.QNorm, eps), end)
		bk := d.rope(c.RMSNorm(d.heads(c.MatMul(h, l.KT), cfg.KVHeads), l.KNorm, eps), end)
		keys := c.Concat(dc.K[i], bk, 2)
		values := c.Concat(dc.V[i], d.heads(c.MatMul(h, l.VT), cfg.KVHeads), 2)
		a := c.SDPA(q, keys, values, scale, false)
		a = c.MatMul(c.Reshape(c.Transpose(a, 0, 2, 1, 3), 1, cfg.Block, cfg.Heads*hd), l.OT)

		summed = c.Add(c.f32(a), c.f32(residual))
		residual = c.model(summed)
		h = c.model(c.RMSNorm(summed, l.PostNorm, eps))
		gate, up := c.f32(c.MatMul(h, l.GateT)), c.f32(c.MatMul(h, l.UpT))
		x = c.MatMul(c.model(c.Mul(c.SiLU(gate), up)), l.DownT)
	}
	dc.taken, dc.pending, dc.pendingN = dc.taken+n, nil, 0

	x = c.model(c.RMSNorm(c.Add(c.f32(x), c.f32(residual)), d.Norm, eps))
	x = c.Slice(x, 1, 1, 1+count)                                    // [1, count, hidden]
	base := c.model(m.Head.Forward(c, x))                            // [1, count, vocab]
	last := d.mix(c.Slice(states, 1, n-1, n), cfg.Layers, d.RNNNorm) // [1, 1, hidden]
	spread := c.Reshape(c.MatMul(column, c.Reshape(last, 1, H)), 1, count, H)
	context := c.Add(c.MatMul(c.f32(c.Concat(x, spread, 2)), d.ContextT), d.RNNBias) // [1, count, 3*rank]

	r := cfg.Rank
	state := c.model(c.FromFloats(make([]float32, r), 1, r))
	prev := tok
	out := make([]int32, 0, count)
	for i := 0; i < count; i++ {
		in := c.f32(c.Concat(state, c.TakeAxis(d.W1, c.FromInt32([]int32{prev}, 1), 0), 1))
		raw := c.model(c.Add(c.MatMul(in, d.StateT), c.Reshape(c.Slice(context, 1, i, i+1), 1, 3*r)))
		gate := c.model(c.Sigmoid(c.f32(c.Slice(raw, 1, 0, r))))
		candidate := c.model(c.Tanh(c.f32(c.Slice(raw, 1, r, 2*r))))
		state = c.Add(c.Mul(gate, state), c.Mul(c.ScalarAdd(c.ScalarMul(gate, -1), 1), candidate))
		latent := c.model(c.Tanh(c.f32(c.Slice(raw, 1, 2*r, 3*r))))
		bias := c.Add(c.MatMul(latent, d.FoldT), d.FoldBias) // [1, vocab]
		logits := c.f32(c.Add(c.Reshape(c.Slice(base, 1, i, i+1), 1, m.Cfg.Vocab), bias))
		if cap := m.Cfg.HeadCap; cap > 0 {
			logits = c.ScalarMul(c.Tanh(c.ScalarMul(logits, 1/cap)), cap)
		}
		prev = c.Ints(c.Argmax(c.Slice(logits, 1, 0, m.Cfg.UnpaddedVocab), 1))[0]
		out = append(out, prev)
	}
	return out
}

// SetDraftCount sets how many tokens the draft head proposes at a time
// (1 to its block less one); no effect without a draft head.
func (m *Model) SetDraftCount(n int) {
	if m.draft != nil {
		m.draft.Count = min(max(n, 1), m.draft.Cfg.Block-1)
	}
}
