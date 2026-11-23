package qwen35

import (
	"math"
	"os"
	"strconv"

	"github.com/guregodevo/janis/engine"
	"github.com/guregodevo/janis/qwen"
	"github.com/guregodevo/janis/safetensors"
)

// Attention is Qwen3.5's gated softmax attention (every
// full_attention_interval-th layer): q_proj emits [query, gate] per head, RoPE
// rotates only the first partial_rotary_factor of head_dim, and the SDPA
// output is sigmoid-gated before o_proj.
type Attention struct {
	QProj, KProj, VProj, OProj *qwen.QuantLinear
	QNorm, KNorm               *qwen.RMSNorm
	NHeads, NKVHeads, HeadDim  int
	RopeDims                   int
	RopeBase, Scale            float32
}

// LoadAttention loads the projections under prefix (".../self_attn").
func LoadAttention(b engine.Backend, st *safetensors.File, prefix string, cfg qwen.Config) (*Attention, error) {
	load := func(name string) (*qwen.QuantLinear, error) {
		gs, bits := cfg.QuantFor(prefix + "." + name)
		return qwen.LoadQuantLinear(b, st, prefix+"."+name, gs, bits, false)
	}
	a := &Attention{
		NHeads: cfg.NHeads, NKVHeads: cfg.NKVHeads, HeadDim: cfg.HeadDim,
		RopeDims: int(float32(cfg.HeadDim) * cfg.PartialRotary),
		RopeBase: cfg.RopeBase,
		Scale:    float32(1.0 / math.Sqrt(float64(cfg.HeadDim))),
	}
	var err error
	if a.QProj, err = load("q_proj"); err != nil {
		return nil, err
	}
	if a.KProj, err = load("k_proj"); err != nil {
		return nil, err
	}
	if a.VProj, err = load("v_proj"); err != nil {
		return nil, err
	}
	if a.OProj, err = load("o_proj"); err != nil {
		return nil, err
	}
	if a.QNorm, err = qwen.LoadRMSNorm(b, st, prefix+".q_norm", cfg.RMSEps); err != nil {
		return nil, err
	}
	if a.KNorm, err = qwen.LoadRMSNorm(b, st, prefix+".k_norm", cfg.RMSEps); err != nil {
		return nil, err
	}
	return a, nil
}

// Forward runs gated attention on x [1, L, hidden] with the layer's KV cache
// at the given RoPE offset -> [1, L, hidden].
func (a *Attention) Forward(b engine.Backend, x engine.Tensor, cache *qwen.KVCache, offset int) engine.Tensor {
	L := x.Shape()[1]
	q, gate, k, v := a.project(b, x, L)
	out := a.attend(b, q, k, v, cache, offset, L)
	return a.output(b, out, gate, L)
}

// ForwardBatch runs B conversations' next tokens, x [1, B, hidden], one row
// each: the projections once over the B rows, RoPE, the cache and attention
// per conversation at its own offset -> [1, B, hidden].
func (a *Attention) ForwardBatch(b engine.Backend, x engine.Tensor, caches []*qwen.KVCache, offsets []int) engine.Tensor {
	B := x.Shape()[1]
	q, gate, k, v := a.project(b, x, B)
	if ra, ok := b.(engine.RowAttention); ok {
		if kp, ok := b.(engine.KVPooler); ok {
			return a.output(b, a.attendPooled(b, ra, kp, q, k, v, caches, offsets), gate, B)
		}
		return a.output(b, a.attendRows(b, ra, q, k, v, caches, offsets), gate, B)
	}
	var out engine.Tensor
	for j := 0; j < B; j++ {
		o := a.attend(b, b.Slice(q, 1, j, j+1), b.Slice(k, 1, j, j+1), b.Slice(v, 1, j, j+1), caches[j], offsets[j], 1)
		if out == nil {
			out = o
		} else {
			out = b.Concat(out, o, 1)
		}
	}
	return a.output(b, out, gate, B)
}

// project is q, its gate, k and v for L rows, each [1, L, heads, hd].
func (a *Attention) project(b engine.Backend, x engine.Tensor, L int) (q, gate, k, v engine.Tensor) {
	// q_proj packs [query, gate] per head: [1, L, H, 2*hd] -> two [1, L, H, hd].
	qg := b.Reshape(a.QProj.Forward(b, x), 1, L, a.NHeads, 2*a.HeadDim)
	q = a.QNorm.Forward(b, b.Slice(qg, 3, 0, a.HeadDim))
	gate = b.Slice(qg, 3, a.HeadDim, 2*a.HeadDim)
	k = a.KNorm.Forward(b, b.Reshape(a.KProj.Forward(b, x), 1, L, a.NKVHeads, a.HeadDim))
	v = b.Reshape(a.VProj.Forward(b, x), 1, L, a.NKVHeads, a.HeadDim)
	return q, gate, k, v
}

// attend is one conversation's attention over L new rows at offset: q, k, v
// [1, L, heads, hd] -> [1, L, H*hd].
func (a *Attention) attend(b engine.Backend, q, k, v engine.Tensor, cache *qwen.KVCache, offset, L int) engine.Tensor {
	q = b.Transpose(q, 0, 2, 1, 3)
	k = b.Transpose(k, 0, 2, 1, 3)
	v = b.Transpose(v, 0, 2, 1, 3)

	// Partial RoPE: only the first RopeDims of head_dim rotate. (The text-only
	// path of Qwen3.5's mrope: all three position streams are the token index,
	// which reduces to plain RoPE over the partial dims — mlx-lm does the same.)
	q = b.RoPE(q, a.RopeDims, false, a.RopeBase, 1.0, offset)
	k = b.RoPE(k, a.RopeDims, false, a.RopeBase, 1.0, offset)

	if cache != nil {
		k, v = cache.Update(b, k, v)
	}

	out := b.SDPA(q, k, v, a.Scale, L > 1) // [1, H, L, hd]
	return b.Reshape(b.Transpose(out, 0, 2, 1, 3), 1, L, a.NHeads*a.HeadDim)
}

// attendRows is attention for B conversations' next rows on a backend
// that runs it as one op: q and k rotated in one RoPE each, row j at its
// conversation's offset, each row's k and v appended to its own cache, and
// every row's attention in one SDPARows. q, k, v [1, B, heads, hd] ->
// [1, B, H*hd].
func (a *Attention) attendRows(b engine.Backend, ra engine.RowAttention, q, k, v engine.Tensor, caches []*qwen.KVCache, offsets []int) engine.Tensor {
	B := q.Shape()[1]
	// [1, B, heads, hd] -> [1, heads, B, hd]: the rows as the tokens RoPE
	// rotates, each at its own position.
	q = ra.RoPEPositions(b.Transpose(q, 0, 2, 1, 3), a.RopeDims, false, a.RopeBase, 1.0, offsets)
	k = ra.RoPEPositions(b.Transpose(k, 0, 2, 1, 3), a.RopeDims, false, a.RopeBase, 1.0, offsets)
	v = b.Transpose(v, 0, 2, 1, 3)
	ks := make([]engine.Tensor, B)
	vs := make([]engine.Tensor, B)
	rp, _ := b.(engine.BatchStepReplayer)
	for j := 0; j < B; j++ {
		if rp != nil {
			rp.AppendRow(j) // a recorded step writes row j's caches at its own inputs
		}
		ks[j], vs[j] = caches[j].Update(b, b.Slice(k, 2, j, j+1), b.Slice(v, 2, j, j+1))
	}
	out := ra.SDPARows(b.Transpose(q, 2, 1, 0, 3), ks, vs, a.Scale) // [B, H, 1, hd]
	return b.Reshape(b.Transpose(out, 0, 2, 1, 3), 1, B, a.NHeads*a.HeadDim)
}

// poolSpare is the room a new pool has past its longest row, in tokens
// (JANIS_POOL_SPARE, 512 unset): a pool is made again, its rows copied,
// when a row fills it.
func poolSpare() int {
	if n, err := strconv.Atoi(os.Getenv("JANIS_POOL_SPARE")); err == nil && n > 0 {
		return n
	}
	return 512
}

// attendPooled is attention for B conversations' next rows with the B
// conversations' keys and values in one pool (engine.KVPooler): one RoPE
// each for q and k, one append, one attention for every row. A pool is
// made, the caches copied in, when the conversations are not in one in
// this order or it has no room. q, k, v [1, B, heads, hd] -> [1, B, H*hd].
func (a *Attention) attendPooled(b engine.Backend, ra engine.RowAttention, kp engine.KVPooler, q, k, v engine.Tensor, caches []*qwen.KVCache, offsets []int) engine.Tensor {
	B := len(caches)
	pool := sharedPool(caches, offsets)
	if pool == nil {
		ks, vs := make([]engine.Tensor, B), make([]engine.Tensor, B)
		room := 0
		for j, c := range caches {
			c.Own()
			ks[j], vs[j] = c.K, c.V
			room = max(room, offsets[j]+1)
		}
		pool = kp.PoolKV(ks, vs, a.NKVHeads, a.HeadDim, a.HeadDim, room+poolSpare())
		for j, c := range caches {
			c.Lend(pool, j, offsets[j])
		}
	}
	// [1, B, heads, hd] -> [1, heads, B, hd]: the rows as the tokens RoPE
	// rotates, each at its own position; then [B, heads, 1, hd].
	q = ra.RoPEPositions(b.Transpose(q, 0, 2, 1, 3), a.RopeDims, false, a.RopeBase, 1.0, offsets)
	k = ra.RoPEPositions(b.Transpose(k, 0, 2, 1, 3), a.RopeDims, false, a.RopeBase, 1.0, offsets)
	pool.Append(b.Transpose(k, 2, 1, 0, 3), b.Transpose(b.Transpose(v, 0, 2, 1, 3), 2, 1, 0, 3), offsets)
	lens := make([]int, B)
	for j, o := range offsets {
		lens[j] = o + 1
	}
	out := pool.Attend(b.Transpose(q, 2, 1, 0, 3), lens, a.Scale) // [B, H, 1, hd]
	for j, c := range caches {
		c.Grow(lens[j])
	}
	return b.Reshape(b.Transpose(out, 0, 2, 1, 3), 1, B, a.NHeads*a.HeadDim)
}

// sharedPool is the pool caches are lent to, rows in order, with room for
// this step; nil when they are not.
func sharedPool(caches []*qwen.KVCache, offsets []int) engine.Pool {
	p, _ := caches[0].Lent()
	if p == nil || p.Rows() != len(caches) {
		return nil
	}
	for j, c := range caches {
		q, r := c.Lent()
		if q != p || r != j || c.Len() != offsets[j] || offsets[j] >= p.Room() {
			return nil
		}
	}
	return p
}

// output gates the attention rows and projects them back: [1, L, H*hd] -> [1, L, hidden].
func (a *Attention) output(b engine.Backend, out, gate engine.Tensor, L int) engine.Tensor {
	out = b.Mul(out, b.Sigmoid(b.Reshape(gate, 1, L, a.NHeads*a.HeadDim)))
	return a.OProj.Forward(b, out)
}
