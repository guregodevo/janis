package qwen35

import (
	"math"

	"memdoor/llm/engine"
	"memdoor/llm/qwen"
)

// Multimodal RoPE (mRoPE), the language model's positions when an image is
// in the prompt. Text tokens carry one position on all three streams
// (time, row, column), so their rotation is plain RoPE — the path the text
// engine already runs. An image's tokens carry the grid: time 0, row and
// column of the merged token, all offset by the text before them; the text
// after the image continues from the largest position used plus one, so an
// image of 112 tokens advances the position count by only 14. Each rotary
// frequency reads one stream, interleaved: index i%3 == 1 reads the row,
// i%3 == 2 the column (up to the section's size), the rest the time.

// Qwen3.5's special tokens and rope layout (config.json; the family's
// constants, not per-checkpoint knobs).
const (
	ImageTokenID       = 248056
	VisionStartTokenID = 248053
	VisionEndTokenID   = 248054
)

var mropeSection = [3]int{11, 11, 10}

// Grid is an image's merged-token grid (t, h, w) as the language model sees
// it — the patch grid divided by the merge size on h and w.
type Grid struct{ T, H, W int }

// MRoPEPositions builds the three position streams for a prompt whose
// image-pad runs correspond to grids, in order, and returns the delta
// between the position after the prompt and its token count (whisper's
// rope_deltas). ImagePositions is whisper's get_rope_index for images.
func MRoPEPositions(ids []int32, grids []Grid) (pos [3][]int32, delta int) {
	L := len(ids)
	for s := range pos {
		pos[s] = make([]int32, L)
	}
	next := int32(0) // the next free position
	i := 0
	g := 0
	for i < L {
		if ids[i] != ImageTokenID || g >= len(grids) {
			for s := range pos {
				pos[s][i] = next
			}
			next++
			i++
			continue
		}
		grid := grids[g]
		g++
		maxPos := next
		for t := 0; t < grid.T; t++ {
			for h := 0; h < grid.H; h++ {
				for w := 0; w < grid.W; w++ {
					if i >= L {
						break
					}
					pos[0][i], pos[1][i], pos[2][i] = next+int32(t), next+int32(h), next+int32(w)
					for _, p := range []int32{pos[0][i], pos[1][i], pos[2][i]} {
						if p > maxPos {
							maxPos = p
						}
					}
					i++
				}
			}
		}
		next = maxPos + 1
	}
	return pos, int(next) - L
}

// streamFor says which position stream rotary frequency i reads.
func streamFor(i int) int {
	if i%3 == 1 && i < mropeSection[1]*3 {
		return 1
	}
	if i%3 == 2 && i < mropeSection[2]*3 {
		return 2
	}
	return 0
}

// MRoPECosSin builds cos and sin [1, 1, L, ropeDims] for the streams.
func MRoPECosSin(b engine.Backend, pos [3][]int32, ropeDims int, base float32) (cos, sin engine.Tensor) {
	L := len(pos[0])
	nf := ropeDims / 2
	inv := make([]float64, nf)
	for i := range inv {
		inv[i] = 1 / math.Pow(float64(base), float64(2*i)/float64(ropeDims))
	}
	cs := make([]float32, L*ropeDims)
	sn := make([]float32, L*ropeDims)
	for t := 0; t < L; t++ {
		for i := 0; i < nf; i++ {
			a := float64(pos[streamFor(i)][t]) * inv[i]
			c, s := float32(math.Cos(a)), float32(math.Sin(a))
			cs[t*ropeDims+i], sn[t*ropeDims+i] = c, s
			cs[t*ropeDims+nf+i], sn[t*ropeDims+nf+i] = c, s
		}
	}
	return b.Cast(b.FromFloats(cs, 1, 1, L, ropeDims), engine.F16), b.Cast(b.FromFloats(sn, 1, 1, L, ropeDims), engine.F16)
}

// applyPartialRotary rotates the first ropeDims of x [1, H, L, hd] with the
// given tables and leaves the rest untouched.
func applyPartialRotary(b engine.Backend, x, cos, sin engine.Tensor, ropeDims int) engine.Tensor {
	hd := x.Shape()[3]
	rot := applyRotary(b, b.Slice(x, 3, 0, ropeDims), cos, sin)
	if ropeDims == hd {
		return rot
	}
	return b.Concat(rot, b.Slice(x, 3, ropeDims, hd), 3)
}

// ForwardMRoPE is Attention.Forward with explicit per-token rotary tables
// (an image prefill); the cache fills exactly as in the plain path.
func (a *Attention) ForwardMRoPE(b engine.Backend, x engine.Tensor, cache *qwen.KVCache, cos, sin engine.Tensor) engine.Tensor {
	L := x.Shape()[1]
	qg := b.Reshape(a.QProj.Forward(b, x), 1, L, a.NHeads, 2*a.HeadDim)
	q := b.Slice(qg, 3, 0, a.HeadDim)
	gate := b.Slice(qg, 3, a.HeadDim, 2*a.HeadDim)
	k := b.Reshape(a.KProj.Forward(b, x), 1, L, a.NKVHeads, a.HeadDim)
	q = a.QNorm.Forward(b, q)
	k = a.KNorm.Forward(b, k)
	q = b.Transpose(q, 0, 2, 1, 3)
	k = b.Transpose(k, 0, 2, 1, 3)
	v := b.Transpose(b.Reshape(a.VProj.Forward(b, x), 1, L, a.NKVHeads, a.HeadDim), 0, 2, 1, 3)
	q = applyPartialRotary(b, q, cos, sin, a.RopeDims)
	k = applyPartialRotary(b, k, cos, sin, a.RopeDims)
	if cache != nil {
		k, v = cache.Update(b, k, v)
	}
	out := b.SDPA(q, k, v, a.Scale, L > 1)
	out = b.Reshape(b.Transpose(out, 0, 2, 1, 3), 1, L, a.NHeads*a.HeadDim)
	out = b.Mul(out, b.Sigmoid(b.Reshape(gate, 1, L, a.NHeads*a.HeadDim)))
	return a.OProj.Forward(b, out)
}

// ForwardCachedMRoPE is Block.ForwardCached with rotary tables.
func (bl *Block) ForwardCachedMRoPE(b engine.Backend, x engine.Tensor, cache qwen.LayerCache, cos, sin engine.Tensor) engine.Tensor {
	if bl.Linear != nil {
		return bl.ForwardCached(b, x, cache, 0) // positions do not reach DeltaNet
	}
	var kv *qwen.KVCache
	if cache != nil {
		kv = cache.(*qwen.KVCache)
	}
	h := bl.InputNorm.Forward(b, x)
	h = b.Add(x, bl.Attn.ForwardMRoPE(b, h, kv, cos, sin))
	return b.Add(h, bl.FFN.Forward(b, bl.PostNorm.Forward(b, h)))
}

// PrefillWithImages runs the prompt through the text stack with the merged
// image features standing in for the <|image_pad|> tokens and mRoPE
// positions, filling caches. It returns the last-position logits
// [1, 1, vocab] and the rope delta the following decode must add to its
// token offset.
func (m *Model) PrefillWithImages(b engine.Backend, ids []int32, features []engine.Tensor, grids []Grid, caches []qwen.LayerCache) (engine.Tensor, int) {
	L := len(ids)
	h := m.Embed.Forward(b, b.FromInt32(ids, L)) // [L, hidden]
	var rows []int32
	for i, id := range ids {
		if id == ImageTokenID {
			rows = append(rows, int32(i))
		}
	}
	if len(rows) > 0 && len(features) > 0 {
		feat := features[0]
		for _, f := range features[1:] {
			feat = b.Concat(feat, f, 0)
		}
		if n := feat.Shape()[0]; n < len(rows) {
			rows = rows[:n]
		}
		rs, ok := b.(engine.RowScatterer)
		if !ok {
			panic("qwen35: the backend cannot scatter rows (image features need engine.RowScatterer)")
		}
		h = rs.ScatterRows(h, rows, b.Cast(feat, engine.F16))
	}
	h = b.Reshape(h, 1, L, m.Cfg.Hidden)
	pos, delta := MRoPEPositions(ids, grids)
	cos, sin := MRoPECosSin(b, pos, m.Blocks[ropeLayer(m.Cfg)].Attn.RopeDims, m.Cfg.RopeBase)
	for i, blk := range m.Blocks {
		var c qwen.LayerCache
		if caches != nil {
			c = caches[i]
		}
		h = blk.ForwardCachedMRoPE(b, h, c, cos, sin)
	}
	h = m.Norm.Forward(b, h)
	h = b.Slice(h, 1, L-1, L)
	return m.head(b, h), delta
}

// ropeLayer is the first softmax-attention layer (they all share RopeDims).
func ropeLayer(cfg qwen.Config) int {
	for i := 0; i < cfg.Layers; i++ {
		if !isLinear(i, cfg) {
			return i
		}
	}
	return 0
}
