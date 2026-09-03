package whisper

import (
	"fmt"
	"math"

	"memdoor/llm/engine"
	"memdoor/llm/qwen"
	"memdoor/llm/safetensors"
)

// The text decoder of whisper-large-v3-turbo: token + learned positional
// embeddings, 4 pre-LN blocks of causal self-attention (with a KV cache),
// cross-attention over the encoder output, and an MLP; final LayerNorm and
// the tied embedding as output projection. F16 throughout, like the encoder.

// DecoderConfig is whisper-large-v3-turbo's text side.
type DecoderConfig struct {
	NVocab, NCtx, NState, NHead, NLayer int
	LNEps                               float32
}

var TurboDecoder = DecoderConfig{NVocab: VocabSize, NCtx: 448, NState: 1280, NHead: 20, NLayer: 4, LNEps: 1e-5}

type decoderBlock struct {
	attnLN, crossLN, mlpLN *layerNorm
	q, k, v, out           *linear
	cq, ck, cv, cout       *linear
	mlp1, mlp2             *linear
}

type Decoder struct {
	cfg     DecoderConfig
	tokEmb  engine.Tensor // [V, D]
	tokEmbT engine.Tensor // [D, V] — the tied output projection
	pos     engine.Tensor // [NCtx, D]
	blks    []*decoderBlock
	ln      *layerNorm
}

// DecoderState is one chunk's decoding state: the cross-attention keys and
// values (computed once from the audio) and the growing self-attention cache.
type DecoderState struct {
	crossK, crossV []engine.Tensor // per block [1, H, NCtxAudio, dh]
	selfK, selfV   []engine.Tensor // per block [1, H, n, dh]; nil before the first step
	n              int             // tokens fed so far
}

// LoadDecoder reads the decoder weights from the model's safetensors.
func LoadDecoder(b engine.Backend, st *safetensors.File, cfg DecoderConfig) (*Decoder, error) {
	d := &Decoder{cfg: cfg}
	var err error
	if d.tokEmb, err = qwen.LoadTensor(b, st, "decoder.token_embedding.weight"); err != nil {
		return nil, err
	}
	d.tokEmbT = b.Transpose(d.tokEmb, 1, 0)
	b.Eval(d.tokEmbT)
	if d.pos, err = qwen.LoadTensor(b, st, "decoder.positional_embedding"); err != nil {
		return nil, err
	}
	for i := 0; i < cfg.NLayer; i++ {
		p := fmt.Sprintf("decoder.blocks.%d", i)
		blk := &decoderBlock{}
		loads := []struct {
			dst  **linear
			name string
			bias bool
		}{
			{&blk.q, ".attn.query", true}, {&blk.k, ".attn.key", false}, {&blk.v, ".attn.value", true}, {&blk.out, ".attn.out", true},
			{&blk.cq, ".cross_attn.query", true}, {&blk.ck, ".cross_attn.key", false}, {&blk.cv, ".cross_attn.value", true}, {&blk.cout, ".cross_attn.out", true},
			{&blk.mlp1, ".mlp1", true}, {&blk.mlp2, ".mlp2", true},
		}
		for _, l := range loads {
			if *l.dst, err = loadLinear(b, st, p+l.name, l.bias); err != nil {
				return nil, err
			}
		}
		if blk.attnLN, err = loadLayerNorm(b, st, p+".attn_ln", cfg.LNEps); err != nil {
			return nil, err
		}
		if blk.crossLN, err = loadLayerNorm(b, st, p+".cross_attn_ln", cfg.LNEps); err != nil {
			return nil, err
		}
		if blk.mlpLN, err = loadLayerNorm(b, st, p+".mlp_ln", cfg.LNEps); err != nil {
			return nil, err
		}
		d.blks = append(d.blks, blk)
	}
	if d.ln, err = loadLayerNorm(b, st, "decoder.ln", cfg.LNEps); err != nil {
		return nil, err
	}
	return d, nil
}

// splitHeads turns [L, H*dh] into [1, H, L, dh]; mergeHeads is its inverse.
func splitHeads(b engine.Backend, t engine.Tensor, L, H, dh int) engine.Tensor {
	return b.Transpose(b.Reshape(t, 1, L, H, dh), 0, 2, 1, 3)
}

func mergeHeads(b engine.Backend, t engine.Tensor, L, H, dh int) engine.Tensor {
	return b.Reshape(b.Transpose(t, 0, 2, 1, 3), L, H*dh)
}

// NewState prepares a chunk's cross-attention from the encoder output
// [NCtxAudio, NState].
func (d *Decoder) NewState(b engine.Backend, audio engine.Tensor) *DecoderState {
	L := audio.Shape()[0]
	H, dh := d.cfg.NHead, d.cfg.NState/d.cfg.NHead
	st := &DecoderState{
		crossK: make([]engine.Tensor, d.cfg.NLayer), crossV: make([]engine.Tensor, d.cfg.NLayer),
		selfK: make([]engine.Tensor, d.cfg.NLayer), selfV: make([]engine.Tensor, d.cfg.NLayer),
	}
	for i, blk := range d.blks {
		st.crossK[i] = splitHeads(b, blk.ck.forward(b, audio), L, H, dh)
		st.crossV[i] = splitHeads(b, blk.cv.forward(b, audio), L, H, dh)
		b.Eval(st.crossK[i], st.crossV[i])
	}
	if sw, ok := b.(engine.Sweeper); ok {
		sw.Pin(st.crossK...)
		sw.Pin(st.crossV...)
		sw.Sweep()
	}
	return st
}

// Reset drops the self-attention cache so the same audio can be decoded
// again from an empty prompt (language detection, then transcription).
func (st *DecoderState) Reset(b engine.Backend) {
	if sw, ok := b.(engine.Sweeper); ok {
		for i := range st.selfK {
			if st.selfK[i] != nil {
				sw.Unpin(st.selfK[i], st.selfV[i])
			}
		}
	}
	for i := range st.selfK {
		st.selfK[i], st.selfV[i] = nil, nil
	}
	st.n = 0
}

// Free releases the state's cross-attention tensors and self cache.
func (st *DecoderState) Free(b engine.Backend) {
	st.Reset(b)
	if sw, ok := b.(engine.Sweeper); ok {
		sw.Unpin(st.crossK...)
		sw.Unpin(st.crossV...)
		sw.Sweep()
	}
}

// ForwardAlign runs a whole token sequence without a cache — whisper's
// alignment pass — and returns the logits of every position (F32,
// [n*NVocab]) and, for each alignment layer (the last half of the blocks,
// whisper's default), the cross-attention scores before softmax (F32,
// [NHead*n*NCtxAudio]).
func (d *Decoder) ForwardAlign(b engine.Backend, st *DecoderState, tokens []int) (logits []float32, cross [][]float32) {
	n := len(tokens)
	H, dh := d.cfg.NHead, d.cfg.NState/d.cfg.NHead
	scale := float32(1 / math.Sqrt(float64(dh)))
	scale4 := float32(math.Pow(float64(dh), -0.25)) // whisper scales q and k each
	ids := make([]int32, n)
	for i, t := range tokens {
		ids[i] = int32(t)
	}
	x := b.TakeAxis(d.tokEmb, b.FromInt32(ids, n), 0)
	x = b.Add(x, b.Slice(d.pos, 0, 0, n))
	alignFrom := d.cfg.NLayer / 2
	for i, blk := range d.blks {
		h := blk.attnLN.forward(b, x)
		q := splitHeads(b, blk.q.forward(b, h), n, H, dh)
		k := splitHeads(b, blk.k.forward(b, h), n, H, dh)
		v := splitHeads(b, blk.v.forward(b, h), n, H, dh)
		o := b.SDPA(q, k, v, scale, n > 1)
		x = b.Add(x, blk.out.forward(b, mergeHeads(b, o, n, H, dh)))

		h = blk.crossLN.forward(b, x)
		q = splitHeads(b, blk.cq.forward(b, h), n, H, dh)
		if i >= alignFrom {
			qs := b.ScalarMul(q, scale4)
			ks := b.ScalarMul(st.crossK[i], scale4)
			qk := b.Cast(b.MatMul(qs, b.Transpose(ks, 0, 1, 3, 2)), engine.F32) // [1, H, n, T]
			w := b.Cast(b.Softmax(qk, 3), engine.F16)
			o = b.MatMul(w, st.crossV[i])
			cross = append(cross, b.Floats(qk))
		} else {
			o = b.SDPA(q, st.crossK[i], st.crossV[i], scale, false)
		}
		x = b.Add(x, blk.cout.forward(b, mergeHeads(b, o, n, H, dh)))
		x = b.Add(x, blk.mlp2.forward(b, b.Gelu(blk.mlp1.forward(b, blk.mlpLN.forward(b, x)))))
	}
	x = d.ln.forward(b, x)
	logits = b.Floats(b.Cast(b.MatMul(x, d.tokEmbT), engine.F32))
	if sw, ok := b.(engine.Sweeper); ok {
		sw.Sweep()
	}
	return logits, cross
}

// Step feeds tokens (the whole prompt on the first call, one token after)
// and returns the next-token logits, F32 [NVocab].
func (d *Decoder) Step(b engine.Backend, st *DecoderState, tokens []int) []float32 {
	n := len(tokens)
	H, dh := d.cfg.NHead, d.cfg.NState/d.cfg.NHead
	scale := float32(1 / math.Sqrt(float64(dh)))
	ids := make([]int32, n)
	for i, t := range tokens {
		ids[i] = int32(t)
	}
	x := b.TakeAxis(d.tokEmb, b.FromInt32(ids, n), 0) // [n, D]
	x = b.Add(x, b.Slice(d.pos, 0, st.n, st.n+n))
	var old []engine.Tensor
	for i, blk := range d.blks {
		h := blk.attnLN.forward(b, x)
		q := splitHeads(b, blk.q.forward(b, h), n, H, dh)
		k := splitHeads(b, blk.k.forward(b, h), n, H, dh)
		v := splitHeads(b, blk.v.forward(b, h), n, H, dh)
		if st.selfK[i] != nil {
			old = append(old, st.selfK[i], st.selfV[i])
			k = b.Concat(st.selfK[i], k, 2)
			v = b.Concat(st.selfV[i], v, 2)
		}
		st.selfK[i], st.selfV[i] = k, v
		o := b.SDPA(q, k, v, scale, n > 1)
		x = b.Add(x, blk.out.forward(b, mergeHeads(b, o, n, H, dh)))

		h = blk.crossLN.forward(b, x)
		q = splitHeads(b, blk.cq.forward(b, h), n, H, dh)
		o = b.SDPA(q, st.crossK[i], st.crossV[i], scale, false)
		x = b.Add(x, blk.cout.forward(b, mergeHeads(b, o, n, H, dh)))

		x = b.Add(x, blk.mlp2.forward(b, b.Gelu(blk.mlp1.forward(b, blk.mlpLN.forward(b, x)))))
	}
	x = d.ln.forward(b, x)
	last := b.Slice(x, 0, n-1, n) // [1, D]
	logits := b.Floats(b.Cast(b.MatMul(last, d.tokEmbT), engine.F32))
	st.n += n
	if sw, ok := b.(engine.Sweeper); ok {
		sw.Unpin(old...)
		sw.Pin(st.selfK...)
		sw.Pin(st.selfV...)
		sw.Sweep()
	}
	return logits
}
