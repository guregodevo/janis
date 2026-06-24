package qwen

import (
	"math"
	"math/rand"
	"sort"

	"memdoor/llm/engine"
	"memdoor/llm/safetensors"
)

// SampleParams controls token sampling. Temp <= 0 means greedy.
type SampleParams struct {
	Temp float32
	TopP float32
	Seed uint64
	Stop map[int32]bool // stop generation when one of these is produced
	// OnToken, if set, is called for each emitted token (for streaming).
	OnToken func(tok int32)
}

// sampleToken draws a token from logits under temperature + nucleus (top-p).
func sampleToken(logits []float32, p SampleParams, rng *rand.Rand) int32 {
	if p.Temp <= 0 {
		best := 0
		for i := range logits {
			if logits[i] > logits[best] {
				best = i
			}
		}
		return int32(best)
	}

	n := len(logits)
	maxL := logits[0]
	for _, x := range logits {
		if x > maxL {
			maxL = x
		}
	}
	probs := make([]float64, n)
	inv := 1.0 / float64(p.Temp)
	var sum float64
	for i, x := range logits {
		e := math.Exp((float64(x) - float64(maxL)) * inv)
		probs[i] = e
		sum += e
	}
	for i := range probs {
		probs[i] /= sum
	}

	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return probs[idx[a]] > probs[idx[b]] })

	topP := float64(p.TopP)
	if topP <= 0 || topP > 1 {
		topP = 1.0
	}
	var cum float64
	cut := n
	for rank, id := range idx {
		cum += probs[id]
		if cum >= topP {
			cut = rank + 1
			break
		}
	}

	r := rng.Float64() * cum
	var acc float64
	for rank := 0; rank < cut; rank++ {
		acc += probs[idx[rank]]
		if r <= acc {
			return int32(idx[rank])
		}
	}
	return int32(idx[cut-1])
}

// SampledGenerate decodes nGen tokens with temperature/top-p sampling. Unlike
// the device-resident greedy path it reads logits to the host each step
// (needed to sample), trading some speed for correct sampling.
func SampledGenerate(b engine.Backend, prompt []int32, nGen, nLayers, vocab int, forward ForwardFunc, p SampleParams) []int32 {
	rng := rand.New(rand.NewSource(int64(p.Seed)))
	sw, _ := b.(engine.Sweeper)
	if sw != nil {
		sw.PinAll()
	}
	caches := make([]*KVCache, nLayers)
	for i := range caches {
		caches[i] = &KVCache{}
	}

	logits := forward(b, b.FromInt32(prompt, len(prompt)), len(prompt), 0, caches)
	tok := sampleToken(b.Floats(logits), p, rng) // forward returns last-position logits
	offset := len(prompt)
	if sw != nil {
		pinCaches(sw, caches)
		sw.Sweep()
	}

	var out []int32
	for {
		if p.Stop[tok] {
			break
		}
		out = append(out, tok)
		if p.OnToken != nil {
			p.OnToken(tok)
		}
		if len(out) >= nGen {
			break
		}
		old := cacheTensors(caches)
		logits = forward(b, b.FromInt32([]int32{tok}, 1), 1, offset, caches)
		tok = sampleToken(b.Floats(logits), p, rng)
		offset++
		if sw != nil {
			pinCaches(sw, caches)
			sw.Unpin(old...)
			sw.Sweep()
		}
	}
	return out
}

// Model is a full Qwen2 causal LM. The output projection is the tied embedding
// (LMHead nil) or a separate quantized lm_head.
type Model struct {
	Embed   *QuantEmbedding
	Blocks  []*Block
	Norm    *RMSNorm
	LMHead  *QuantLinear // nil when embeddings are tied
	Cfg     Config
	Experts *ExpertStore // non-nil for MoE models (offloaded experts)
}

// moeExpertBudgetBytes bounds resident expert weights for MoE models. ~9 GB
// leaves room for non-expert weights + KV + activations inside 16 GB.
const moeExpertBudgetBytes = 9 << 30

// LoadModel loads the embedding, all decoder blocks, the final norm, and (when
// untied) the lm_head. For MoE models it builds an ExpertStore so expert weights
// are streamed on demand instead of loaded resident.
func LoadModel(b engine.Backend, st *safetensors.File, cfg Config) (*Model, error) {
	emb, err := LoadQuantEmbedding(b, st, "model.embed_tokens", cfg.GroupSize, cfg.Bits)
	if err != nil {
		return nil, err
	}
	var store *ExpertStore
	if cfg.NumExperts > 0 {
		store = NewExpertStore(b, st, cfg.GroupSize, cfg.Bits, moeExpertBudgetBytes)
	}
	blocks := make([]*Block, cfg.Layers)
	for i := range blocks {
		if blocks[i], err = LoadBlock(b, st, i, cfg, store); err != nil {
			return nil, err
		}
	}
	norm, err := LoadRMSNorm(b, st, "model.norm", cfg.RMSEps)
	if err != nil {
		return nil, err
	}
	m := &Model{Embed: emb, Blocks: blocks, Norm: norm, Cfg: cfg, Experts: store}
	if !cfg.TieWordEmbeddings {
		if m.LMHead, err = LoadQuantLinear(b, st, "lm_head", cfg.GroupSize, cfg.Bits, false); err != nil {
			return nil, err
		}
	}
	return m, nil
}

// head applies the output projection (tied embedding or separate lm_head).
func (m *Model) head(b engine.Backend, h engine.Tensor) engine.Tensor {
	if m.LMHead != nil {
		return m.LMHead.Forward(b, h)
	}
	return m.Embed.AsLinear(b, h)
}

// Logits runs the full forward pass over token ids and returns logits
// [1, len(ids), vocab].
func (m *Model) Logits(b engine.Backend, ids []int32) engine.Tensor {
	idsT := b.FromInt32(ids, len(ids))
	h := b.Reshape(m.Embed.Forward(b, idsT), 1, len(ids), m.Cfg.Hidden)
	for _, blk := range m.Blocks {
		h = blk.Forward(b, h)
	}
	h = m.Norm.Forward(b, h)
	return m.head(b, h)
}

// forwardCachedT runs the forward pass over a device-resident index tensor
// (length seqLen) through the KV caches at the given RoPE offset, returning
// logits [1, seqLen, vocab]. Keeping ids on-device lets the decode loop feed
// each predicted token straight back in without a host round-trip.
func (m *Model) forwardCachedT(b engine.Backend, idsT engine.Tensor, seqLen, offset int, caches []*KVCache) engine.Tensor {
	h := b.Reshape(m.Embed.Forward(b, idsT), 1, seqLen, m.Cfg.Hidden)
	for i, blk := range m.Blocks {
		h = blk.ForwardCached(b, h, caches[i], offset)
	}
	h = m.Norm.Forward(b, h)
	// Only the last position's logits are ever used — project just that one.
	h = b.Slice(h, 1, seqLen-1, seqLen)
	return m.head(b, h)
}

// Generate greedily decodes nGen tokens after the prompt. The decode loop is
// device-resident: each predicted token is fed back as a GPU index tensor with
// no host round-trip, and (when supported) async eval keeps the GPU queue full
// while Go builds the next step. Intermediates are swept each step so memory
// stays flat. Tokens are collected to the host once at the end.
func (m *Model) Generate(b engine.Backend, prompt []int32, nGen int) []int32 {
	return GreedyGenerate(b, prompt, nGen, m.Cfg.Layers, m.forwardCachedT)
}

// GenerateSampled decodes with temperature/top-p sampling.
func (m *Model) GenerateSampled(b engine.Backend, prompt []int32, nGen int, p SampleParams) []int32 {
	return SampledGenerate(b, prompt, nGen, m.Cfg.Layers, m.Cfg.Vocab, m.forwardCachedT, p)
}

// NewSession returns a prefix-reusing session for multi-turn serving.
func (m *Model) NewSession() *Session {
	return &Session{NLayers: m.Cfg.Layers, Vocab: m.Cfg.Vocab, Forward: m.forwardCachedT}
}

// ForwardFunc computes logits [1, seqLen, vocab] for a device index tensor
// through per-layer caches at a RoPE offset.
type ForwardFunc func(b engine.Backend, idsT engine.Tensor, seqLen, offset int, caches []*KVCache) engine.Tensor

// GreedyGenerate is the shared device-resident decode loop, parameterized by an
// architecture's forward function, so Qwen and Gemma reuse the same KV-cache /
// Pin-Sweep / async-eval machinery.
func GreedyGenerate(b engine.Backend, prompt []int32, nGen, nLayers int, forward ForwardFunc) []int32 {
	sw, _ := b.(engine.Sweeper)
	ae, _ := b.(engine.AsyncEvaler)
	if sw != nil {
		sw.PinAll() // protect the weights
	}

	caches := make([]*KVCache, nLayers)
	for i := range caches {
		caches[i] = &KVCache{}
	}

	// Prefill (host prompt) -> first token. The one host sync of the whole run.
	logits := forward(b, b.FromInt32(prompt, len(prompt)), len(prompt), 0, caches)
	host := b.Ints(b.Argmax(logits, 2))
	tokT := b.FromInt32([]int32{host[len(host)-1]}, 1) // device [1]
	offset := len(prompt)
	if sw != nil {
		pinCaches(sw, caches)
		sw.Pin(tokT)
		sw.Sweep()
	}

	devToks := make([]engine.Tensor, 0, nGen)
	devToks = append(devToks, tokT)
	for len(devToks) < nGen {
		old := cacheTensors(caches)

		logits = forward(b, tokT, 1, offset, caches) // [1,1,vocab]
		tokT = b.Reshape(b.Argmax(logits, 2), 1)     // device [1] token
		devToks = append(devToks, tokT)
		offset++

		if ae != nil {
			ae.AsyncEval(append(cacheTensors(caches), tokT)...)
		}
		if sw != nil {
			pinCaches(sw, caches)
			sw.Pin(tokT)     // keep every device token for the final readback
			sw.Unpin(old...) // release the replaced caches
			sw.Sweep()
		}
	}

	// Stack all device tokens and copy to the host in one shot.
	acc := devToks[0]
	for _, t := range devToks[1:] {
		acc = b.Concat(acc, t, 0)
	}
	return b.Ints(acc)[:nGen]
}

func cacheTensors(caches []*KVCache) []engine.Tensor {
	ts := make([]engine.Tensor, 0, 2*len(caches))
	for _, c := range caches {
		ts = append(ts, c.K, c.V)
	}
	return ts
}

func pinCaches(sw engine.Sweeper, caches []*KVCache) {
	for _, c := range caches {
		sw.Pin(c.K, c.V)
	}
}
