package qwen

import (
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"sync"

	"github.com/guregodevo/janis/engine"
	"github.com/guregodevo/janis/safetensors"
)

// SampleParams controls token sampling. Temp <= 0 means greedy.
type SampleParams struct {
	Temp float32
	TopP float32
	// TopK (>0) keeps the K most likely tokens before the top-p cut, which is
	// then taken over those K. 0 keeps every token.
	TopK int
	Seed uint64
	Stop map[int32]bool // stop generation when one of these is produced
	// OnToken, if set, is called for each emitted token (for streaming).
	OnToken func(tok int32)
	// Cancel, if set, is polled between prefill chunks and before each decode
	// step; returning true stops generation early and returns the partial output.
	// Lets a client interrupt a long prompt (prefill) or a runaway decode.
	Cancel func() bool
	// Grammar, if set, constrains decoding: while it is Active the sampler only
	// draws tokens it Allows, and every committed token is fed back via Advance so
	// the grammar can track its state. nil (the default) leaves sampling untouched.
	Grammar Grammar
	// RepeatPenalty (>1) divides the logit of any token seen in the last RepeatLastN
	// generated tokens, discouraging repetition so the model doesn't fall into a
	// decode loop on free text. RepeatPenalty <= 1 disables it. Not applied while a
	// grammar is Active — tool-call JSON legitimately repeats structural tokens.
	RepeatPenalty float32
	RepeatLastN   int
}

// Grammar constrains sampling to tokens that keep the output within a formal
// structure (grammar-constrained decoding). Advance is called for every committed
// token so the grammar can follow along; Allows is consulted only when Active
// reports the grammar is currently constraining, keeping the free-run path fast.
type Grammar interface {
	Active() bool
	Allows(id int32) bool
	Advance(id int32)
	// InString reports whether the grammar is currently inside a string literal
	// (content), as opposed to structural position. The repetition penalty applies
	// to content (where a small model loops generating the same lines) but not to
	// structural tokens (JSON braces/quotes/commas legitimately repeat).
	InString() bool
}

// sampleToken draws a token from logits under temperature + nucleus (top-p).
// When a grammar is Active it forbids tokens the grammar disallows: greedy picks
// the highest-logit allowed token; sampling renormalizes over allowed tokens. With
// no grammar (or an inactive one) the code path is unchanged.
func sampleToken(logits []float32, p SampleParams, rng *rand.Rand, recent []int32) int32 {
	constrain := p.Grammar != nil && p.Grammar.Active()

	// Repetition penalty (llama.cpp-style): down-weight tokens seen in the recent
	// window so generation can't lock into a decode loop. Applied on free text AND on
	// string CONTENT inside a tool call (where a small model loops generating the same
	// lines — e.g. an apply_patch body repeating a func block) — but NOT on structural
	// JSON tokens, which legitimately repeat and would corrupt the call if penalized.
	inContent := !constrain || (p.Grammar != nil && p.Grammar.InString())
	if p.RepeatPenalty > 1 && inContent && len(recent) > 0 {
		lastN := p.RepeatLastN
		if lastN <= 0 || lastN > len(recent) {
			lastN = len(recent)
		}
		seen := make(map[int32]struct{}, lastN)
		for _, id := range recent[len(recent)-lastN:] {
			if _, ok := seen[id]; ok {
				continue
			}
			seen[id] = struct{}{}
			if int(id) < 0 || int(id) >= len(logits) {
				continue
			}
			if logits[id] > 0 {
				logits[id] /= p.RepeatPenalty
			} else {
				logits[id] *= p.RepeatPenalty
			}
		}
	}

	if p.Temp <= 0 {
		best := -1
		if !constrain {
			best = argmax(logits)
		}
		for i := 0; constrain && i < len(logits); i++ {
			if !p.Grammar.Allows(int32(i)) {
				continue
			}
			if best < 0 || logits[i] > logits[best] {
				best = i
			}
		}
		if best < 0 {
			// The grammar allowed nothing (shouldn't happen for a well-formed
			// grammar) — fall back to the unconstrained argmax rather than deadlock.
			best = argmax(logits)
		}
		return int32(best)
	}

	if p.TopK > 0 {
		return sampleTopK(logits, p, rng, constrain)
	}

	if !constrain {
		return sampleTopP(logits, p, rng)
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

	topP := float64(p.TopP)
	if topP <= 0 || topP > 1 {
		topP = 1.0
	}

	// Grammar constraint: only allowed candidates, in the same prob order. If
	// the grammar allows none (shouldn't happen), fall back to the full set
	// rather than deadlock.
	var allow func(int) bool
	if constrain {
		allow = func(id int) bool { return p.Grammar.Allows(int32(id)) }
	}
	idx := topPCandidates(probs, topP, allow)
	if len(idx) == 0 {
		idx = topPCandidates(probs, topP, nil)
	}
	var cum float64
	cut := len(idx)
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

// sampleTopP is sampleToken's draw with no top-k and no grammar: the
// softmax's sum is taken on several cores and only the most likely tokens,
// enough to hold the top-p share, get a probability of their own. A slice
// of 248k probabilities a token, sorted from a heap of them, took 1.4 ms.
func sampleTopP(logits []float32, p SampleParams, rng *rand.Rand) int32 {
	maxL := float64(logits[argmax(logits)])
	inv := 1.0 / float64(p.Temp)
	sum := expSum(logits, maxL, inv)
	prob := func(id int) float64 { return math.Exp((float64(logits[id])-maxL)*inv) / sum }
	topP := float64(p.TopP)
	// With no top-p the draw is over every token: the candidates are taken
	// as far as the draw reaches, not all 248k sorted.
	need, all := topP, false
	var u float64
	if topP <= 0 || topP >= 1 {
		u, all = rng.Float64(), true
		need = u
	}
	var idx []int
	var probs []float64
	var cum float64
	cut := 0
	for k := 64; cut == 0; k *= 8 {
		k = min(k, len(logits))
		idx = topProbs(logits, k, nil)
		probs, cum = probs[:0], 0
		for rank, id := range idx {
			probs = append(probs, prob(id))
			cum += probs[rank]
			if cum >= need {
				cut = rank + 1
				break
			}
		}
		if cut == 0 && k == len(logits) {
			cut = len(idx)
		}
	}
	r := u
	if !all {
		r = rng.Float64() * cum
	}
	var acc float64
	for rank := 0; rank < cut; rank++ {
		acc += probs[rank]
		if r <= acc {
			return int32(idx[rank])
		}
	}
	return int32(idx[cut-1])
}

// expSum is the sum of exp((x - maxL) * inv) over logits, in parts on as
// many goroutines for a vocabulary's worth; the parts are added in order,
// so the sum is the same on any host.
func expSum(logits []float32, maxL, inv float64) float64 {
	part := func(xs []float32) float64 {
		var s float64
		for _, x := range xs {
			if d := (float64(x) - maxL) * inv; d > -expFloor {
				s += expNeg(d)
			}
		}
		return s
	}
	if len(logits) < scanMin {
		return part(logits)
	}
	var sums [scanParts]float64
	inParts(len(logits), func(i, lo, hi int) { sums[i] = part(logits[lo:hi]) })
	var s float64
	for _, v := range sums {
		s += v
	}
	return s
}

// expNeg is exp(d) for d in (-expFloor, 0], within 1e-8 of math.Exp's value:
// a table of exp at every sixteenth and four terms of the series for what is
// left. math.Exp over a vocabulary was a fifth of the host's time in a
// sampled decode of Qwen3-8B. A term under e^-expFloor is left out: all of
// them together are under 1e-8 of the sum, which is at least 1.
const expFloor = 32

var expSixteenths = func() (t [expFloor*16 + 1]float64) {
	for i := range t {
		t[i] = math.Exp(-float64(i) / 16)
	}
	return t
}()

func expNeg(d float64) float64 {
	k := int(-d * 16)
	r := d + float64(k)/16 // in (-1/16, 0]
	return expSixteenths[k] * (1 + r*(1+r*(0.5+r*(1.0/6+r*(1.0/24)))))
}

// argmax is the first index of the largest logit.
func argmax(logits []float32) int {
	if len(logits) < scanMin {
		return argmaxOf(logits)
	}
	var at [scanParts]int
	inParts(len(logits), func(i, lo, hi int) { at[i] = lo + argmaxOf(logits[lo:hi]) })
	best := at[0]
	for _, id := range at[1:] {
		if logits[id] > logits[best] { // parts in order: the first largest stays
			best = id
		}
	}
	return best
}

func argmaxOf(logits []float32) int {
	best, top := 0, logits[0]
	for i, x := range logits {
		if x > top {
			best, top = i, x
		}
	}
	return best
}

// A scan of the logits (248k for the 27B, 151k for Qwen3) is cut in
// scanParts parts on as many goroutines from scanMin values: one core took
// 0.2 to 0.3 ms a token on a 2.1 GHz Xeon, 3% of an 8B's decode step.
const (
	scanParts = 8
	scanMin   = 1 << 15
)

// inParts runs part on scanParts consecutive ranges of n values and waits.
func inParts(n int, part func(i, lo, hi int)) {
	var wg sync.WaitGroup
	for i := 0; i < scanParts; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			part(i, i*n/scanParts, (i+1)*n/scanParts)
		}(i)
	}
	wg.Wait()
}

// sampleTopK is sampleToken's draw with a top-k: the K most likely tokens
// are the K largest logits, and top-p is a share of what those K hold, so
// only K exponentials are taken, where the softmax over a 248k vocabulary
// took 1.5 ms a token.
func sampleTopK(logits []float32, p SampleParams, rng *rand.Rand, constrain bool) int32 {
	var idx []int
	if constrain {
		idx = topProbs(logits, p.TopK, func(id int) bool { return p.Grammar.Allows(int32(id)) })
	}
	if len(idx) == 0 {
		// No grammar, or one that allowed nothing (shouldn't happen): the
		// full set rather than a deadlock.
		idx = topProbs(logits, p.TopK, nil)
	}
	topP := float64(p.TopP)
	if topP <= 0 || topP > 1 {
		topP = 1.0
	}
	inv := 1.0 / float64(p.Temp)
	w := make([]float64, len(idx))
	var total float64
	for i, id := range idx {
		w[i] = math.Exp((float64(logits[id]) - float64(logits[idx[0]])) * inv)
		total += w[i]
	}
	var cum float64
	cut := len(idx)
	for rank := range idx {
		cum += w[rank]
		if cum >= topP*total {
			cut = rank + 1
			break
		}
	}
	r := rng.Float64() * cum
	var acc float64
	for rank := 0; rank < cut; rank++ {
		acc += w[rank]
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
	caches := make([]LayerCache, nLayers)
	for i := range caches {
		caches[i] = &KVCache{}
	}

	logits := forward(b, b.FromInt32(prompt, len(prompt)), len(prompt), 0, caches)
	tok := sampleToken(b.Floats(logits), p, rng, nil) // forward returns last-position logits
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
		tok = sampleToken(b.Floats(logits), p, rng, out)
		offset++
		if sw != nil {
			pinCaches(sw, caches)
			unpinGone(sw, old, cacheTensors(caches))
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

// moeExpertBudget bounds the pinned GPU expert cache for MoE models. The
// default (2.5 GB ≈ 16 hot experts per layer on a 48-layer 30B-A3B, mirroring
// TurboFieldfare's validated slot count) balances two consumers of the same
// RAM: a bigger cache raises the hit rate, but starves the OS page cache that
// makes the remaining misses cheap. JANIS_MOE_CACHE_MB overrides for tuning.
func moeExpertBudget() int64 {
	if mb, _ := strconv.Atoi(os.Getenv("JANIS_MOE_CACHE_MB")); mb > 0 {
		return int64(mb) << 20
	}
	return 2500 << 20
}

// LoadModel loads the embedding, all decoder blocks, the final norm, and (when
// untied) the lm_head. For MoE models it builds an ExpertStore so expert weights
// are streamed on demand instead of loaded resident.
func LoadModel(b engine.Backend, st *safetensors.File, cfg Config) (*Model, error) {
	embGS, embBits := cfg.QuantFor("model.embed_tokens")
	emb, err := LoadQuantEmbedding(b, st, "model.embed_tokens", embGS, embBits)
	if err != nil {
		return nil, err
	}
	var store *ExpertStore
	if cfg.NumExperts > 0 {
		store = NewExpertStore(b, st, cfg.GroupSize, cfg.Bits, moeExpertBudget())
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
		gs, bits := cfg.QuantFor("lm_head")
		if m.LMHead, err = LoadQuantLinear(b, st, "lm_head", gs, bits, false); err != nil {
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
func (m *Model) forwardCachedT(b engine.Backend, idsT engine.Tensor, seqLen, offset int, caches []LayerCache) engine.Tensor {
	h := b.Reshape(m.Embed.Forward(b, idsT), 1, seqLen, m.Cfg.Hidden)
	for i, blk := range m.Blocks {
		h = blk.ForwardCached(b, h, caches[i].(*KVCache), offset)
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

// NewSession returns a prefix-reusing session for multi-turn serving. MoE models
// prefill one token at a time (PrefillChunk=1) to use the stream-free decode path
// per token instead of a slow batched per-layer-eval pass.
func (m *Model) NewSession() *Session {
	s := &Session{NLayers: m.Cfg.Layers, Vocab: m.Cfg.Vocab, Forward: m.forwardCachedT}
	if m.Cfg.NumExperts > 0 {
		s.PrefillChunk = moePrefillChunk()
	}
	return s
}

// moePrefillChunk is the MoE prefill row-group size. 1 stays the default:
// chunk=128 (TurboFieldfare's production number) prefills ~30% faster
// (2.5 vs 1.9 tok/s measured on Qwen3-30B-A3B cold, 2026-08-03) by fetching
// each layer's expert union once per chunk — but it leaves the decode slot
// cache unwarmed (the hybrid path deliberately never writes it), and decode
// after a from-scratch chunked prefill measured 0.10 tok/s vs ~3. The trade
// only pays for generate-nothing calls (KV warmup, MaxTokens=1), so warmup
// boots may set JANIS_MOE_PREFILL_CHUNK=128 explicitly; interactive
// sessions keep 1.
func moePrefillChunk() int {
	if n, _ := strconv.Atoi(os.Getenv("JANIS_MOE_PREFILL_CHUNK")); n > 0 {
		return n
	}
	return 1
}

// ForwardFunc computes logits [1, seqLen, vocab] for a device index tensor
// through per-layer caches at a RoPE offset. Each architecture asserts its
// own concrete cache types out of the slice.
type ForwardFunc func(b engine.Backend, idsT engine.Tensor, seqLen, offset int, caches []LayerCache) engine.Tensor

// GreedyGenerate is the shared device-resident decode loop, parameterized by an
// architecture's forward function, so Qwen and Gemma reuse the same KV-cache /
// Pin-Sweep / async-eval machinery.
func GreedyGenerate(b engine.Backend, prompt []int32, nGen, nLayers int, forward ForwardFunc) []int32 {
	sw, _ := b.(engine.Sweeper)
	ae, _ := b.(engine.AsyncEvaler)
	if sw != nil {
		sw.PinAll() // protect the weights
	}

	caches := make([]LayerCache, nLayers)
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
			sw.Pin(tokT) // keep every device token for the final readback
			unpinGone(sw, old, cacheTensors(caches))
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

func cacheTensors(caches []LayerCache) []engine.Tensor {
	ts := make([]engine.Tensor, 0, 2*len(caches))
	for _, c := range caches {
		ts = append(ts, c.Tensors()...)
	}
	return ts
}

func pinCaches(sw engine.Sweeper, caches []LayerCache) {
	for _, c := range caches {
		sw.Pin(c.Tensors()...)
	}
}

// unpinGone unpins the tensors of old that kept no longer holds. Pins are
// flags, not counts: a tensor a cache keeps across steps (a recurrent
// layer's mark, or the state the mark restored) is in both lists, and an
// unpin of all of old would hand it to the next sweep.
func unpinGone(sw engine.Sweeper, old, kept []engine.Tensor) {
	live := make(map[engine.Tensor]bool, len(kept))
	for _, t := range kept {
		live[t] = true
	}
	for _, t := range old {
		if !live[t] {
			sw.Unpin(t)
		}
	}
}

// topPCandidates returns the candidates by descending probability (ties by
// token id), at least up to where their mass reaches topP: what the top-p cut
// reads. It ranks the best k with a bounded heap, k growing until they reach
// topP, instead of sorting the whole vocabulary: that sort was 18 ms a token
// on a 248k vocabulary, a third of a 9B's decode step on an M4. allow, when
// set, keeps only the tokens it allows.
func topPCandidates(probs []float64, topP float64, allow func(int) bool) []int {
	for k := 64; ; k *= 8 {
		if k >= len(probs) {
			k = len(probs)
		}
		c := topProbs(probs, k, allow)
		if len(c) < k || k == len(probs) {
			return c // every allowed token is in c
		}
		var cum float64
		for _, id := range c {
			cum += probs[id]
			if cum >= topP {
				return c
			}
		}
	}
}

// before is the candidate order: higher probability first, then lower id.
func before[T float32 | float64](probs []T, a, b int) bool {
	if probs[a] != probs[b] {
		return probs[a] > probs[b]
	}
	return a < b
}

// topProbs returns up to k allowed token ids in candidate order.
func topProbs[T float32 | float64](probs []T, k int, allow func(int) bool) []int {
	if allow != nil || len(probs) < scanMin || k > len(probs)/(4*scanParts) {
		return topProbsOf(probs, k, allow)
	}
	// Each part's k best, then the k best of those: the candidates keep
	// their order by id, so ties fall as they do in one scan.
	var kept [scanParts][]int
	inParts(len(probs), func(i, lo, hi int) {
		kept[i] = topProbsOf(probs[lo:hi], k, nil)
		for j := range kept[i] {
			kept[i][j] += lo
		}
		sort.Ints(kept[i])
	})
	var ids []int
	for _, part := range kept {
		ids = append(ids, part...)
	}
	vals := make([]T, len(ids))
	for i, id := range ids {
		vals[i] = probs[id]
	}
	out := topProbsOf(vals, k, nil)
	for i, at := range out {
		out[i] = ids[at]
	}
	return out
}

// topProbsOf is topProbs in one scan.
func topProbsOf[T float32 | float64](probs []T, k int, allow func(int) bool) []int {
	h := make([]int, 0, k) // min-heap by candidate order: h[0] is the worst kept
	down := func(i int) {
		for {
			l, r, m := 2*i+1, 2*i+2, i
			if l < len(h) && before(probs, h[m], h[l]) {
				m = l
			}
			if r < len(h) && before(probs, h[m], h[r]) {
				m = r
			}
			if m == i {
				return
			}
			h[i], h[m] = h[m], h[i]
			i = m
		}
	}
	var worst T // the least likely kept, once k are
	for id, v := range probs {
		if len(h) == k && v <= worst {
			// ids come in order: one no more likely than the worst kept
			// is after it.
			continue
		}
		if allow != nil && !allow(id) {
			continue
		}
		if len(h) < k {
			h = append(h, id)
			for i := len(h) - 1; i > 0; {
				parent := (i - 1) / 2
				if !before(probs, h[parent], h[i]) {
					break
				}
				h[i], h[parent] = h[parent], h[i]
				i = parent
			}
		} else {
			h[0] = id
			down(0)
		}
		worst = probs[h[0]]
	}
	sort.Slice(h, func(a, b int) bool { return before(probs, h[a], h[b]) })
	return h
}
