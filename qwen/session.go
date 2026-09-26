package qwen

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"

	"memdoor/llm/engine"
)

// debugPrefill logs the prefix-reuse split per generation (MLX_DEBUG_PREFILL=1)
// so we can see how much of a prompt is re-prefilled vs reused across turns.
var debugPrefill = os.Getenv("MLX_DEBUG_PREFILL") == "1"

func debugLog(format string, a ...any) {
	fmt.Fprintf(os.Stderr, "[mlx] "+format+"\n", a...)
}

// prefillChunk is the chunked-prefill window: peak activation during prefill is
// bounded by this many tokens rather than the whole prompt. 256 keeps the spike
// small while staying efficient; override with MLX_PREFILL_CHUNK.
var prefillChunk = func() int {
	if v := os.Getenv("MLX_PREFILL_CHUNK"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return 256
}()

// Session keeps a conversation's KV cache alive across requests so a follow-up
// only prefills the new suffix (longest-common-prefix reuse), instead of
// re-prefilling the whole growing conversation each turn.
type Session struct {
	NLayers int
	Vocab   int
	Forward ForwardFunc

	// PrefillChunk overrides the global prefill window for this session (0 = use
	// the global default). MoE models set it to 1 so each prompt token is
	// prefilled through the stream-free decode path (nTok==1, no per-layer eval),
	// instead of one batched per-layer-eval pass that re-materializes ~all the
	// prompt's experts at once — ~3x faster prefill on Qwen3-30B-A3B.
	PrefillChunk int

	// NewCache builds layer i's cache. Nil means plain KV attention on every
	// layer; hybrid architectures (qwen3_5) supply their own per-layer mix.
	NewCache func(layer int) LayerCache

	IDs    []int32 // tokens currently materialized in the caches
	Caches []LayerCache
}

func (s *Session) newCache(layer int) LayerCache {
	if s.NewCache != nil {
		return s.NewCache(layer)
	}
	return &KVCache{}
}

func lcp(a, b []int32) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	i := 0
	for i < n && a[i] == b[i] {
		i++
	}
	return i
}

func cachedLen(caches []LayerCache) int {
	if len(caches) == 0 || caches[0] == nil {
		return 0
	}
	return caches[0].Len()
}

// prefill runs the chunked prefill of newIDs over the reused prefix and returns
// the last-position logits on the host; false means cancel fired first. The
// caller owns s.IDs: Prefill records the whole prompt, Generate appends the
// reply to it.
func (s *Session) prefill(b engine.Backend, newIDs []int32, cancel func() bool) ([]float32, bool) {
	sw, _ := b.(engine.Sweeper)
	old := cacheTensors(s.Caches) // previous turn's caches, to be freed

	// Reuse the shared prefix; keep at least one token to prefill (we need its
	// logits to predict the first reply token).
	P := lcp(s.IDs, newIDs)
	if P > len(newIDs)-1 {
		P = len(newIDs) - 1
	}
	if s.Caches == nil {
		s.Caches = make([]LayerCache, s.NLayers)
		for i := range s.Caches {
			s.Caches[i] = s.newCache(i)
		}
	} else if P < cachedLen(s.Caches) {
		// A recurrent (linear-attention) cache stores only its latest state, so
		// it cannot be cut back to a mid-history prefix: when any layer can't
		// truncate to P, the whole session resets and re-prefills from scratch.
		// Prefix EXTENSION (the common multi-turn case) never lands here.
		for _, c := range s.Caches {
			if !c.CanTruncate(P) {
				P = 0
				break
			}
		}
		for _, c := range s.Caches {
			c.Truncate(b, P)
		}
	}

	suffix := newIDs[P:]
	offset := P

	// Chunked prefill. Forwarding a long prompt in one pass holds EVERY layer's
	// activations (~seq × intermediate × layers) live until the single eval —
	// multiple GB at a few-thousand-token RAG prompt, which jetsam-kills the
	// process on constrained unified memory (measured: a ~5k-token prefill on a
	// 3B model OOMs a 16GB Mac). Processing the suffix in fixed-size chunks caps
	// peak activation at one chunk's worth; the KV cache (pinned) carries state
	// across chunks, so the final logits are identical to a single forward — the
	// same mechanism as incremental decode. We MUST eval+sweep each chunk, else
	// MLX defers every chunk to one eval and the spike returns.
	chunkSize := prefillChunk
	if s.PrefillChunk > 0 {
		chunkSize = s.PrefillChunk
	}

	if debugPrefill {
		debugLog("prefill: total=%d reused=%d new=%d (chunk=%d)", len(newIDs), P, len(suffix), chunkSize)
	}
	var logits engine.Tensor
	for i := 0; i < len(suffix); i += chunkSize {
		end := i + chunkSize
		if end > len(suffix) {
			end = len(suffix)
		}
		chunk := suffix[i:end]
		prev := cacheTensors(s.Caches)
		logits = s.Forward(b, b.FromInt32(chunk, len(chunk)), len(chunk), offset, s.Caches)
		offset += len(chunk)
		if sw != nil {
			// Force this chunk's compute so its intermediates become free, then
			// sweep them. Keep the caches (state) and the running logits (needed
			// for sampling after the last chunk) pinned across the sweep.
			sw.Pin(logits)
			b.Eval(append(cacheTensors(s.Caches), logits)...)
			pinCaches(sw, s.Caches)
			sw.Unpin(prev...)
			sw.Sweep()
		}
		if cancel != nil && cancel() {
			// Cancelled mid-prefill: the cache now holds `offset` tokens, so record
			// that exact prefix as the session state — keeps the next turn's KV
			// prefix-reuse consistent instead of claiming the full prompt is cached.
			s.IDs = append([]int32(nil), newIDs[:offset]...)
			return nil, false
		}
	}
	floats := b.Floats(logits) // last-position logits
	if sw != nil {
		sw.Unpin(logits)
		pinCaches(sw, s.Caches)
		sw.Unpin(old...)
		sw.Sweep()
	}
	return floats, true
}

// Prefill materializes newIDs into the session's caches (reusing the cached
// prefix it shares with the previous call) and returns the last-position
// logits on the host, or nil when cancel fired mid-prefill. No token is
// sampled and nothing is decoded: this is the read-only half of Generate, for
// callers that consume the next-token distribution directly (Engine.Decide).
// The full prompt is recorded as the session state, so a second call sharing
// its prefix (same state, different question) prefills only the suffix.
func (s *Session) Prefill(b engine.Backend, newIDs []int32, cancel func() bool) []float32 {
	if len(newIDs) == 0 {
		return nil
	}
	floats, ok := s.prefill(b, newIDs, cancel)
	if !ok {
		return nil
	}
	s.IDs = append([]int32(nil), newIDs...)
	return floats
}

// Generate decodes a reply for the full prompt newIDs, reusing the cached prefix
// it shares with the previous turn.
func (s *Session) Generate(b engine.Backend, newIDs []int32, nGen int, p SampleParams) []int32 {
	if len(newIDs) == 0 {
		return nil
	}
	rng := rand.New(rand.NewSource(int64(p.Seed)))
	sw, _ := b.(engine.Sweeper)
	floats, ok := s.prefill(b, newIDs, p.Cancel)
	if !ok {
		return nil
	}
	offset := len(newIDs)
	tok := sampleToken(floats, p, rng, nil)
	if p.Stop[tok] && nGen > 0 {
		// A turn that ends before it begins is never the right outcome: on the
		// full 8k-token agent prompt the streamed 30B sampled <|im_end|> as its
		// FIRST token, producing zero-token "empty final text" turns after 25
		// minutes of prefill. Mask every stop token for the first position and
		// resample — the model must say something; downstream absorbers can
		// handle whatever that is.
		for id := range p.Stop {
			if int(id) >= 0 && int(id) < len(floats) {
				floats[id] = float32(math.Inf(-1))
			}
		}
		tok = sampleToken(floats, p, rng, nil)
	}

	var gen []int32
	var logits engine.Tensor
	for {
		if p.Stop[tok] {
			break
		}
		gen = append(gen, tok)
		if p.Grammar != nil {
			p.Grammar.Advance(tok) // commit the token so the next sample sees updated state
		}
		if p.OnToken != nil {
			p.OnToken(tok)
		}
		if len(gen) >= nGen {
			break
		}
		if p.Cancel != nil && p.Cancel() {
			break
		}
		prev := cacheTensors(s.Caches)
		logits = s.Forward(b, b.FromInt32([]int32{tok}, 1), 1, offset, s.Caches)
		tok = sampleToken(b.Floats(logits), p, rng, gen)
		offset++
		if sw != nil {
			pinCaches(sw, s.Caches)
			sw.Unpin(prev...)
			sw.Sweep()
		}
	}

	s.IDs = append(append([]int32(nil), newIDs...), gen...)
	return gen
}
