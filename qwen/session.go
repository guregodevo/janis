package qwen

import (
	"fmt"
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

	IDs    []int32 // tokens currently materialized in the caches
	Caches []*KVCache
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

func cachedLen(caches []*KVCache) int {
	if len(caches) == 0 || caches[0] == nil {
		return 0
	}
	return caches[0].Len()
}

// Generate decodes a reply for the full prompt newIDs, reusing the cached prefix
// it shares with the previous turn.
func (s *Session) Generate(b engine.Backend, newIDs []int32, nGen int, p SampleParams) []int32 {
	if len(newIDs) == 0 {
		return nil
	}
	rng := rand.New(rand.NewSource(int64(p.Seed)))
	sw, _ := b.(engine.Sweeper)
	old := cacheTensors(s.Caches) // previous turn's caches, to be freed

	// Reuse the shared prefix; keep at least one token to prefill (we need its
	// logits to predict the first reply token).
	P := lcp(s.IDs, newIDs)
	if P > len(newIDs)-1 {
		P = len(newIDs) - 1
	}
	if s.Caches == nil {
		s.Caches = make([]*KVCache, s.NLayers)
		for i := range s.Caches {
			s.Caches[i] = &KVCache{}
		}
	} else if P < cachedLen(s.Caches) {
		for _, c := range s.Caches {
			c.Truncate(b, P)
		}
	}

	suffix := newIDs[P:]
	offset := P

	if debugPrefill {
		debugLog("prefill: total=%d reused=%d new=%d (chunk=%d)", len(newIDs), P, len(suffix), prefillChunk)
	}

	// Chunked prefill. Forwarding a long prompt in one pass holds EVERY layer's
	// activations (~seq × intermediate × layers) live until the single eval —
	// multiple GB at a few-thousand-token RAG prompt, which jetsam-kills the
	// process on constrained unified memory (measured: a ~5k-token prefill on a
	// 3B model OOMs a 16GB Mac). Processing the suffix in fixed-size chunks caps
	// peak activation at one chunk's worth; the KV cache (pinned) carries state
	// across chunks, so the final logits are identical to a single forward — the
	// same mechanism as incremental decode. We MUST eval+sweep each chunk, else
	// MLX defers every chunk to one eval and the spike returns.
	var logits engine.Tensor
	for i := 0; i < len(suffix); i += prefillChunk {
		end := i + prefillChunk
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
	}
	tok := sampleToken(b.Floats(logits), p, rng) // last-position logits
	if sw != nil {
		sw.Unpin(logits)
		pinCaches(sw, s.Caches)
		sw.Unpin(old...)
		sw.Sweep()
	}

	var gen []int32
	for {
		if p.Stop[tok] {
			break
		}
		gen = append(gen, tok)
		if p.OnToken != nil {
			p.OnToken(tok)
		}
		if len(gen) >= nGen {
			break
		}
		prev := cacheTensors(s.Caches)
		logits = s.Forward(b, b.FromInt32([]int32{tok}, 1), 1, offset, s.Caches)
		tok = sampleToken(b.Floats(logits), p, rng)
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
