package qwen

import (
	"math/rand"

	"memdoor/llm/engine"
)

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
	logits := s.Forward(b, b.FromInt32(suffix, len(suffix)), len(suffix), offset, s.Caches)
	offset += len(suffix)
	tok := sampleToken(b.Floats(logits), p, rng) // last-position logits
	if sw != nil {
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
