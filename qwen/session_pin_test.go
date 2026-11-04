package qwen

import (
	"testing"

	"github.com/guregodevo/janis/cpu"
	"github.com/guregodevo/janis/engine"
)

// pinSpy is a CPU backend that records pins like the MLX one does: flags, not
// counts. Sweep frees nothing; the test reads what is still pinned.
type pinSpy struct {
	engine.Backend
	pinned map[engine.Tensor]bool
}

func (p *pinSpy) PinAll() {}
func (p *pinSpy) Pin(ts ...engine.Tensor) {
	for _, t := range ts {
		if t != nil {
			p.pinned[t] = true
		}
	}
}
func (p *pinSpy) Unpin(ts ...engine.Tensor) {
	for _, t := range ts {
		delete(p.pinned, t)
	}
}
func (p *pinSpy) Sweep() {}

// stateCache is a recurrent-style cache: one tensor, replaced every forward.
type stateCache struct {
	t engine.Tensor
	n int
}

func (c *stateCache) Len() int                         { return c.n }
func (c *stateCache) CanTruncate(n int) bool           { return n == 0 || n >= c.n }
func (c *stateCache) Truncate(_ engine.Backend, n int) { c.t, c.n = nil, 0 }
func (c *stateCache) Mark()                            {}
func (c *stateCache) Forget(int)                       {}
func (c *stateCache) Tensors() []engine.Tensor {
	if c.t == nil {
		return nil
	}
	return []engine.Tensor{c.t}
}

// A CHUNKED PREFILL LEAVES ONLY THE CACHES PINNED.
//
// Each chunk's logits were pinned to survive that chunk's sweep, and only the
// last chunk's were unpinned: on MLX every request kept a vocab-wide row per
// chunk for good (Qwen3.5-0.8B grew 21 MB a request, 5,600 tokens in 22 chunks).
func TestAChunkedPrefillLeavesOnlyTheCachesPinned(t *testing.T) {
	const vocab = 16
	spy := &pinSpy{Backend: cpu.New(), pinned: map[engine.Tensor]bool{}}
	s := &Session{
		NLayers:      2,
		Vocab:        vocab,
		PrefillChunk: 2,
		NewCache:     func(int) LayerCache { return &stateCache{} },
		Forward: func(b engine.Backend, _ engine.Tensor, seqLen, _ int, caches []LayerCache) engine.Tensor {
			for _, c := range caches {
				sc := c.(*stateCache)
				sc.t = b.FromFloats([]float32{float32(sc.n + seqLen)}, 1)
				sc.n += seqLen
			}
			return b.FromFloats(make([]float32, vocab), 1, 1, vocab)
		},
	}
	for round, prompt := range [][]int32{
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},                 // five chunks
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}, // extends it: two more
		{9, 9, 9, 9, 9, 9, 9},                           // a new prompt: four chunks from scratch
	} {
		if s.Prefill(spy, prompt, nil) == nil {
			t.Fatalf("round %d: no logits", round)
		}
		live := map[engine.Tensor]bool{}
		for _, ct := range cacheTensors(s.Caches) {
			live[ct] = true
		}
		for p := range spy.pinned {
			if !live[p] {
				t.Fatalf("round %d: %d tensors pinned, %d of them the caches': a chunk's logits stayed pinned", round, len(spy.pinned), len(live))
			}
		}
	}
}

// DECODING LEAVES ONLY THE CACHES PINNED TOO.
//
// Each step pins its logits across the sweep that now runs before the read,
// and must unpin them after: a pin left behind is the same leak, one row a
// generated token.
func TestDecodingLeavesOnlyTheCachesPinned(t *testing.T) {
	const vocab = 16
	spy := &pinSpy{Backend: cpu.New(), pinned: map[engine.Tensor]bool{}}
	s := &Session{
		NLayers:      2,
		Vocab:        vocab,
		PrefillChunk: 2,
		NewCache:     func(int) LayerCache { return &stateCache{} },
		Forward: func(b engine.Backend, _ engine.Tensor, seqLen, _ int, caches []LayerCache) engine.Tensor {
			for _, c := range caches {
				sc := c.(*stateCache)
				sc.t = b.FromFloats([]float32{float32(sc.n + seqLen)}, 1)
				sc.n += seqLen
			}
			logits := make([]float32, vocab)
			logits[3] = 1
			return b.FromFloats(logits, 1, 1, vocab)
		},
	}
	gen := s.Generate(spy, []int32{1, 2, 3, 4, 5}, 6, SampleParams{})
	if len(gen) != 6 {
		t.Fatalf("generated %d tokens, want 6", len(gen))
	}
	live := map[engine.Tensor]bool{}
	for _, ct := range cacheTensors(s.Caches) {
		live[ct] = true
	}
	for p := range spy.pinned {
		if !live[p] {
			t.Fatalf("%d tensors pinned after decoding, %d of them the caches'", len(spy.pinned), len(live))
		}
	}
}
