package qwen35

import (
	"testing"

	"github.com/guregodevo/janis/cpu"
	"github.com/guregodevo/janis/engine"
	"github.com/guregodevo/janis/qwen"
)

// A hybrid session's recurrent layer cannot be cut back to a prefix, so the
// second conversation sharing a system prompt with the first prefills from
// scratch; it marks where the two parted, and the third starts there.
func TestHybridSessionReturnsToTheMark(t *testing.T) {
	b := cpu.New()
	var forwarded [][2]int // offset, tokens of each Forward
	forward := func(b engine.Backend, _ engine.Tensor, n, offset int, caches []qwen.LayerCache) engine.Tensor {
		forwarded = append(forwarded, [2]int{offset, n})
		for _, c := range caches {
			switch c := c.(type) {
			case *LinearCache:
				at := []float32{float32(offset + n)}
				c.Conv, c.State, c.steps = b.FromFloats(at, 1, 1, 1), b.FromFloats(at, 1, 1, 1), c.steps+n
			case *qwen.KVCache:
				kv := make([]float32, offset+n)
				c.K, c.V = b.FromFloats(kv, 1, 1, offset+n, 1), b.FromFloats(kv, 1, 1, offset+n, 1)
			}
		}
		return b.FromFloats([]float32{0, 1}, 1, 2)
	}
	s := &qwen.Session{NLayers: 2, Forward: forward, PrefillChunk: 4, NewCache: func(layer int) qwen.LayerCache {
		if layer == 0 {
			return &LinearCache{}
		}
		return &qwen.KVCache{}
	}}
	total := func() int {
		n := 0
		for _, f := range forwarded {
			n += f[1]
		}
		return n
	}
	system := []int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	conversation := func(turn ...int32) []int32 { return append(append([]int32(nil), system...), turn...) }

	s.Prefill(b, conversation(101, 102, 103), nil)
	if total() != 13 || s.Mark != 0 {
		t.Fatalf("the first conversation prefills everything and marks nothing: %d tokens, mark %d", total(), s.Mark)
	}

	forwarded = nil
	s.Prefill(b, conversation(201, 202), nil)
	if total() != 12 || s.Mark != 10 {
		t.Fatalf("the second shares the system prompt but the recurrent state cannot return to it: everything again (%d tokens), marked where they parted (%d)", total(), s.Mark)
	}
	ends := map[int]bool{}
	for _, f := range forwarded {
		ends[f[0]+f[1]] = true
	}
	if !ends[10] {
		t.Fatalf("a chunk ends at the mark so the state kept is the one at that length: %v", forwarded)
	}
	linear := s.Caches[0].(*LinearCache)
	if linear.mark == nil || linear.mark.steps != 10 || b.Floats(linear.mark.state)[0] != 10 {
		t.Fatalf("the recurrent layer keeps its state at the mark: %+v", linear.mark)
	}

	forwarded = nil
	s.Prefill(b, conversation(301), nil)
	if total() != 1 || s.Mark != 10 || linear.Len() != 11 || s.Caches[1].Len() != 11 {
		t.Fatalf("the third starts at the mark: %d tokens prefilled, mark %d, lengths %d %d", total(), s.Mark, linear.Len(), s.Caches[1].Len())
	}
	if linear.mark == nil || linear.mark.steps != 10 {
		t.Fatal("the mark stays for the next conversation")
	}

	forwarded = nil
	s.Prefill(b, conversation(301, 302), nil)
	if total() != 1 {
		t.Fatalf("a turn that extends the prompt prefills the new token only: %d", total())
	}

	forwarded = nil
	s.Prefill(b, []int32{1, 2, 3, 99, 98}, nil)
	if total() != 5 || s.Mark != 3 || linear.mark == nil || linear.mark.steps != 3 {
		t.Fatalf("a prompt that parts before the mark drops it, prefills everything and marks the new parting: %d tokens, mark %d", total(), s.Mark)
	}

	// A mark at 3, then prompts that share 10 with each other: the first
	// returns to 3 (all it shares with the prompt before), the second returns
	// to 3 and moves the mark to 10, the third starts at 10. The shape of a
	// server's warm-up request ("ok", sharing the template's first token)
	// followed by real conversations.
	forwarded = nil
	s.Prefill(b, conversation(401), nil)
	if total() != 8 || s.Mark != 3 {
		t.Fatalf("a prompt sharing the mark exactly returns to it: %d tokens, mark %d", total(), s.Mark)
	}
	forwarded = nil
	s.Prefill(b, conversation(501), nil)
	if total() != 8 || s.Mark != 10 || linear.mark.steps != 10 {
		t.Fatalf("a prompt sharing more than the mark returns to it and moves the mark up: %d tokens, mark %d", total(), s.Mark)
	}
	forwarded = nil
	s.Prefill(b, conversation(601), nil)
	if total() != 1 {
		t.Fatalf("the next one starts at the moved mark: %d tokens", total())
	}
}

// The mark's tensors are in the cache's tensor list, so the sweeper keeps
// them, and a reset drops them.
func TestLinearCacheMark(t *testing.T) {
	b := cpu.New()
	c := &LinearCache{}
	c.Mark()
	if c.mark != nil {
		t.Fatal("an empty cache marks nothing")
	}
	c.Conv, c.State, c.steps = b.FromFloats([]float32{1}, 1), b.FromFloats([]float32{1}, 1), 4
	c.Mark()
	c.Conv, c.State, c.steps = b.FromFloats([]float32{2}, 1), b.FromFloats([]float32{2}, 1), 7
	if len(c.Tensors()) != 4 {
		t.Fatalf("live state and mark are both kept: %d tensors", len(c.Tensors()))
	}
	for n, want := range map[int]bool{0: true, 3: false, 4: true, 5: false, 7: true, 9: true} {
		if c.CanTruncate(n) != want {
			t.Errorf("CanTruncate(%d) = %v", n, !want)
		}
	}
	c.Truncate(b, 4)
	if c.Len() != 4 || b.Floats(c.State)[0] != 1 || c.mark == nil {
		t.Fatalf("Truncate to the mark restores its state and keeps the mark: len %d", c.Len())
	}
	c.Truncate(b, 0)
	if c.Len() != 0 || c.mark != nil || len(c.Tensors()) != 0 {
		t.Fatal("a reset drops the state and the mark")
	}
}
