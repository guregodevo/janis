package qwen35

import (
	"fmt"
	"testing"

	"github.com/guregodevo/janis/cpu"
	"github.com/guregodevo/janis/engine"
	"github.com/guregodevo/janis/qwen"
)

// hybridSession is a two-layer session, one recurrent layer and one KV
// layer, on a fake forward that records what it was asked to prefill.
func hybridSession(b engine.Backend) (*qwen.Session, func() int, func()) {
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
	return s, total, func() { forwarded = nil }
}

// A hybrid session's recurrent layer cannot be cut back to a prefix, so the
// second conversation sharing a system prompt with the first prefills from
// scratch; it marks where the two parted, and the third starts there.
func TestHybridSessionReturnsToTheMark(t *testing.T) {
	b := cpu.New()
	s, total, reset := hybridSession(b)
	system := []int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	conversation := func(turn ...int32) []int32 { return append(append([]int32(nil), system...), turn...) }
	step := func(name string, prompt []int32, wantForwarded int, wantMarks ...int) {
		t.Helper()
		reset()
		s.Prefill(b, prompt, nil)
		if total() != wantForwarded || fmt.Sprint(s.Marks()) != fmt.Sprint(wantMarks) {
			t.Fatalf("%s: %d tokens prefilled, marks %v; want %d and %v", name, total(), s.Marks(), wantForwarded, wantMarks)
		}
	}

	step("the first conversation prefills everything and marks nothing", conversation(101, 102, 103), 13)
	step("the second shares the system prompt but the recurrent state cannot return to it: everything again, a mark where they parted", conversation(201, 202), 12, 10)
	linear := s.Caches[0].(*LinearCache)
	if m, ok := linear.marks[10]; !ok || b.Floats(m.state)[0] != 10 {
		t.Fatalf("the recurrent layer keeps its state at the mark: %v", linear.marks)
	}
	step("the third starts at the mark", conversation(301), 1, 10)
	if linear.Len() != 11 || s.Caches[1].Len() != 11 {
		t.Fatalf("both layers hold the prompt: %d %d", linear.Len(), s.Caches[1].Len())
	}
	step("a turn that extends the prompt prefills the new token only", conversation(301, 302), 1, 10)
	step("a follow-up whose rendered reply differs from the generated one parts after the turn: back to the system prompt, a mark at the turn's end too", conversation(301, 303, 304), 3, 10, 11)
	step("a new conversation after that still returns to the system prompt", conversation(401), 1, 10)
	step("a prompt that parts before every mark drops them, prefills everything and marks the new parting", []int32{1, 2, 3, 99, 98}, 5, 3)
	step("a prompt sharing the mark exactly returns to it", conversation(501), 8, 3)
	step("a prompt sharing more than the mark returns to it and adds a mark where the two parted", conversation(601), 8, 3, 10)
	step("the next one starts at the new mark", conversation(701), 1, 3, 10)
}

// The marks stay few: the least recently returned to goes first.
func TestHybridSessionForgetsTheLeastUsedMark(t *testing.T) {
	b := cpu.New()
	s, _, reset := hybridSession(b)
	prefix := func(n int, tail ...int32) []int32 {
		ids := make([]int32, n)
		for i := range ids {
			ids[i] = int32(i + 1)
		}
		return append(ids, tail...)
	}
	// Marks nest: each is a prefix of the next. Two prompts parting at a
	// length make its mark; in rising order they make 2, 4, 6, 8.
	s.Prefill(b, prefix(10), nil)
	for _, at := range []int{2, 4, 6, 8} {
		reset()
		s.Prefill(b, prefix(at, 90), nil)
		reset()
		s.Prefill(b, prefix(at, 91), nil)
	}
	if fmt.Sprint(s.Marks()) != "[2 4 6 8]" {
		t.Fatalf("four marks: %v", s.Marks())
	}
	reset()
	s.Prefill(b, prefix(2, 80), nil) // returns to 2: drops the marks past it
	if fmt.Sprint(s.Marks()) != "[2]" {
		t.Fatalf("a return to 2 drops the marks past it: %v", s.Marks())
	}
	for _, at := range []int{4, 6, 8} {
		reset()
		s.Prefill(b, prefix(at, 70, 71), nil)
		reset()
		s.Prefill(b, prefix(at, 72, 73), nil)
	}
	// Marks 2 (used first), 4, 6, 8 (each used by its second prompt): a fifth
	// forgets 2, the least recently used.
	reset()
	s.Prefill(b, prefix(9, 60, 61), nil)
	reset()
	s.Prefill(b, prefix(9, 62), nil)
	if fmt.Sprint(s.Marks()) != "[4 6 8 9]" {
		t.Fatalf("a fifth mark forgets the least used: %v", s.Marks())
	}
	linear := s.Caches[0].(*LinearCache)
	if len(linear.marks) != 4 || len(linear.Tensors()) != 2+2*4 {
		t.Fatalf("the recurrent layer forgot it too: %d marks, %d tensors", len(linear.marks), len(linear.Tensors()))
	}
}

// The marks' tensors are in the cache's tensor list, so the sweeper keeps
// them; a truncation drops the marks past it, a reset drops them all.
func TestLinearCacheMarks(t *testing.T) {
	b := cpu.New()
	c := &LinearCache{}
	c.Mark()
	if len(c.marks) != 0 {
		t.Fatal("an empty cache marks nothing")
	}
	at := func(n int) {
		c.Conv, c.State, c.steps = b.FromFloats([]float32{float32(n)}, 1), b.FromFloats([]float32{float32(n)}, 1), n
	}
	at(4)
	c.Mark()
	at(7)
	c.Mark()
	at(9)
	if len(c.Tensors()) != 6 {
		t.Fatalf("live state and two marks are kept: %d tensors", len(c.Tensors()))
	}
	for n, want := range map[int]bool{0: true, 3: false, 4: true, 5: false, 7: true, 9: true, 12: true} {
		if c.CanTruncate(n) != want {
			t.Errorf("CanTruncate(%d) = %v", n, !want)
		}
	}
	c.Truncate(b, 4)
	if c.Len() != 4 || b.Floats(c.State)[0] != 4 || len(c.marks) != 1 {
		t.Fatalf("Truncate to a mark restores its state and drops the marks past it: len %d, %d marks", c.Len(), len(c.marks))
	}
	c.Forget(4)
	if len(c.marks) != 0 || len(c.Tensors()) != 2 {
		t.Fatal("a forgotten mark is gone")
	}
	c.Mark()
	c.Truncate(b, 0)
	if c.Len() != 0 || len(c.marks) != 0 || len(c.Tensors()) != 0 {
		t.Fatal("a reset drops the state and the marks")
	}
}
