package inkling2

import "github.com/guregodevo/janis/engine"

// convSlots is the number of short convolutions in a layer: the keys', the
// values', the one after attention and the one after the MLP.
const convSlots = 4

// VisitCache is the cross-step state of one layer visit: the keys and values
// its attention reads (the last window-1 on a sliding layer, all of them on
// a full one) and the tail of each short convolution's input.
//
// A convolution tail holds no history and a sliding layer drops its old
// keys, so the cache returns only to a mark, as a recurrent layer's does.
type VisitCache struct {
	K, V engine.Tensor // [1, kv heads, T, head dim], model dtype
	// Conv is each convolution's tail, float32 [1, kernel-1, channels].
	// When rel is set it is the last window instead, [1, channels, kernel]
	// with the newest input last: a token's input is shifted into it in
	// one node (tail and kept turn the one into the other).
	Conv  [convSlots]engine.Tensor
	steps int

	// rel is set on a backend that runs a decode step's attention itself
	// (engine.RelAttention): the keys and values are kept f16, in buffers
	// the backend appends to in place, and a sliding layer's window is the
	// attention's mask. A cut is a shorter view.
	rel engine.RelAttention
	// first is the position of the first token K and V hold: a sliding
	// layer on such a backend drops the tokens its window no longer
	// reaches (trim), and a cut can then go back no further than first.
	first int

	marks map[int]visitMark

	// last is what the last forward of more than one token read, kept so
	// the cache can be cut back inside it: a run of drafted tokens is
	// verified in one pass, and the ones the model would not have written
	// are taken back.
	last *visitSpan
}

// visitSpan is one forward of n tokens: the keys and values before any
// trimming, and each convolution's whole input (its tail before the pass,
// then the n new rows).
type visitSpan struct {
	n       int
	k, v    engine.Tensor
	windows [convSlots]engine.Tensor
	keep    int // the bound on the keys kept, 0 for none
	tail    int // rows of a convolution tail (kernel-1)
}

// visitMark is the state at one earlier length. A forward pass makes new
// tensors and never writes into these.
type visitMark struct {
	k, v engine.Tensor
	conv [convSlots]engine.Tensor
}

func (c *VisitCache) Len() int { return c.steps }

func (c *VisitCache) CanTruncate(n int) bool {
	if n == 0 || n >= c.steps {
		return true
	}
	if c.rel != nil && n < c.first {
		return false // its tokens were dropped
	}
	if c.last != nil && n > c.steps-c.last.n {
		return true
	}
	_, ok := c.marks[n]
	return ok
}

// Truncate returns to the mark at n, or resets. The marks past n go.
func (c *VisitCache) Truncate(b engine.Backend, n int) {
	if n >= c.steps {
		return
	}
	if s := c.last; s != nil && n > c.steps-s.n {
		// Inside the last pass: its first kept tokens stay.
		kept := n - (c.steps - s.n)
		total := s.k.Shape()[2] - (s.n - kept)
		if c.rel != nil {
			c.K, c.V = c.cut(b, n)
		} else {
			c.K, c.V = b.Slice(s.k, 2, 0, total), b.Slice(s.v, 2, 0, total)
			if s.keep > 0 && total > s.keep {
				c.K, c.V = b.Slice(c.K, 2, total-s.keep, total), b.Slice(c.V, 2, total-s.keep, total)
			}
		}
		for i, w := range s.windows {
			if w != nil {
				c.Conv[i] = c.kept(b, b.Slice(w, 1, kept, kept+s.tail))
			}
		}
		c.steps, c.last = n, nil
		for at := range c.marks {
			if at > n {
				delete(c.marks, at)
			}
		}
		return
	}
	c.last = nil
	if m, ok := c.marks[n]; ok && c.rel != nil && n >= c.first {
		k, v := c.cut(b, n)
		c.K, c.V, c.Conv, c.steps = k, v, m.conv, n
	} else if ok && c.rel == nil {
		c.K, c.V, c.Conv, c.steps = m.k, m.v, m.conv, n
	} else {
		c.K, c.V, c.Conv, c.steps, c.first = nil, nil, [convSlots]engine.Tensor{}, 0, 0
	}
	for at := range c.marks {
		if at > n {
			delete(c.marks, at)
		}
	}
}

func (c *VisitCache) Mark() {
	if c.steps == 0 {
		return
	}
	if c.marks == nil {
		c.marks = map[int]visitMark{}
	}
	if c.rel != nil {
		c.marks[c.steps] = visitMark{conv: c.Conv}
		return
	}
	c.marks[c.steps] = visitMark{k: c.K, v: c.V, conv: c.Conv}
}

func (c *VisitCache) Forget(n int) { delete(c.marks, n) }

func (c *VisitCache) Tensors() []engine.Tensor {
	var ts []engine.Tensor
	add := func(k, v engine.Tensor, conv [convSlots]engine.Tensor) {
		for _, t := range append([]engine.Tensor{k, v}, conv[:]...) {
			if t != nil {
				ts = append(ts, t)
			}
		}
	}
	add(c.K, c.V, c.Conv)
	if c.last != nil {
		add(c.last.k, c.last.v, c.last.windows)
	}
	for _, m := range c.marks {
		add(m.k, m.v, m.conv)
	}
	return ts
}

// update appends this step's keys and values and returns all the attention
// may read. keep bounds what stays for the next step (0: everything).
func (c *VisitCache) update(b engine.Backend, k, v engine.Tensor, keep int) (engine.Tensor, engine.Tensor) {
	if c.rel != nil {
		k, v = b.Cast(k, engine.F16), b.Cast(v, engine.F16)
		if c.K != nil {
			k, v = b.Concat(c.K, k, 2), c.rel.AppendValuesT(c.V, v)
		}
		c.K, c.V = k, v
		return k, v
	}
	if c.K != nil {
		k, v = b.Concat(c.K, k, 2), b.Concat(c.V, v, 2)
	}
	c.K, c.V = k, v
	if c.last != nil {
		c.last.k, c.last.v, c.last.keep = k, v, keep
	}
	if n := k.Shape()[2]; keep > 0 && n > keep {
		c.K, c.V = b.Slice(k, 2, n-keep, n), b.Slice(v, 2, n-keep, n)
	}
	return k, v
}

// cut is K and V at n positions, on a backend that keeps them whole.
func (c *VisitCache) cut(b engine.Backend, n int) (engine.Tensor, engine.Tensor) {
	return b.Slice(c.K, 2, 0, n-c.first), b.Slice(c.V, 2, 0, n-c.first)
}

// trimSlack is how far past its window a sliding layer's keys and values
// grow before the oldest are dropped. A drop copies what is kept into a
// buffer of its own, so it is done in strides; a cut back into a reply
// reaches this far behind the window.
var trimSlack = 4096

// trim drops the keys and values a sliding layer's window no longer
// reaches, on a backend that keeps them whole (rel): the next token at
// position steps sees window-1 before it. Without it every sliding layer
// kept the whole context, and a 32 GB card held about 40,000 tokens of the
// published model where its full layers alone allow 160,000 ("no room on
// the device" at 28,000 tokens of reply, 2026-11-27). Called between
// forward passes; the marks before the new first position go.
func (c *VisitCache) trim(window int) {
	if c.rel == nil || window <= 0 || c.K == nil {
		return
	}
	keep := window - 1
	held := c.steps - c.first
	if held <= keep+trimSlack {
		return
	}
	drop := held - keep
	c.K, c.V = c.rel.TrimFront(c.K, drop), c.rel.TrimFront(c.V, drop)
	c.first += drop
	c.last = nil
	for at := range c.marks {
		if at < c.first {
			delete(c.marks, at)
		}
	}
}

// begin opens a forward of n tokens: one of several is kept whole, to be
// cut back into.
func (c *VisitCache) begin(n int) {
	c.last = nil
	if n > 1 {
		c.last = &visitSpan{n: n}
	}
}

// slide records a convolution's input for this forward and keeps its tail
// for the next: the last kernel-1 rows.
func (c *VisitCache) slide(b engine.Backend, slot int, window engine.Tensor, kernel int) {
	if c.last != nil {
		c.last.windows[slot], c.last.tail = window, kernel-1
	}
	n := window.Shape()[1]
	c.Conv[slot] = c.kept(b, b.Slice(window, 1, n-(kernel-1), n))
}

// kept is a tail [1, kernel-1, channels] as the cache keeps it.
func (c *VisitCache) kept(b engine.Backend, tail engine.Tensor) engine.Tensor {
	if c.rel == nil {
		return tail
	}
	ch := tail.Shape()[2]
	return b.Concat(b.FromFloats(make([]float32, ch), 1, ch, 1), b.Transpose(tail, 0, 2, 1), 2)
}

// tail is slot's tail as [1, kernel-1, channels], nil at the start.
func (c *VisitCache) tail(b engine.Backend, slot int) engine.Tensor {
	t := c.Conv[slot]
	if t == nil || c.rel == nil {
		return t
	}
	return b.Transpose(b.Slice(t, 2, 1, t.Shape()[2]), 0, 2, 1)
}

// Carry is the state a recorded decode step reads and replaces: the tails
// of the convolutions (qwen's replayCache; a slot a layer does not use is
// nil and left out). The keys and values are
// written in place and are no part of it.
func (c *VisitCache) Carry() []engine.Tensor {
	ts := []engine.Tensor{}
	for _, t := range c.Conv {
		if t != nil {
			ts = append(ts, t)
		}
	}
	return ts
}

func (c *VisitCache) SetCarry(ts []engine.Tensor) {
	for i, t := range c.Conv {
		if t != nil && len(ts) > 0 {
			c.Conv[i], ts = ts[0], ts[1:]
		}
	}
}

// Settle sets the cache at n tokens after a recorded step's runs: the
// tails were advanced in place, the keys and values are viewed at n
// positions (the backend counts the ones a trim dropped).
func (c *VisitCache) Settle(rp engine.StepReplayer, n int) {
	if c.rel != nil && c.K != nil {
		c.K, c.V = rp.CacheView(c.K, n), rp.CacheView(c.V, n)
	}
	c.steps, c.last = n, nil
}
