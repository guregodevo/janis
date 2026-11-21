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
	K, V  engine.Tensor            // [1, kv heads, T, head dim], model dtype
	Conv  [convSlots]engine.Tensor // [1, kernel-1, channels], float32
	steps int

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
		c.K, c.V = b.Slice(s.k, 2, 0, total), b.Slice(s.v, 2, 0, total)
		if s.keep > 0 && total > s.keep {
			c.K, c.V = b.Slice(c.K, 2, total-s.keep, total), b.Slice(c.V, 2, total-s.keep, total)
		}
		for i, w := range s.windows {
			c.Conv[i] = b.Slice(w, 1, kept, kept+s.tail)
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
	if m, ok := c.marks[n]; ok {
		c.K, c.V, c.Conv, c.steps = m.k, m.v, m.conv, n
	} else {
		c.K, c.V, c.Conv, c.steps = nil, nil, [convSlots]engine.Tensor{}, 0
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
	c.Conv[slot] = b.Slice(window, 1, n-(kernel-1), n)
}
