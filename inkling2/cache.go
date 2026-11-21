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
	_, ok := c.marks[n]
	return ok
}

// Truncate returns to the mark at n, or resets. The marks past n go.
func (c *VisitCache) Truncate(_ engine.Backend, n int) {
	if n >= c.steps {
		return
	}
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
	if n := k.Shape()[2]; keep > 0 && n > keep {
		c.K, c.V = b.Slice(k, 2, n-keep, n), b.Slice(v, 2, n-keep, n)
	}
	return k, v
}
