package qwen35

import (
	"github.com/guregodevo/janis/engine"
	"github.com/guregodevo/janis/qwen"
)

// BatchDecoder runs decode steps of several conversations, one token each,
// on a recorded step when the backend records one
// (engine.BatchStepReplayer): the step is recorded once for a set of
// conversations in an order and run again while the same ones decode in
// the same order and their caches have room, then recorded again.
type BatchDecoder struct {
	m       *Model
	rp      engine.BatchStepReplayer
	step    engine.BatchStep
	members []qwen.LayerCache // each conversation's first cache, in the step's order
	lens    []int             // each conversation's length after the step's runs
	all     [][]qwen.LayerCache
	off     bool
	failed  int
	// Last is how the last Step ran: "replay" (the recorded step run
	// again), "record" (recorded, then run) or "plain" (computed as usual).
	Last string
}

// NewBatchDecoder is a batch decoder for the model.
func (m *Model) NewBatchDecoder() *BatchDecoder { return &BatchDecoder{m: m} }

// Step computes one decode step: conversation j reads toks[j] at
// offsets[j] on caches[j]. It returns B rows of vocab logits.
func (d *BatchDecoder) Step(b engine.Backend, toks []int32, offsets []int, caches [][]qwen.LayerCache) []float32 {
	return d.StepOn(b, toks, offsets, caches, nil)
}

// StepOn is Step with rows held: on[j] false holds conversation j, whose
// caches stay as they were (its row is computed, and its logits are not
// to be read). on nil holds none. A batch whose rows stay while some are
// held keeps its stacks and its recorded step (qwen.StepRows).
func (d *BatchDecoder) StepOn(b engine.Backend, toks []int32, offsets []int, caches [][]qwen.LayerCache, on []bool) []float32 {
	if d.step != nil {
		hs, holds := d.step.(engine.HeldStep)
		if d.same(caches) && d.step.FitsRows(offsets) && (on == nil || holds) {
			var floats []float32
			if on != nil {
				floats = hs.RunRowsOn(toks, offsets, on)
			} else {
				floats = d.step.RunRows(toks, offsets)
			}
			for j, cs := range caches {
				if on != nil && !on[j] {
					d.lens[j] = offsets[j]
					continue
				}
				for _, c := range cs {
					switch c := c.(type) {
					case *LinearCache:
						c.steps++
					case *qwen.KVCache:
						if p, _ := c.Lent(); p != nil {
							c.Grow(offsets[j] + 1)
						}
					}
				}
				d.lens[j] = offsets[j] + 1
			}
			d.Last = "replay"
			return floats
		}
		d.Settle()
	}
	if floats := d.record(b, toks, offsets, caches, on); floats != nil {
		d.Last = "record"
		return floats
	}
	d.Last = "plain"
	var held *rowsOn
	if on != nil {
		held = rowsOnFor(b, on)
	}
	floats := b.Floats(d.m.forwardBatchT(b, b.FromInt32(toks, len(toks)), offsets, caches, held))
	holdRows(b, caches, offsets, on)
	return floats
}

// rowsOnFor is the rows on and off: the backend's (a recorded step's
// inputs) when it holds rows, else tensors of their own.
func rowsOnFor(b engine.Backend, on []bool) *rowsOn {
	if rh, ok := b.(engine.RowHolder); ok {
		o, f := rh.RowsOn(on)
		return &rowsOn{on: o, off: f}
	}
	ov, fv := make([]float32, len(on)), make([]float32, len(on))
	for j, r := range on {
		if r {
			ov[j] = 1
		} else {
			fv[j] = 1
		}
	}
	return &rowsOn{on: b.FromFloats(ov, len(on), 1, 1, 1), off: b.FromFloats(fv, len(on), 1, 1, 1)}
}

// holdRows sets a held conversation's caches back to its length after a
// step that wrote every row: the recurrent state kept itself (rowsOn), the
// keys and values the step wrote past the length are not read.
func holdRows(b engine.Backend, caches [][]qwen.LayerCache, offsets []int, on []bool) {
	for j, cs := range caches {
		if on == nil || on[j] {
			continue
		}
		for _, c := range cs {
			switch c := c.(type) {
			case *LinearCache:
				c.steps = offsets[j]
			case *qwen.KVCache:
				if p, _ := c.Lent(); p != nil {
					c.Grow(offsets[j])
				} else if c.Len() > offsets[j] {
					c.Truncate(b, offsets[j])
				}
			}
		}
	}
}

// Settle ends the recorded step: each conversation's caches are viewed at
// their real lengths. The scheduler calls it before anything else touches
// the caches (a prefill, a conversation joining or leaving).
func (d *BatchDecoder) Settle() {
	if d.step == nil {
		return
	}
	for j, cs := range d.all {
		for _, c := range cs {
			if kv, ok := c.(*qwen.KVCache); ok {
				settleKV(kv, d.rp, d.lens[j])
			}
		}
	}
	d.step.Close()
	d.step, d.members, d.all, d.lens = nil, nil, nil, nil
}

// Drop ends the recorded step without setting the caches at their
// lengths: for caches about to be dropped (Engine.ResetSessions).
func (d *BatchDecoder) Drop() {
	if d.step != nil {
		d.step.Close()
	}
	d.step, d.members, d.all, d.lens = nil, nil, nil, nil
}

func (d *BatchDecoder) same(caches [][]qwen.LayerCache) bool {
	if len(caches) != len(d.members) {
		return false
	}
	for j, cs := range caches {
		if cs[0] != d.members[j] {
			return false
		}
	}
	return true
}

// record records the step and computes it, or returns nil when it cannot
// be recorded now: the conversations' recurrent states are not stacked in
// this order yet (the plain step that comes first stacks them), or the
// backend does not record.
func (d *BatchDecoder) record(b engine.Backend, toks []int32, offsets []int, caches [][]qwen.LayerCache, on []bool) []float32 {
	if d.off || len(caches) < 2 {
		return nil
	}
	rp, ok := b.(engine.BatchStepReplayer)
	if !ok {
		d.off = true
		return nil
	}
	if _, holds := b.(engine.RowHolder); on != nil && !holds {
		return nil // computed as usual
	}
	var stacks []*lentRow // per recurrent layer, the batch's stack
	var layers []int
	for i, c := range caches[0] {
		lc, ok := c.(*LinearCache)
		if !ok {
			continue
		}
		col := make([]*LinearCache, len(caches))
		for j := range caches {
			col[j] = caches[j][i].(*LinearCache)
		}
		st := lc.lentStack(col)
		if st == nil {
			return nil
		}
		stacks = append(stacks, st)
		layers = append(layers, i)
	}
	var carry []engine.Tensor
	for _, st := range stacks {
		carry = append(carry, st.conv, st.state)
	}
	moved := rp.BeginBatchStep(offsets, carry)
	if moved == nil {
		d.off = true
		return nil
	}
	hand := func(ts []engine.Tensor) {
		for k, i := range layers {
			for j := range caches {
				caches[j][i].(*LinearCache).lent = &lentRow{b: b, conv: ts[2*k], state: ts[2*k+1], row: j}
			}
		}
	}
	hand(moved)
	var held *rowsOn
	if on != nil {
		held = rowsOnFor(b, on)
	}
	idsT := b.FromInt32(toks, len(toks))
	logits := d.m.forwardBatchT(b, idsT, offsets, caches, held)
	var next []engine.Tensor
	for _, i := range layers {
		l := caches[0][i].(*LinearCache).lent
		next = append(next, l.conv, l.state)
	}
	step, floats := rp.EndBatchStep(idsT, logits, next)
	lens := make([]int, len(offsets))
	for j, o := range offsets {
		lens[j] = o + 1
		if on != nil && !on[j] {
			lens[j] = o
		}
	}
	holdRows(b, caches, offsets, on)
	if step == nil {
		// Computed as usual: the caches hold the step; view the key and
		// value caches at their lengths.
		d.failed++
		d.off = d.failed >= 3
		for j, cs := range caches {
			for _, c := range cs {
				if kv, ok := c.(*qwen.KVCache); ok {
					settleKV(kv, rp, lens[j])
				}
			}
		}
		return floats
	}
	hand(moved)
	d.rp, d.step, d.failed, d.lens = rp, step, 0, lens
	d.members = make([]qwen.LayerCache, len(caches))
	d.all = make([][]qwen.LayerCache, len(caches))
	for j, cs := range caches {
		d.members[j], d.all[j] = cs[0], cs
	}
	return floats
}

// settleKV views a key and value cache a recorded step wrote at n tokens.
func settleKV(kv *qwen.KVCache, rp engine.BatchStepReplayer, n int) {
	if kv.K != nil {
		kv.K, kv.V = rp.CacheView(kv.K, n), rp.CacheView(kv.V, n)
	}
}
