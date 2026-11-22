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
}

// NewBatchDecoder is a batch decoder for the model.
func (m *Model) NewBatchDecoder() *BatchDecoder { return &BatchDecoder{m: m} }

// Step computes one decode step: conversation j reads toks[j] at
// offsets[j] on caches[j]. It returns B rows of vocab logits.
func (d *BatchDecoder) Step(b engine.Backend, toks []int32, offsets []int, caches [][]qwen.LayerCache) []float32 {
	if d.step != nil {
		if d.same(caches) && d.step.FitsRows(offsets) {
			floats := d.step.RunRows(toks, offsets)
			for j, cs := range caches {
				for _, c := range cs {
					if lc, ok := c.(*LinearCache); ok {
						lc.steps++
					}
				}
				d.lens[j] = offsets[j] + 1
			}
			return floats
		}
		d.Settle()
	}
	if floats := d.record(b, toks, offsets, caches); floats != nil {
		return floats
	}
	return b.Floats(d.m.ForwardBatch(b, toks, offsets, caches))
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
func (d *BatchDecoder) record(b engine.Backend, toks []int32, offsets []int, caches [][]qwen.LayerCache) []float32 {
	if d.off || len(caches) < 2 {
		return nil
	}
	rp, ok := b.(engine.BatchStepReplayer)
	if !ok {
		d.off = true
		return nil
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
	idsT := b.FromInt32(toks, len(toks))
	logits := d.m.forwardBatchT(b, idsT, offsets, caches)
	var next []engine.Tensor
	for _, i := range layers {
		l := caches[0][i].(*LinearCache).lent
		next = append(next, l.conv, l.state)
	}
	step, floats := rp.EndBatchStep(idsT, logits, next)
	lens := make([]int, len(offsets))
	for j, o := range offsets {
		lens[j] = o + 1
	}
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
