package qwen

import "github.com/guregodevo/janis/engine"

// replayCache is a layer cache a recorded decode step can run on
// (engine.StepReplayer).
type replayCache interface {
	// Carry is the state a step reads and replaces, nil for none;
	// SetCarry hands the cache the tensors that hold it from now on.
	Carry() []engine.Tensor
	SetCarry([]engine.Tensor)
	// Settle sets the cache at n tokens after a recorded step's runs.
	Settle(rp engine.StepReplayer, n int)
}

// replaying is the decode loop's recorded step, when the backend records
// one: the step is run again for each token until its caches' bucket is
// full, then recorded again.
type replaying struct {
	step   engine.Step
	rp     engine.StepReplayer
	off    bool // the backend does not replay, or recordings keep failing
	failed int  // recordings that could not be replayed, in a row
}

// maxRecordFailures is how many recordings in a row may fail before the
// loop stops trying: the first decode step after a prompt moves the KV
// caches into buffers with room, which a recording cannot do, and the next
// one records.
const maxRecordFailures = 3

// replay computes the step for tok at offset at on a recorded step and
// returns its logits, or nil when the session decodes as usual.
func (s *Session) replay(b engine.Backend, r *replaying, tok int32, at int) []float32 {
	if r.off || s.Speculator != nil {
		return nil
	}
	if r.step != nil {
		if r.step.Fits(at) {
			return r.step.Run(tok, at)
		}
		r.settle(s, at)
	}
	rp, ok := b.(engine.StepReplayer)
	if !ok {
		r.off = true
		return nil
	}
	caches := make([]replayCache, len(s.Caches))
	var carry []engine.Tensor
	for i, c := range s.Caches {
		rc, ok := c.(replayCache)
		if !ok {
			r.off = true
			return nil
		}
		caches[i] = rc
		carry = append(carry, rc.Carry()...)
	}
	moved := rp.BeginStep(at, carry)
	if moved == nil {
		r.off = true
		return nil
	}
	hand := func(ts []engine.Tensor) {
		for _, rc := range caches {
			n := len(rc.Carry())
			rc.SetCarry(ts[:n])
			ts = ts[n:]
		}
	}
	prev := cacheTensors(s.Caches)
	hand(moved)
	ids := b.FromInt32([]int32{tok}, 1)
	logits := s.Forward(b, ids, 1, at, s.Caches)
	var next []engine.Tensor
	for _, rc := range caches {
		next = append(next, rc.Carry()...)
	}
	step, floats := rp.EndStep(ids, logits, next)
	if step == nil {
		// Computed as usual: the caches hold next; read at their length.
		r.failed++
		r.off = r.failed >= maxRecordFailures
		for _, rc := range caches {
			rc.Settle(rp, at+1)
		}
	} else {
		hand(moved)
		r.step, r.rp, r.failed = step, rp, 0

	}
	if sw, ok := b.(engine.Sweeper); ok {
		pinCaches(sw, s.Caches)
		unpinGone(sw, prev, cacheTensors(s.Caches))
		sw.Sweep()
	}
	return floats
}

// settle ends the recorded step: the caches are set at n tokens.
func (r *replaying) settle(s *Session, n int) {
	if r.step == nil {
		return
	}
	for _, c := range s.Caches {
		c.(replayCache).Settle(r.rp, n)
	}
	r.step.Close()
	r.step = nil
}

// Carry: a KV cache has no state a step replaces; the step writes into it.
func (c *KVCache) Carry() []engine.Tensor { return nil }

func (c *KVCache) SetCarry([]engine.Tensor) {}

// Settle views the keys and values at n tokens.
func (c *KVCache) Settle(rp engine.StepReplayer, n int) {
	if c.K != nil {
		c.K, c.V = rp.CacheView(c.K, n), rp.CacheView(c.V, n)
	}
}
