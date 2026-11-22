package qwen

import (
	"math"
	"math/rand"

	"github.com/guregodevo/janis/engine"
)

// BatchStepper computes one decode step for B sessions: session j reads
// toks[j] at offsets[j] on its caches[j]. It returns B rows of vocab
// logits (qwen35.BatchDecoder).
type BatchStepper interface {
	Step(b engine.Backend, toks []int32, offsets []int, caches [][]LayerCache) []float32
}

// Gen is one session's reply in progress, decoded a token at a time together
// with other sessions' by StepBatch. Begin makes one; Finish records it.
// The tokens it commits are the ones Generate commits, without speculation
// and without replay: StepBatch reads every token through the model.
type Gen struct {
	S      *Session
	p      SampleParams
	rng    *rand.Rand
	newIDs []int32
	nGen   int
	gen    []int32
	tok    int32 // the token sampled and not yet committed
	valid  int   // tokens the caches hold
	done   bool
}

// Begin prefills newIDs into the session, as Generate does, and samples the
// reply's first token. It returns nil when newIDs is empty or Cancel fired
// during the prefill.
func (s *Session) Begin(b engine.Backend, newIDs []int32, nGen int, p SampleParams) *Gen {
	if len(newIDs) == 0 {
		return nil
	}
	if r, ok := b.(engine.CacheReserver); ok {
		r.ReserveCache(len(newIDs) + nGen)
	}
	floats, ok := s.prefill(b, newIDs, p.Cancel)
	if !ok {
		return nil
	}
	g := &Gen{S: s, p: p, rng: rand.New(rand.NewSource(int64(p.Seed))), newIDs: newIDs, nGen: nGen, valid: len(newIDs)}
	g.tok = sampleToken(floats, p, g.rng, nil)
	if p.Stop[g.tok] && nGen > 0 {
		// As in Generate: a reply never ends before its first token.
		for id := range p.Stop {
			if int(id) >= 0 && int(id) < len(floats) {
				floats[id] = float32(math.Inf(-1))
			}
		}
		g.tok = sampleToken(floats, p, g.rng, nil)
	}
	g.commit()
	return g
}

// commit takes the sampled token into the reply, or ends the reply on a stop
// token, the token budget or Cancel: Generate's loop head.
func (g *Gen) commit() {
	if g.p.Stop[g.tok] {
		g.done = true
		return
	}
	g.gen = append(g.gen, g.tok)
	if g.p.Grammar != nil {
		g.p.Grammar.Advance(g.tok)
	}
	if g.p.OnToken != nil {
		g.p.OnToken(g.tok)
	}
	if len(g.gen) >= g.nGen || (g.p.Cancel != nil && g.p.Cancel()) {
		g.done = true
	}
}

// Done reports whether the reply has ended.
func (g *Gen) Done() bool { return g.done }

// Finish records the prompt and the reply as the session's tokens, for the
// next prompt's prefix reuse, and returns the reply.
func (g *Gen) Finish() []int32 {
	g.S.IDs = append(append([]int32(nil), g.newIDs...), g.gen...)
	return g.gen
}

// StepBatch reads the last committed token of every Gen not done in one
// step, then samples and commits each one's next token. The order of gens
// is the batch's row order: a stable order lets the model keep the batch's
// stacked state between steps.
func StepBatch(b engine.Backend, st BatchStepper, gens []*Gen) {
	var live []*Gen
	for _, g := range gens {
		if !g.done {
			live = append(live, g)
		}
	}
	if len(live) == 0 {
		return
	}
	ids := make([]int32, len(live))
	offsets := make([]int, len(live))
	caches := make([][]LayerCache, len(live))
	var prev []engine.Tensor
	for j, g := range live {
		ids[j], offsets[j], caches[j] = g.tok, g.valid, g.S.Caches
		prev = append(prev, cacheTensors(g.S.Caches)...)
	}
	all := st.Step(b, ids, offsets, caches)
	if sw, ok := b.(engine.Sweeper); ok {
		var kept []engine.Tensor
		for _, g := range live {
			pinCaches(sw, g.S.Caches)
			kept = append(kept, cacheTensors(g.S.Caches)...)
		}
		unpinGone(sw, prev, kept)
		sw.Sweep()
	}
	width := len(all) / len(live)
	for j, g := range live {
		floats := all[j*width : (j+1)*width]
		g.valid++
		if g.S.OnLogits != nil {
			g.S.OnLogits(floats)
		}
		g.tok = sampleToken(floats, g.p, g.rng, g.gen)
		if len(g.gen) < len(g.S.Force) {
			g.tok = g.S.Force[len(g.gen)]
		}
		g.commit()
	}
}

// Release takes back the rows sessions lent to the batch they decoded in
// last (recurrent states' stacks, key/value pools), all of them before any
// pin changes. Call it on every member of that
// batch when the batch changes (a reply ends, or a session is used alone):
// the members share one stacked tensor, and pins are flags, so a member
// that took its row back alone would unpin the stack under the others.
func Release(b engine.Backend, sessions []*Session) {
	var prev []engine.Tensor
	for _, s := range sessions {
		prev = append(prev, cacheTensors(s.Caches)...)
	}
	for _, s := range sessions {
		for _, c := range s.Caches {
			if kv, ok := c.(*KVCache); ok {
				kv.Own() // a key/value pool row
			} else if rc, ok := c.(replayCache); ok {
				rc.Carry() // a recurrent state's row of the stack
			}
		}
	}
	if sw, ok := b.(engine.Sweeper); ok {
		var kept []engine.Tensor
		for _, s := range sessions {
			pinCaches(sw, s.Caches)
			kept = append(kept, cacheTensors(s.Caches)...)
		}
		unpinGone(sw, prev, kept)
	}
}
