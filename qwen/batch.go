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
	// late: tok is the prompt's last token, read by the first step, whose
	// logits give the reply's first token (BeginLate).
	late bool
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
	floats, ok := s.prefill(b, newIDs, p.Cancel, true) // the prompt-end mark, as Generate
	if !ok {
		return nil
	}
	g := &Gen{S: s, p: p, rng: rand.New(rand.NewSource(int64(p.Seed))), newIDs: newIDs, nGen: nGen, valid: len(newIDs)}
	g.first(floats)
	return g
}

// BeginLate is Begin with the prompt's last token left to the first
// batched step: the prefill reads all but the last token and marks the
// state there, where Begin marks it too, and the step that reads the last
// token with the other replies' gives the reply's first token. Begin reads
// that token in a forward of its own, every weight read for one token. A
// prompt the caches hold all but its last token of, or a reply of no
// token, begins as Begin does.
func (s *Session) BeginLate(b engine.Backend, newIDs []int32, nGen int, p SampleParams) *Gen {
	n := len(newIDs)
	if n < 3 || nGen < 1 || (lcp(s.IDs, newIDs) > n-2 && cachedLen(s.Caches) > n-2) {
		return s.Begin(b, newIDs, nGen, p)
	}
	if r, ok := b.(engine.CacheReserver); ok {
		r.ReserveCache(n + nGen)
	}
	if _, ok := s.prefill(b, newIDs[:n-1], p.Cancel, false); !ok {
		return nil
	}
	if !canTruncate(s.Caches, 1) {
		s.mark(n - 1) // the end of the prompt but its last token, as Begin's prefill marks
	}
	return &Gen{S: s, p: p, rng: rand.New(rand.NewSource(int64(p.Seed))), newIDs: newIDs, nGen: nGen,
		valid: n - 1, tok: newIDs[n-1], late: true}
}

// first samples the reply's first token from the prompt's last logits.
func (g *Gen) first(floats []float32) {
	g.tok = sampleToken(floats, g.p, g.rng, nil)
	if g.p.Stop[g.tok] && g.nGen > 0 {
		// As in Generate: a reply never ends before its first token.
		for id := range g.p.Stop {
			if int(id) >= 0 && int(id) < len(floats) {
				floats[id] = float32(math.Inf(-1))
			}
		}
		g.tok = sampleToken(floats, g.p, g.rng, nil)
	}
	g.commit()
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
		g.take(all[j*width : (j+1)*width])
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

// RowStepper is a BatchStepper that runs with rows held
// (qwen35.BatchDecoder.StepOn): on[j] false keeps conversation j's caches
// as they were.
type RowStepper interface {
	StepOn(b engine.Backend, toks []int32, offsets []int, caches [][]LayerCache, on []bool) []float32
}

// RowJoiner is a layer cache that joins a batch's stack at its row in
// place: sib is the same layer's cache of a conversation in the stack,
// rows the stack's rows.
type RowJoiner interface {
	CanJoin(b engine.Backend, sib LayerCache, j, rows int) bool
	Join(b engine.Backend, sib LayerCache, j int)
}

// StepRows is StepBatch on fixed rows: every session decodes as its row,
// in sessions' order, whether it has a reply or not. gens[i] is session
// i's reply, nil when it has none; a session without one, or whose reply
// is done, is held: computed with the others, its caches kept as they
// were. The rows never change, so the stacks and the recorded step stay
// while replies come and go (Reclaim, Relend).
func StepRows(b engine.Backend, st RowStepper, sessions []*Session, gens []*Gen) {
	n := len(sessions)
	ids := make([]int32, n)
	offsets := make([]int, n)
	caches := make([][]LayerCache, n)
	on := make([]bool, n)
	any := false
	var prev []engine.Tensor
	for i, s := range sessions {
		if s.Caches == nil {
			s.Caches = make([]LayerCache, s.NLayers)
			for l := range s.Caches {
				s.Caches[l] = s.newCache(l)
			}
		}
		if g := gens[i]; g != nil && !g.done {
			ids[i], offsets[i], on[i], any = g.tok, g.valid, true, true
		} else {
			offsets[i] = cachedLen(s.Caches)
		}
		caches[i] = s.Caches
		prev = append(prev, cacheTensors(s.Caches)...)
	}
	if !any {
		return
	}
	all := st.StepOn(b, ids, offsets, caches, on)
	if sw, ok := b.(engine.Sweeper); ok {
		var kept []engine.Tensor
		for _, s := range sessions {
			pinCaches(sw, s.Caches)
			kept = append(kept, cacheTensors(s.Caches)...)
		}
		unpinGone(sw, prev, kept)
		sw.Sweep()
	}
	width := len(all) / n
	for i, g := range gens {
		if on[i] {
			g.take(all[i*width : (i+1)*width])
		}
	}
}

// take reads the logits of the token the step read: the reply's next token.
func (g *Gen) take(floats []float32) {
	g.valid++
	if g.late {
		g.late = false
		g.first(floats)
		return
	}
	if g.S.OnLogits != nil {
		g.S.OnLogits(floats)
	}
	g.tok = sampleToken(floats, g.p, g.rng, g.gen)
	if len(g.gen) < len(g.S.Force) {
		g.tok = g.S.Force[len(g.gen)]
	}
	g.commit()
}

// Reclaim takes session j's rows back from the batch's stacks, for a use
// of its own (a prefill); the other sessions' rows stay where they are.
func Reclaim(b engine.Backend, sessions []*Session, j int) {
	restack(b, sessions, func() {
		for _, c := range sessions[j].Caches {
			if kv, ok := c.(*KVCache); ok {
				kv.Own()
			} else if rc, ok := c.(replayCache); ok {
				rc.Carry()
			}
		}
	})
}

// Relend writes session j's caches into its row of the stacks the other
// sessions decode in, and lends them there: the batch's stacks and its
// recorded step stay as they are. False, changing nothing, when a layer
// cannot join (no stack yet, no room in a pool, a backend that does not
// write rows): the next step then stacks the rows again.
func Relend(b engine.Backend, sessions []*Session, j int) bool {
	s := sessions[j]
	sib := -1
	for i, o := range sessions {
		if i != j && o.Caches != nil && lentAll(o.Caches) {
			sib = i
			break
		}
	}
	if sib < 0 || s.Caches == nil {
		return false
	}
	for l, c := range s.Caches {
		rj, ok := c.(RowJoiner)
		if !ok || !rj.CanJoin(b, sessions[sib].Caches[l], j, len(sessions)) {
			return false
		}
	}
	restack(b, sessions, func() {
		for l, c := range s.Caches {
			c.(RowJoiner).Join(b, sessions[sib].Caches[l], j)
		}
	})
	return true
}

// PoolsFull reports whether a session's keys and values fill the pool
// they are lent to: the next step makes a new pool, every row copied.
func PoolsFull(sessions []*Session) bool {
	for _, s := range sessions {
		for _, c := range s.Caches {
			if kv, ok := c.(*KVCache); ok {
				if p, _ := kv.Lent(); p != nil && kv.Len() >= p.Room() {
					return true
				}
			}
		}
	}
	return false
}

// lentAll reports whether every cache is lent to a batch's stack.
func lentAll(caches []LayerCache) bool {
	for _, c := range caches {
		if l, ok := c.(interface{ Lent() bool }); ok {
			if !l.Lent() {
				return false
			}
		} else if kv, ok := c.(*KVCache); ok {
			if p, _ := kv.Lent(); p == nil {
				return false
			}
		} else {
			return false
		}
	}
	return true
}

// restack runs change on the sessions' caches and moves the pins: the
// stacks the sessions share stay pinned while any session holds them.
func restack(b engine.Backend, sessions []*Session, change func()) {
	var prev []engine.Tensor
	for _, s := range sessions {
		prev = append(prev, cacheTensors(s.Caches)...)
	}
	change()
	if sw, ok := b.(engine.Sweeper); ok {
		var kept []engine.Tensor
		for _, s := range sessions {
			pinCaches(sw, s.Caches)
			kept = append(kept, cacheTensors(s.Caches)...)
		}
		unpinGone(sw, prev, kept)
	}
}

// CanJoin: the stack sib's pool, of rows rows, has room for the cache
// and writes rows (RowJoiner).
func (c *KVCache) CanJoin(_ engine.Backend, sib LayerCache, _, rows int) bool {
	s, ok := sib.(*KVCache)
	if !ok || s.pool == nil || c.pool != nil || s.pool.Rows() != rows {
		return false
	}
	if _, ok := s.pool.(engine.PoolRowWriter); !ok {
		return false
	}
	return c.Len() < s.pool.Room()
}

// Join writes the cache's tokens into row j of sib's pool and lends them
// there (RowJoiner).
func (c *KVCache) Join(_ engine.Backend, sib LayerCache, j int) {
	p := sib.(*KVCache).pool
	n := c.Len()
	if n > 0 {
		p.(engine.PoolRowWriter).SetRow(j, c.K, c.V)
	}
	c.Lend(p, j, n)
}
