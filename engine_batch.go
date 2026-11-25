package llm

import (
	"errors"
	"os"
	"slices"
	"time"

	"github.com/guregodevo/janis/chat"
	"github.com/guregodevo/janis/engine"
	"github.com/guregodevo/janis/qwen"
)

// parallel is how many replies a server decodes together (JANIS_PARALLEL).
// 1, the default, decodes one at a time, as before.
var parallel = envInt("JANIS_PARALLEL", 1)

// fixedRows (JANIS_FIXED_ROWS=1) decodes every slot as a row of its own,
// held while the slot has no reply: a reply that ends leaves its row in
// place, and a new one is written into its slot's row after its prefill.
// The batch's stacks and its recorded step then stay while replies come
// and go, where otherwise each change takes every row back, stacks them
// again and records the step again. Off, the batch is the replies, as before.
var fixedRows = os.Getenv("JANIS_FIXED_ROWS") == "1"

// ErrNoSlot is BeginChat's answer when every slot holds a reply.
var ErrNoSlot = errors.New("llm: every slot holds a reply")

// A Reply is a chat reply in progress: BeginChat makes one, StepChats
// advances it a token, FinishChat ends it.
type Reply interface{ Done() bool }

// chatGen is the Engine's Reply: a generation on one slot's session.
type chatGen struct {
	slot int
	g    *qwen.Gen
}

func (c *chatGen) Done() bool { return c.g.Done() }

// Slots is how many replies StepChats decodes together: JANIS_PARALLEL on a
// Qwen3.5 text model, whose ForwardBatch reads the weights once for every
// reply; 1 on any other model, and with JANIS_PARALLEL unset.
func (e *Engine) Slots() int {
	if e.q35 == nil || e.vision != nil || e.ink != nil || parallel < 2 {
		return 1
	}
	return parallel
}

// padForSlots turns padded cache reads on when replies decode together
// (JANIS_GGML_PADDECODE=1 for the process): a batched step, plain or
// recorded, then reads every cache padded to the recorded step's bucket, as
// a reply decoded alone with replay does, and greedy replies decoded
// together are the replies decoded alone. Exact reads round a few logits
// differently and can flip a near tie. JANIS_PARALLEL unset leaves the
// reads as they were.
func (e *Engine) padForSlots() {
	if e.Slots() > 1 && os.Getenv("JANIS_GGML_PADDECODE") == "" {
		os.Setenv("JANIS_GGML_PADDECODE", "1")
	}
}

// slotSessions makes the slots' sessions once. Slot 0 is the main session,
// so a server that decoded one reply at a time before keeps its prefix.
func (e *Engine) slotSessions() {
	if e.slots != nil {
		return
	}
	n := e.Slots()
	e.slots = make([]*qwen.Session, n)
	e.slotBusy = make([]bool, n)
	e.slotUsed = make([]int, n)
	e.slots[0] = e.session
	for i := 1; i < n; i++ {
		e.slots[i] = e.q35.NewSession()
	}
	e.slotDecoder = e.q35.NewBatchDecoder()
	if fixedRows && os.Getenv("JANIS_POOL_SPARE") == "" {
		// A pool with room for a few more turns: a reply that joins a
		// pool with no room for it stacks every row again.
		os.Setenv("JANIS_POOL_SPARE", "2048")
	}
	e.slotStats = newBatchStats()
}

// settleSlots ends the batch's recorded step and takes back the rows the
// slots lent to it: before a slot is used alone or the batch changes.
func (e *Engine) settleSlots() {
	e.slotDecoder.Settle()
	qwen.Release(e.bk, e.slots)
	e.slotStep = nil
}

// restackSlots takes every row back from the batch and frees the old
// stacks, for the next step to stack the rows again (fixed rows: a reply
// that could not join, or a pool with no room left). The next step's pool
// is made while the rows' copies are held; the old pool held too ran a
// 6-slot 27B out of a 48 GB card.
func (e *Engine) restackSlots() {
	e.settleSlots()
	if sw, ok := e.bk.(engine.Sweeper); ok {
		sw.Sweep()
	}
}

// pickSlot is the free slot whose tokens share the longest prefix with ids,
// the least recently used among equals; -1 when every slot is busy. An
// agent's next call lands on the slot that holds its conversation.
func (e *Engine) pickSlot(ids []int32) int {
	best, bestLen := -1, -1
	for i, s := range e.slots {
		if e.slotBusy[i] {
			continue
		}
		n := sharedPrefix(s.IDs, ids)
		if n > bestLen || (n == bestLen && e.slotUsed[i] < e.slotUsed[best]) {
			best, bestLen = i, n
		}
	}
	return best
}

func sharedPrefix(a, b []int32) int {
	n := 0
	for n < len(a) && n < len(b) && a[n] == b[n] {
		n++
	}
	return n
}

// BeginChat prefills msgs on a free slot and samples the reply's first
// token: Chat's prompt, sampling and token budget. It returns ErrNoSlot when
// every slot is busy, and a nil Reply when opts.Cancel fired during the
// prefill. A message with images is refused: the slots read text only.
func (e *Engine) BeginChat(msgs []Message, opts Options, onDelta func(string)) (Reply, error) {
	for _, m := range msgs {
		if len(m.Images) > 0 {
			return nil, errors.New("llm: a batched reply reads text only")
		}
	}
	te := time.Now()
	ids := e.tok.Encode(chat.ApplyTemplateEffort(e.mtype, msgs, opts.ReasoningEffort))
	p, maxTok := e.sampleParams(opts, onDelta)
	encoded := time.Since(te)

	mlxComputeMu.Lock()
	defer mlxComputeMu.Unlock()
	e.slotSessions()
	e.slotStats.add("encode", encoded)
	slot := e.pickSlot(ids)
	if slot < 0 {
		return nil, ErrNoSlot
	}
	t := time.Now()
	if fixedRows {
		qwen.Reclaim(e.bk, e.slots, slot) // the prefill uses the slot alone
		e.slotStats.add("reclaim", time.Since(t))
	} else {
		e.settleSlots() // the prefill uses the slot alone
		e.slotStats.add("settle", time.Since(t))
	}
	if e.slotStats != nil {
		e.slotStats.prefill += len(ids) - sharedPrefix(e.slots[slot].IDs, ids)
	}
	t = time.Now()
	var g *qwen.Gen
	if fixedRows {
		g = e.slots[slot].BeginLate(e.bk, ids, maxTok, p) // the last token in the next step
	} else {
		g = e.slots[slot].Begin(e.bk, ids, maxTok, p)
	}
	e.slotStats.add("prefill", time.Since(t))
	if fixedRows {
		t = time.Now()
		if !qwen.Relend(e.bk, e.slots, slot) {
			e.restackSlots()
			e.slotStats.add("relend failed", time.Since(t))
		} else {
			e.slotStats.add("relend", time.Since(t))
		}
	}
	if g == nil {
		return nil, nil
	}
	e.slotBusy[slot] = true
	e.slotClock++
	e.slotUsed[slot] = e.slotClock
	return &chatGen{slot: slot, g: g}, nil
}

// StepChats decodes one token of every reply not done, in one forward. Keep
// the replies' order from step to step: the model keeps the batch's stacked
// state while the rows stay the same.
func (e *Engine) StepChats(rs []Reply) {
	gens := make([]*qwen.Gen, 0, len(rs))
	set := make([]int, 0, len(rs))
	for _, r := range rs {
		c := r.(*chatGen)
		gens = append(gens, c.g)
		set = append(set, c.slot)
	}
	mlxComputeMu.Lock()
	defer mlxComputeMu.Unlock()
	if fixedRows {
		t0 := time.Now()
		if !slices.Equal(set, e.slotStep) && e.slotStats != nil {
			e.slotStats.changes++
		}
		e.slotStep = set
		if qwen.PoolsFull(e.slots) {
			e.restackSlots() // the next step makes a new pool
			e.slotStats.add("restack", time.Since(t0))
		}
		byslot := make([]*qwen.Gen, len(e.slots))
		for i, g := range gens {
			byslot[set[i]] = g
		}
		t := time.Now()
		qwen.StepRows(e.bk, e.slotDecoder, e.slots, byslot)
		e.slotStats.step(e.slotDecoder.Last, len(gens), time.Since(t))
		e.slotStats.flush()
		return
	}
	if !slices.Equal(set, e.slotStep) {
		// Another set of rows than last step: a row that left still holds
		// its part of the old stack.
		t := time.Now()
		e.settleSlots()
		e.slotStats.add("settle", time.Since(t))
		if e.slotStats != nil {
			e.slotStats.changes++
		}
		e.slotStep = set
	}
	t := time.Now()
	qwen.StepBatch(e.bk, e.slotDecoder, gens)
	e.slotStats.step(e.slotDecoder.Last, len(gens), time.Since(t))
	e.slotStats.flush()
}

// ResetSessions drops every session's conversation and the batch's
// recorded step, after an engine call failed partway (out of device
// memory): a session's caches may hold a prefill or a step that never
// ended. Each session reads its next prompt from the first token.
func (e *Engine) ResetSessions() {
	mlxComputeMu.Lock()
	defer mlxComputeMu.Unlock()
	if a, ok := e.bk.(interface{ AbortStep() }); ok {
		a.AbortStep()
	}
	if e.slotDecoder != nil {
		e.slotDecoder.Drop()
	}
	for _, s := range append([]*qwen.Session{e.session, e.utilSession}, e.slots...) {
		if s != nil {
			s.Reset(e.bk)
		}
	}
	for i := range e.slotBusy {
		e.slotBusy[i] = false
	}
	e.slotStep = nil
	if sw, ok := e.bk.(engine.Sweeper); ok {
		sw.Sweep()
	}
}

// FinishChat ends a reply, done or not, frees its slot and returns its text.
func (e *Engine) FinishChat(r Reply) string {
	c := r.(*chatGen)
	out := func() []int32 {
		mlxComputeMu.Lock()
		defer mlxComputeMu.Unlock()
		if !fixedRows {
			t := time.Now()
			e.settleSlots() // the batch loses a row
			e.slotStats.add("settle", time.Since(t))
		}
		e.slotBusy[c.slot] = false
		return c.g.Finish()
	}()
	text, _ := e.reply(out)
	return text
}
