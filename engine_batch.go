package llm

import (
	"errors"
	"slices"

	"github.com/guregodevo/janis/chat"
	"github.com/guregodevo/janis/qwen"
)

// parallel is how many replies a server decodes together (JANIS_PARALLEL).
// 1, the default, decodes one at a time, as before.
var parallel = envInt("JANIS_PARALLEL", 1)

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
}

// settleSlots ends the batch's recorded step and takes back the rows the
// slots lent to it: before a slot is used alone or the batch changes.
func (e *Engine) settleSlots() {
	e.slotDecoder.Settle()
	qwen.Release(e.bk, e.slots)
	e.slotStep = nil
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
	ids := e.tok.Encode(chat.ApplyTemplateEffort(e.mtype, msgs, opts.ReasoningEffort))
	p, maxTok := e.sampleParams(opts, onDelta)

	mlxComputeMu.Lock()
	defer mlxComputeMu.Unlock()
	e.slotSessions()
	slot := e.pickSlot(ids)
	if slot < 0 {
		return nil, ErrNoSlot
	}
	e.settleSlots() // the prefill uses the slot alone
	g := e.slots[slot].Begin(e.bk, ids, maxTok, p)
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
	if !slices.Equal(set, e.slotStep) {
		// Another set of rows than last step: a row that left still holds
		// its part of the old stack.
		e.settleSlots()
		e.slotStep = set
	}
	qwen.StepBatch(e.bk, e.slotDecoder, gens)
	mlxComputeMu.Unlock()
}

// FinishChat ends a reply, done or not, frees its slot and returns its text.
func (e *Engine) FinishChat(r Reply) string {
	c := r.(*chatGen)
	mlxComputeMu.Lock()
	e.settleSlots() // the batch loses a row
	out := c.g.Finish()
	e.slotBusy[c.slot] = false
	mlxComputeMu.Unlock()
	text, _ := e.reply(out)
	return text
}
