package serve

import (
	"context"
	"sync"

	llm "github.com/guregodevo/janis"
)

// Batcher is a Generator that decodes several replies together
// (llm.Engine with JANIS_PARALLEL above 1 on a Qwen3.5 text model).
type Batcher interface {
	Slots() int
	BeginChat(msgs []llm.Message, opts llm.Options, onDelta func(string)) (llm.Reply, error)
	StepChats(rs []llm.Reply)
	FinishChat(r llm.Reply) string
}

// batched is the Batcher a request decodes on, nil when it decodes alone.
func (s *Server) batched(req chatRequest, msgs []llm.Message) Batcher {
	b, ok := s.current.(Batcher)
	if !ok || b.Slots() < 2 {
		return nil
	}
	for _, m := range msgs {
		if len(m.Images) > 0 {
			return nil
		}
	}
	return b
}

// pendingChat is a request waiting for, then holding, a slot.
type pendingChat struct {
	ctx     context.Context
	msgs    []llm.Message
	opts    llm.Options
	onDelta func(string)
	reply   string
	err     error
	done    chan struct{}
}

// queue holds the requests the batch loop has not started, and whether the
// loop runs. decoding is the number of replies the loop holds, read and
// written on the engine thread only.
type queue struct {
	mu       sync.Mutex
	waiting  []*pendingChat
	running  bool
	decoding int
}

// chatBatched is ChatStream on b: the request joins the batch loop's queue,
// starting the loop if it is not running, and returns its reply once the
// loop has decoded it. ctx done before the request got a slot is ctx.Err().
func (s *Server) chatBatched(ctx context.Context, b Batcher, msgs []llm.Message, opts llm.Options, onDelta func(string)) (string, error) {
	pc := &pendingChat{ctx: ctx, msgs: msgs, opts: opts, onDelta: onDelta, done: make(chan struct{})}
	s.batch.mu.Lock()
	s.batch.waiting = append(s.batch.waiting, pc)
	start := !s.batch.running
	s.batch.running = true
	s.batch.mu.Unlock()
	if start {
		go s.onEngine(func() { s.runBatch(b) })
	}
	<-pc.done
	return pc.reply, pc.err
}

// runBatch is the batch loop, on the engine thread. Each round it runs the
// engine calls queued meanwhile (a new request's set-up among them, so it
// can join), starts at most one waiting request, a prefill of its own, then
// decodes one token of every reply in one forward. The replies keep their
// order, so the model keeps their stacked state between rounds. It returns
// when nothing waits and nothing decodes.
func (s *Server) runBatch(b Batcher) {
	var replies []llm.Reply
	var owners []*pendingChat
	defer func() { s.batch.decoding = 0 }()
	finish := func(i int) {
		pc := owners[i]
		pc.reply = b.FinishChat(replies[i])
		if pc.ctx.Err() != nil {
			pc.err = pc.ctx.Err()
		}
		close(pc.done)
		replies = append(replies[:i], replies[i+1:]...)
		owners = append(owners[:i], owners[i+1:]...)
	}
	for {
		s.batch.decoding = len(replies)
		s.runQueued()
		if len(replies) < b.Slots() {
			if pc := s.nextWaiting(len(replies) == 0); pc != nil {
				s.start(b, pc, &replies, &owners)
			} else if len(replies) == 0 {
				return
			}
		}
		if len(replies) == 0 {
			continue
		}
		b.StepChats(replies)
		for i := len(replies) - 1; i >= 0; i-- {
			if replies[i].Done() {
				finish(i)
			}
		}
	}
}

// runQueued runs the engine calls waiting for the thread, without waiting
// for one.
func (s *Server) runQueued() {
	for {
		select {
		case f := <-s.work:
			f()
		default:
			return
		}
	}
}

// nextWaiting takes the first waiting request whose client is still there;
// the others are answered with their ctx's error. With none waiting and last
// set (nothing decodes), the loop is marked stopped under the same lock, so a
// request that arrives after starts a new loop.
func (s *Server) nextWaiting(last bool) *pendingChat {
	s.batch.mu.Lock()
	defer s.batch.mu.Unlock()
	for len(s.batch.waiting) > 0 {
		pc := s.batch.waiting[0]
		s.batch.waiting = s.batch.waiting[1:]
		if err := pc.ctx.Err(); err != nil {
			pc.err = err
			close(pc.done)
			continue
		}
		return pc
	}
	if last {
		s.batch.running = false
	}
	return nil
}

// start prefills pc on a slot and adds its reply at the end of the batch.
func (s *Server) start(b Batcher, pc *pendingChat, replies *[]llm.Reply, owners *[]*pendingChat) {
	r, err := b.BeginChat(pc.msgs, pc.opts, pc.onDelta)
	switch {
	case err != nil:
		pc.err = err
	case r == nil:
		pc.err = context.Canceled // cancelled during the prefill
		if e := pc.ctx.Err(); e != nil {
			pc.err = e
		}
	case r.Done():
		pc.reply = b.FinishChat(r)
	default:
		*replies = append(*replies, r)
		*owners = append(*owners, pc)
		return
	}
	close(pc.done)
}
