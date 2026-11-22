package serve

import (
	"bufio"
	"context"
	"encoding/json"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	llm "github.com/guregodevo/janis"
)

// fakeBatcher decodes "reply to <last message>" a word per step, with every
// running reply in one StepChats call; it records the largest batch.
type fakeBatcher struct {
	fakeEngine
	slots int

	mu      sync.Mutex
	begun   []string // the last message of each reply begun
	largest int
}

type fakeReply struct {
	words   []string
	at      int
	onDelta func(string)
	cancel  func() bool
	done    bool
}

func (r *fakeReply) Done() bool { return r.done }

func (f *fakeBatcher) Slots() int { return f.slots }

func (f *fakeBatcher) BeginChat(msgs []llm.Message, opts llm.Options, onDelta func(string)) (llm.Reply, error) {
	last := msgs[len(msgs)-1].Content
	f.mu.Lock()
	f.begun = append(f.begun, last)
	f.mu.Unlock()
	r := &fakeReply{words: strings.Fields("reply to " + last), onDelta: onDelta, cancel: opts.Cancel}
	r.emit()
	return r, nil
}

func (r *fakeReply) emit() {
	if r.onDelta != nil {
		if r.at > 0 {
			r.onDelta(" ")
		}
		r.onDelta(r.words[r.at])
	}
	r.at++
	r.done = r.at == len(r.words) || (r.cancel != nil && r.cancel())
}

func (f *fakeBatcher) StepChats(rs []llm.Reply) {
	f.mu.Lock()
	f.largest = max(f.largest, len(rs))
	f.mu.Unlock()
	time.Sleep(2 * time.Millisecond) // a forward
	for _, r := range rs {
		if fr := r.(*fakeReply); !fr.done {
			fr.emit()
		}
	}
}

func (f *fakeBatcher) FinishChat(r llm.Reply) string {
	fr := r.(*fakeReply)
	return strings.Join(fr.words[:fr.at], " ")
}

func newBatchServer(t *testing.T, eng *fakeBatcher) (*Server, string) {
	t.Helper()
	installFake(t, "Qwen3-8B-4bit", 40960)
	s := NewWithLoader(func(dir string) (Generator, error) { return eng, nil })
	s.logf = t.Logf
	srv := httptest.NewServer(s.Handler())
	t.Cleanup(srv.Close)
	return s, srv.URL
}

// deltaContent is the content delta of one SSE line, "" for any other line.
func deltaContent(line string) string {
	data, ok := strings.CutPrefix(line, "data: ")
	if !ok {
		return ""
	}
	var ev struct {
		Choices []struct {
			Delta struct {
				Content string `json:"content"`
			} `json:"delta"`
		} `json:"choices"`
	}
	if json.Unmarshal([]byte(data), &ev) != nil || len(ev.Choices) == 0 {
		return ""
	}
	return ev.Choices[0].Delta.Content
}

// A long reply: "reply to" and the question's words, a word a step.
func question(i int) string {
	return "question " + itoa(i) + strings.Repeat(" and more", 40)
}

// REQUESTS AT ONCE ARE DECODED IN ONE BATCH, each with its own reply.
//
// Four requests at once on four slots: every one gets the reply to its own
// question, and a step decoded all four.
func TestRequestsAtOnceAreDecodedInOneBatch(t *testing.T) {
	eng := &fakeBatcher{slots: 4}
	_, url := newBatchServer(t, eng)
	var wg sync.WaitGroup
	got := make([]string, 4)
	for i := range got {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, v := post(t, url+"/v1/chat/completions", `{"model":"Qwen3-8B-4bit","messages":[{"role":"user","content":"`+question(i)+`"}]}`)
			got[i] = v["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)["content"].(string)
		}()
	}
	wg.Wait()
	for i, g := range got {
		if want := "reply to " + question(i); g != want {
			t.Errorf("request %d: %q, want %q", i, g, want)
		}
	}
	if eng.largest != 4 {
		t.Errorf("largest batch %d, want 4", eng.largest)
	}
}

// MORE REQUESTS THAN SLOTS WAIT FOR ONE, and a streamed reply streams.
//
// Five streamed requests on two slots: no step holds more than two replies,
// and each stream carries its own reply and ends with [DONE].
func TestMoreRequestsThanSlotsWaitForOne(t *testing.T) {
	eng := &fakeBatcher{slots: 2}
	_, url := newBatchServer(t, eng)
	var wg sync.WaitGroup
	got := make([]string, 5)
	for i := range got {
		wg.Add(1)
		go func() {
			defer wg.Done()
			resp, _ := post(t, url+"/v1/chat/completions", `{"model":"Qwen3-8B-4bit","stream":true,"messages":[{"role":"user","content":"`+question(i)+`"}]}`)
			defer resp.Body.Close()
			var text strings.Builder
			sc := bufio.NewScanner(resp.Body)
			for sc.Scan() {
				line := sc.Text()
				if line == "data: [DONE]" {
					got[i] = text.String()
					return
				}
				if c := deltaContent(line); c != "" {
					text.WriteString(c)
				}
			}
		}()
	}
	wg.Wait()
	for i, g := range got {
		if want := "reply to " + question(i); g != want {
			t.Errorf("request %d: %q, want %q", i, g, want)
		}
	}
	if eng.largest > 2 {
		t.Errorf("largest batch %d, over the 2 slots", eng.largest)
	}
}

// THE LOOP GIVES THE ENGINE BACK WHEN IDLE, and a request after it starts
// a new loop.
func TestTheBatchLoopGivesTheEngineBackWhenIdle(t *testing.T) {
	eng := &fakeBatcher{slots: 2}
	s, url := newBatchServer(t, eng)
	post(t, url+"/v1/chat/completions", `{"model":"Qwen3-8B-4bit","messages":[{"role":"user","content":"one"}]}`)
	ran := make(chan struct{})
	go s.onEngine(func() { close(ran) })
	select {
	case <-ran:
	case <-time.After(5 * time.Second):
		t.Fatal("the engine thread is still held after the reply")
	}
	_, v := post(t, url+"/v1/chat/completions", `{"model":"Qwen3-8B-4bit","messages":[{"role":"user","content":"two"}]}`)
	if g := v["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)["content"]; g != "reply to two" {
		t.Errorf("second request: %q", g)
	}
}

// A REQUEST WHOSE CLIENT LEFT WHILE IT WAITED IS NEVER BEGUN.
func TestARequestWhoseClientLeftWhileItWaitedIsNeverBegun(t *testing.T) {
	eng := &fakeBatcher{slots: 2}
	s, _ := newBatchServer(t, eng)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	pc := &pendingChat{ctx: ctx, msgs: []llm.Message{{Role: "user", Content: "gone"}}, done: make(chan struct{})}
	s.batch.waiting = append(s.batch.waiting, pc)
	s.batch.running = true
	s.onEngine(func() { s.runBatch(eng) })
	<-pc.done
	if pc.err == nil {
		t.Error("no error for a request whose client left")
	}
	if len(eng.begun) != 0 {
		t.Errorf("begun %v", eng.begun)
	}
	if s.batch.running {
		t.Error("the loop is still marked running")
	}
}
