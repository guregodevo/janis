package serve

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	llm "github.com/guregodevo/janis"
)

// failingBatcher is a fakeBatcher whose engine calls panic, as a ggml
// flush with no room on the device does, for the reply whose last
// message says so: "fail the prefill" in BeginChat, "fail the step" in
// StepChats. It counts ResetSessions.
type failingBatcher struct {
	fakeBatcher
	resets atomic.Int32
}

func (f *failingBatcher) BeginChat(msgs []llm.Message, opts llm.Options, onDelta func(string)) (llm.Reply, error) {
	if msgs[len(msgs)-1].Content == "fail the prefill" {
		panic("ggml: no room on the device for a batch of 32 results")
	}
	return f.fakeBatcher.BeginChat(msgs, opts, onDelta)
}

func (f *failingBatcher) StepChats(rs []llm.Reply) {
	for _, r := range rs {
		if strings.Join(r.(*fakeReply).words, " ") == "reply to fail the step" {
			panic("ggml: no room on the device for a batch of 2599 results")
		}
	}
	f.fakeBatcher.StepChats(rs)
}

func (f *failingBatcher) ResetSessions() { f.resets.Add(1) }

// decoding is whether a step has run.
func (f *failingBatcher) decoding() bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.largest > 0
}

func newFailingServer(t *testing.T, eng *failingBatcher) string {
	t.Helper()
	installFake(t, "Qwen3-8B-4bit", 40960)
	s := NewWithLoader(func(dir string) (Generator, error) { return eng, nil })
	s.logf = func(string, ...any) {}
	return serveFor(t, s)
}

// serveFor serves s; net/http's own log of a handler's panic is dropped.
func serveFor(t *testing.T, s *Server) string {
	t.Helper()
	srv := httptest.NewUnstartedServer(s.Handler())
	srv.Config.ErrorLog = log.New(io.Discard, "", 0)
	srv.Start()
	t.Cleanup(srv.Close)
	return srv.URL
}

func ask(t *testing.T, url, content string, stream bool) (int, string) {
	t.Helper()
	st := ""
	if stream {
		st = `"stream":true,`
	}
	resp, err := http.Post(url+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"Qwen3-8B-4bit",`+st+`"messages":[{"role":"user","content":"`+content+`"}]}`))
	if err != nil {
		return 0, err.Error()
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, string(b)
}

// AN ENGINE FAILURE FAILS ITS REQUESTS, AND THE SERVER GOES ON.
//
// A batched prefill that panics answers its request with a 500 and the
// reason; the sessions are reset once, and the next request is answered.
func TestAFailedPrefillAnswersAnErrorAndTheServerGoesOn(t *testing.T) {
	eng := &failingBatcher{fakeBatcher: fakeBatcher{slots: 4}}
	url := newFailingServer(t, eng)
	code, body := ask(t, url, "fail the prefill", false)
	if code != http.StatusInternalServerError || !strings.Contains(body, "no room on the device") {
		t.Fatalf("failed prefill: %d %s, want a 500 naming the failure", code, body)
	}
	if code, body = ask(t, url, "hello", false); code != http.StatusOK || !strings.Contains(body, "reply to hello") {
		t.Fatalf("next request: %d %s", code, body)
	}
	if n := eng.resets.Load(); n != 1 {
		t.Errorf("sessions reset %d times, want 1", n)
	}
}

// A step that panics fails every reply in it, streamed or not: a stream
// ends with an error event and no [DONE]. Requests after are answered.
func TestAFailedStepFailsEveryReplyInItAndTheServerGoesOn(t *testing.T) {
	eng := &failingBatcher{fakeBatcher: fakeBatcher{slots: 4}}
	url := newFailingServer(t, eng)
	var codes [2]int
	var bodies [2]string
	streamed := make(chan struct{})
	go func() {
		defer close(streamed)
		codes[1], bodies[1] = ask(t, url, "a long question"+strings.Repeat(" and more", 200), true)
	}()
	for !eng.decoding() {
		time.Sleep(time.Millisecond)
	}
	codes[0], bodies[0] = ask(t, url, "fail the step", false)
	<-streamed
	if codes[0] != http.StatusInternalServerError || !strings.Contains(bodies[0], "no room on the device") {
		t.Errorf("failing reply: %d %s", codes[0], bodies[0])
	}
	if strings.Contains(bodies[1], "[DONE]") {
		t.Errorf("the stream decoded with it ended as done: %s", bodies[1])
	}
	if !strings.Contains(bodies[1], "no room on the device") {
		t.Errorf("the stream decoded with it has no error event: %s", bodies[1])
	}
	if code, body := ask(t, url, "hello", false); code != http.StatusOK || !strings.Contains(body, "reply to hello") {
		t.Fatalf("next request: %d %s", code, body)
	}
	if eng.resets.Load() < 1 {
		t.Error("sessions not reset")
	}
}

// panickingEngine is a one-at-a-time engine whose generation panics once.
type panickingEngine struct {
	fakeEngine
	panics atomic.Int32
	resets atomic.Int32
}

func (p *panickingEngine) ChatStream(msgs []llm.Message, opts llm.Options, onDelta func(string)) string {
	if p.panics.Add(1) == 1 {
		panic("ggml: no room on the device")
	}
	return p.fakeEngine.ChatStream(msgs, opts, onDelta)
}

func (p *panickingEngine) ResetSessions() { p.resets.Add(1) }

// One at a time (no slots), a generation that panics ends its request; the
// sessions are reset and the engine thread answers the next request. The
// log has the panic's own stack, which names the engine's function: raised
// again on the request's goroutine, it showed only the server's.
func TestAFailedGenerationEndsItsRequestAndTheServerGoesOn(t *testing.T) {
	installFake(t, "Qwen3-8B-4bit", 40960)
	eng := &panickingEngine{fakeEngine: fakeEngine{reply: "fine"}}
	s := NewWithLoader(func(dir string) (Generator, error) { return eng, nil })
	var logged strings.Builder
	var mu sync.Mutex
	s.logf = func(f string, a ...any) { mu.Lock(); fmt.Fprintf(&logged, f, a...); mu.Unlock() }
	url := serveFor(t, s)
	defer func() {
		mu.Lock()
		defer mu.Unlock()
		if !strings.Contains(logged.String(), "engine panic") || !strings.Contains(logged.String(), "panickingEngine") {
			t.Errorf("the log does not name where the panic began:\n%s", logged.String())
		}
	}()
	if code, _ := ask(t, url, "first", false); code == http.StatusOK {
		t.Fatal("the failed generation answered 200")
	}
	if code, body := ask(t, url, "second", false); code != http.StatusOK || !strings.Contains(body, "fine") {
		t.Fatalf("next request: %d %s", code, body)
	}
	if n := eng.resets.Load(); n != 1 {
		t.Errorf("sessions reset %d times, want 1", n)
	}
}
