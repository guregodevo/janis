package main

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	llm "github.com/guregodevo/janis"
	"github.com/guregodevo/janis/serve"
)

// A server's state file is its own: a server that stops late never removes
// the file of the one that replaced it.
func TestTheStateFileIsRemovedOnlyByItsServer(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	if _, err := readState(18090); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("no file yet: %v", err)
	}
	st := runState{PID: 41, Host: "127.0.0.1", Port: 18090, Model: "Qwen3.5-9B-MLX-4bit", Started: time.Now().UTC()}
	if err := st.write(); err != nil {
		t.Fatal(err)
	}
	if got, err := readState(18090); err != nil || got.PID != 41 || got.Model != st.Model {
		t.Fatalf("read back: %+v %v", got, err)
	}
	removeState(18090, 42)
	if _, err := readState(18090); err != nil {
		t.Fatalf("another pid removed it: %v", err)
	}
	removeState(18090, 41)
	if _, err := readState(18090); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("its own pid left it: %v", err)
	}
}

// restart keeps the saved flags; one given on the command line replaces it.
func TestRestartTakesTheSavedFlagsUnlessGiven(t *testing.T) {
	st := runState{Host: "0.0.0.0", Port: 8090, Model: "Qwen3.5-9B-MLX-4bit", Log: "/tmp/j.log"}
	f := newServeFlags("restart")
	_ = f.fs.Parse(nil)
	if h, m, l := restartFlags(st, f); h != "0.0.0.0" || m != "Qwen3.5-9B-MLX-4bit" || l != "/tmp/j.log" {
		t.Fatalf("saved: %s %s %s", h, m, l)
	}
	f = newServeFlags("restart")
	_ = f.fs.Parse([]string{"--model", "mlx-community/Qwen3-8B-4bit"})
	if h, m, _ := restartFlags(st, f); h != "0.0.0.0" || m != "Qwen3-8B-4bit" {
		t.Fatalf("--model given: %s %s", h, m)
	}
}

// A pid that is not a live janis is not running: the test binary is
// janis.test, a pid nothing has is nothing.
func TestRunningIsALiveJanis(t *testing.T) {
	if !running(os.Getpid()) {
		t.Fatal("this janis.test process")
	}
	if running(0) || running(1<<22) {
		t.Fatal("no process")
	}
}

func TestStopOnAnEmptyPortSaysSo(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	l, _ := net.Listen("tcp", "127.0.0.1:0")
	port := l.Addr().(*net.TCPAddr).Port
	if _, err := liveState(port); err == nil || !strings.Contains(err.Error(), "not by a janis this version started") {
		t.Fatalf("a port something else serves: %v", err)
	}
	l.Close()
	if _, err := liveState(port); err == nil || err.Error() != fmt.Sprintf("no janis server on port %d", port) {
		t.Fatalf("an empty port: %v", err)
	}
	if err := (runState{PID: 1 << 22, Port: port}).write(); err != nil {
		t.Fatal(err)
	}
	if _, err := liveState(port); err == nil || !strings.Contains(err.Error(), "had stopped already") {
		t.Fatalf("a dead server's file: %v", err)
	}
	if _, err := readState(port); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("a dead server's file is removed")
	}
}

type idleEngine struct{}

func (idleEngine) ModelType() string { return "qwen3" }
func (idleEngine) ChatStream([]llm.Message, llm.Options, func(string)) string {
	return ""
}
func (idleEngine) DecideContext(context.Context, string, []llm.Question) ([]llm.Decision, error) {
	return nil, nil
}
func (idleEngine) NumTokens(string) int { return 0 }
func (idleEngine) Close()               {}

// Shutdown: a request that finishes within the grace is answered; one still
// running after it is cancelled; the engine is closed; no new connection is
// taken.
func TestShutdownLetsRequestsFinishThenCancelsTheRest(t *testing.T) {
	saved := [3]time.Duration{shutdownGrace, cancelGrace, engineGrace}
	t.Cleanup(func() { shutdownGrace, cancelGrace, engineGrace = saved[0], saved[1], saved[2] })
	shutdownGrace, cancelGrace, engineGrace = 300*time.Millisecond, 300*time.Millisecond, 300*time.Millisecond

	srv := serve.NewWithLoader(func(string) (serve.Generator, error) { return idleEngine{}, nil })
	inFlight, cancelInFlight := context.WithCancel(context.Background())
	var cancelled atomic.Bool
	mux := http.NewServeMux()
	mux.HandleFunc("/short", func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(100 * time.Millisecond)
		w.Write([]byte("done"))
	})
	mux.HandleFunc("/long", func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
			cancelled.Store(true)
		case <-time.After(10 * time.Second):
		}
	})
	l, _ := net.Listen("tcp", "127.0.0.1:0")
	hs := &http.Server{Handler: mux, BaseContext: func(net.Listener) context.Context { return inFlight }}
	go hs.Serve(l)
	base := "http://" + l.Addr().String()

	short := make(chan string, 1)
	go func() {
		resp, err := http.Get(base + "/short")
		if err != nil {
			short <- err.Error()
			return
		}
		defer resp.Body.Close()
		b := make([]byte, 4)
		n, _ := resp.Body.Read(b)
		short <- string(b[:n])
	}()
	go http.Get(base + "/long")
	time.Sleep(50 * time.Millisecond)

	t0 := time.Now()
	shutdown(hs, srv, cancelInFlight)
	took := time.Since(t0)

	if got := <-short; got != "done" {
		t.Fatalf("the short request: %q", got)
	}
	if !cancelled.Load() {
		t.Fatal("the long request was not cancelled")
	}
	if took > shutdownGrace+cancelGrace+engineGrace {
		t.Fatalf("shutdown took %s", took)
	}
	if _, err := http.Get(base + "/short"); err == nil {
		t.Fatal("a new request was taken after shutdown")
	}
}
