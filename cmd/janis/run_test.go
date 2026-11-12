package main

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
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
	if _, err := liveState(port); err == nil || !strings.Contains(err.Error(), "served by a program that is not janis") {
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
	started := make(chan struct{}, 3)
	mux := http.NewServeMux()
	mux.HandleFunc("/short", func(w http.ResponseWriter, r *http.Request) {
		started <- struct{}{}
		time.Sleep(100 * time.Millisecond)
		w.Write([]byte("done"))
	})
	mux.HandleFunc("/long", func(w http.ResponseWriter, r *http.Request) {
		started <- struct{}{}
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
	<-started
	<-started // both requests are in flight before the shutdown begins

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

// TestMain runs the test binary as a stand-in `janis serve` when
// JANIS_TEST_OLD_SERVE is set: a server an older janis started, with no
// state file.
func TestMain(m *testing.M) {
	if os.Getenv("JANIS_TEST_OLD_SERVE") == "1" {
		time.Sleep(time.Minute)
		os.Exit(0)
	}
	os.Exit(m.Run())
}

func TestAServeCommandLineIsRead(t *testing.T) {
	st, ok := parseServeArgs([]string{"--host", "0.0.0.0", "--port", "8091", "--model", "mlx-community/Qwen3.5-9B-MLX-4bit"})
	if !ok || st.Host != "0.0.0.0" || st.Port != 8091 || st.Model != "Qwen3.5-9B-MLX-4bit" {
		t.Fatalf("%+v %v", st, ok)
	}
	if st, ok := parseServeArgs(nil); !ok || st.Port != 8090 || st.Host != "127.0.0.1" {
		t.Fatalf("the defaults: %+v %v", st, ok)
	}
	if _, ok := parseServeArgs([]string{"--detach", "--port", "8090"}); ok {
		t.Fatal("serve --detach returns once its server answers: not the server")
	}
	if _, ok := parseServeArgs([]string{"--what"}); ok {
		t.Fatal("a flag it cannot read")
	}
}

// A `janis serve` an older janis started has no state file: status finds
// it on its port, restart reads its flags, stop ends it.
func TestAnOlderJanisOnThePortIsFoundAndStopped(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	l, _ := net.Listen("tcp", "127.0.0.1:0")
	port := l.Addr().(*net.TCPAddr).Port
	l.Close()
	old := exec.Command(os.Args[0])
	old.Args = []string{"janis", "serve", "--host", "0.0.0.0", "--port", fmt.Sprint(port), "--model", "Qwen3.5-9B-MLX-4bit"}
	old.Env = append(os.Environ(), "JANIS_TEST_OLD_SERVE=1")
	if err := old.Start(); err != nil {
		t.Fatal(err)
	}
	exited := make(chan struct{})
	go func() { _ = old.Wait(); close(exited) }()
	t.Cleanup(func() { _ = old.Process.Kill() })

	st, err := liveState(port)
	if err != nil || st.PID != old.Process.Pid || st.Model != "Qwen3.5-9B-MLX-4bit" || st.Host != "0.0.0.0" || !st.Started.IsZero() {
		t.Fatalf("found: %+v %v", st, err)
	}
	f := newServeFlags("restart")
	_ = f.fs.Parse(nil)
	if h, m, _ := restartFlags(st, f); h != "0.0.0.0" || m != "Qwen3.5-9B-MLX-4bit" {
		t.Fatalf("restart takes its flags: %s %s", h, m)
	}
	if _, ok := serveProcess(port + 1); ok {
		t.Fatal("another port")
	}
	stopServer(st)
	select {
	case <-exited:
	case <-time.After(5 * time.Second):
		t.Fatal("the older janis still runs")
	}
}
