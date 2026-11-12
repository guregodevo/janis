package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/guregodevo/janis/serve"
)

// A server's lifecycle (janis#18): every `janis serve`, in the foreground or
// detached, writes ~/.janis/run/PORT.json with its pid and flags, and
// removes it when it stops. start, stop, restart and status read it.

const (
	defaultPort = 8090
	defaultHost = "127.0.0.1"
	// logEnv carries --log from `serve --detach` to the server it starts,
	// so the state file names the log.
	logEnv = "JANIS_LOG"
	// stopWait is how long stop waits after SIGTERM before SIGKILL. The
	// server's own shutdown ends within it (shutdownGrace + cancelGrace +
	// engineGrace).
	stopWait   = 10 * time.Second
	healthWait = 2 * time.Second
)

// shutdownGrace is how long in-flight requests get to finish on SIGTERM;
// cancelGrace how long they get to stop once cancelled; engineGrace how long
// the engine gets to close. Together they end within stopWait. Vars: a test
// shortens them.
var (
	shutdownGrace = 5 * time.Second
	cancelGrace   = 2 * time.Second
	engineGrace   = 2 * time.Second
)

// runState is one server's state file.
type runState struct {
	PID     int       `json:"pid"`
	Host    string    `json:"host"`
	Port    int       `json:"port"`
	Model   string    `json:"model,omitempty"`
	Log     string    `json:"log,omitempty"`
	Started time.Time `json:"started"`
}

func runDir() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".janis", "run")
}

func statePath(port int) string { return filepath.Join(runDir(), fmt.Sprintf("%d.json", port)) }

func readState(port int) (runState, error) {
	var st runState
	b, err := os.ReadFile(statePath(port))
	if err != nil {
		return st, err
	}
	if err := json.Unmarshal(b, &st); err != nil {
		return st, fmt.Errorf("%s: %w", statePath(port), err)
	}
	return st, nil
}

func (st runState) write() error {
	if err := os.MkdirAll(runDir(), 0o755); err != nil {
		return err
	}
	b, _ := json.MarshalIndent(st, "", "  ")
	tmp := statePath(st.Port) + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, statePath(st.Port))
}

// removeState removes port's state file when it is pid's: a server that
// stops late never removes the file of the one that replaced it.
func removeState(port, pid int) {
	if st, err := readState(port); err == nil && st.PID == pid {
		_ = os.Remove(statePath(port))
	}
}

// running reports whether pid is a live janis process. A pid the system
// gave to another program since is not.
func running(pid int) bool {
	if pid <= 0 || syscall.Kill(pid, 0) != nil {
		return false
	}
	out, err := exec.Command("ps", "-o", "command=", "-p", fmt.Sprint(pid)).Output()
	return err == nil && strings.Contains(string(out), "janis")
}

// serveFlags are the flags serve, start and restart share.
type serveFlags struct {
	fs               *flag.FlagSet
	port             *int
	host, model, log *string
}

func newServeFlags(name string) serveFlags { return newServeFlagSet(name, flag.ExitOnError) }

func newServeFlagSet(name string, handling flag.ErrorHandling) serveFlags {
	fs := flag.NewFlagSet(name, handling)
	return serveFlags{
		fs:    fs,
		port:  fs.Int("port", defaultPort, "the port to listen on"),
		host:  fs.String("host", defaultHost, "the address to bind (0.0.0.0 for the network)"),
		model: fs.String("model", "", "the installed model to load now (default: the first a request names; in the background, the model that fits this machine)"),
		log:   fs.String("log", "", "in the background, the server's output (default: janis-PORT.log in the temp directory)"),
	}
}

func (f serveFlags) modelName() string { return strings.TrimPrefix(*f.model, "mlx-community/") }

// serveForeground is `janis serve`: it serves until SIGTERM, Ctrl-C or its
// terminal closing (SIGHUP), then shuts down gracefully (shutdown).
func serveForeground(f serveFlags) {
	addr := net.JoinHostPort(*f.host, fmt.Sprint(*f.port))
	l, err := net.Listen("tcp", addr)
	if err != nil {
		fail(fmt.Errorf("janis serve: %s is taken: %w", addr, err))
	}
	srv, err := serve.New(*f.model)
	if err != nil {
		l.Close()
		fail(err)
	}
	st := runState{PID: os.Getpid(), Host: *f.host, Port: *f.port, Model: f.modelName(), Log: os.Getenv(logEnv), Started: time.Now().UTC()}
	if err := st.write(); err != nil {
		fmt.Fprintf(os.Stderr, "janis serve: no state file, janis stop will not find this server: %v\n", err)
	}

	signals, stopSignals := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM, syscall.SIGHUP)
	defer stopSignals()
	inFlight, cancelInFlight := context.WithCancel(context.Background())
	hs := &http.Server{Handler: srv.Handler(), BaseContext: func(net.Listener) context.Context { return inFlight }}
	served := make(chan error, 1)
	go func() { served <- hs.Serve(l) }()
	fmt.Printf("janis serving on http://%s/v1 — GET /v1/models, POST /v1/chat/completions, POST /v1/systemone (pid %d)\n", addr, st.PID)

	select {
	case err := <-served:
		srv.Close()
		removeState(st.Port, st.PID)
		fail(err)
	case <-signals.Done():
	}
	stopSignals() // a second Ctrl-C ends the process at once
	fmt.Printf("janis stopping: in-flight requests have %s to finish\n", shutdownGrace)
	shutdown(hs, srv, cancelInFlight)
	removeState(st.Port, st.PID)
	fmt.Println("janis stopped")
}

// shutdown stops a server gracefully: no new connection is taken, and the
// requests in flight get shutdownGrace to finish. Those still running are
// then cancelled: a decision stops before its next question, a generation
// at its next token, a request waiting for the engine leaves the queue
// (janis#19). The engine is closed last, its memory freed.
func shutdown(hs *http.Server, srv *serve.Server, cancelInFlight context.CancelFunc) {
	ctx, cancel := context.WithTimeout(context.Background(), shutdownGrace)
	err := hs.Shutdown(ctx)
	cancel()
	if err != nil {
		fmt.Println("janis stopping: cancelling the requests still running")
		cancelInFlight()
		ctx, cancel = context.WithTimeout(context.Background(), cancelGrace)
		if hs.Shutdown(ctx) != nil {
			_ = hs.Close()
		}
		cancel()
	}
	cancelInFlight()
	closed := make(chan struct{})
	go func() { srv.Close(); close(closed) }()
	select {
	case <-closed:
	case <-time.After(engineGrace):
		fmt.Println("janis stopping: the engine did not close in time; exiting")
	}
}

// serveProcess finds a `janis serve` on port by its command line, for a
// server with no state file: one started by janis v0.1.15 or older, before
// the file. Its host and model are read from its flags. ps, not lsof: Cloud
// Shell and slim Linux images have no lsof.
func serveProcess(port int) (runState, bool) {
	out, err := exec.Command("ps", "-e", "-o", "pid=", "-o", "args=").Output()
	if err != nil {
		return runState{}, false
	}
	for _, line := range strings.Split(string(out), "\n") {
		f := strings.Fields(line)
		if len(f) < 3 || filepath.Base(f[1]) != "janis" || f[2] != "serve" {
			continue
		}
		pid, err := strconv.Atoi(f[0])
		if err != nil || pid == os.Getpid() {
			continue
		}
		st, ok := parseServeArgs(f[3:])
		if !ok || st.Port != port {
			continue
		}
		st.PID = pid
		return st, true
	}
	return runState{}, false
}

// parseServeArgs reads a `janis serve` command line's flags. ok is false for
// `serve --detach`, which returns once its server answers, and for flags it
// cannot read.
func parseServeArgs(args []string) (runState, bool) {
	f := newServeFlagSet("serve", flag.ContinueOnError)
	f.fs.SetOutput(io.Discard)
	detach := f.fs.Bool("detach", false, "")
	if f.fs.Parse(args) != nil || *detach {
		return runState{}, false
	}
	return runState{Host: *f.host, Port: *f.port, Model: f.modelName(), Log: *f.log}, true
}

// startCmd is `janis start`: `janis serve --detach`.
func startCmd(args []string) {
	f := newServeFlags("start")
	_ = f.fs.Parse(args)
	if st, err := liveState(*f.port); err == nil {
		fail(fmt.Errorf("janis is serving on port %d already (pid %d): janis restart replaces it", *f.port, st.PID))
	}
	detachServe(*f.host, *f.port, f.modelName(), *f.log)
}

func portFlag(name string, args []string) int {
	fs := flag.NewFlagSet(name, flag.ExitOnError)
	port := fs.Int("port", defaultPort, "the server's port")
	_ = fs.Parse(args)
	return *port
}

// stopCmd is `janis stop`: SIGTERM, up to stopWait for the graceful
// shutdown, then SIGKILL.
func stopCmd(args []string) {
	port := portFlag("stop", args)
	st, err := liveState(port)
	if err != nil {
		fail(err)
	}
	stopServer(st)
}

// liveState is port's state when its server runs: its state file, else a
// `janis serve` found on the port (serveProcess). An error in words
// otherwise. A state file left by a server that died is removed.
func liveState(port int) (runState, error) {
	st, err := readState(port)
	if errors.Is(err, os.ErrNotExist) {
		if old, ok := serveProcess(port); ok {
			return old, nil
		}
		return st, notStarted(port)
	}
	if err != nil {
		return st, err
	}
	if !running(st.PID) {
		_ = os.Remove(statePath(port))
		if old, ok := serveProcess(port); ok {
			return old, nil
		}
		return st, fmt.Errorf("janis on port %d (pid %d) had stopped already, without janis stop", port, st.PID)
	}
	return st, nil
}

// notStarted says no janis serves port, and whether another program does.
func notStarted(port int) error {
	if !portServed(port) {
		return fmt.Errorf("no janis server on port %d", port)
	}
	return fmt.Errorf("port %d is served by a program that is not janis", port)
}

// portServed reports whether something answers on port on this machine.
func portServed(port int) bool {
	c, err := net.DialTimeout("tcp", net.JoinHostPort(defaultHost, fmt.Sprint(port)), time.Second)
	if err != nil {
		return false
	}
	c.Close()
	return true
}

func stopServer(st runState) {
	if err := syscall.Kill(st.PID, syscall.SIGTERM); err != nil {
		fail(fmt.Errorf("janis on port %d (pid %d): %w", st.Port, st.PID, err))
	}
	deadline := time.Now().Add(stopWait)
	for running(st.PID) && time.Now().Before(deadline) {
		time.Sleep(100 * time.Millisecond)
	}
	if running(st.PID) {
		_ = syscall.Kill(st.PID, syscall.SIGKILL)
		for i := 0; i < 50 && running(st.PID); i++ {
			time.Sleep(100 * time.Millisecond)
		}
		removeState(st.Port, st.PID)
		fmt.Printf("janis on port %d did not stop in %s: killed (pid %d)\n", st.Port, stopWait, st.PID)
		return
	}
	removeState(st.Port, st.PID)
	fmt.Printf("✓ janis on port %d stopped (pid %d)\n", st.Port, st.PID)
}

// restartCmd is `janis restart`: stop, then start on the saved flags, or
// on those of a `janis serve` found on the port with no state file (an older
// janis); a flag on the command line replaces them. With no janis on the
// port it starts one, as `janis start` does (systemctl's restart).
func restartCmd(args []string) {
	f := newServeFlags("restart")
	_ = f.fs.Parse(args)
	st, err := readState(*f.port)
	if errors.Is(err, os.ErrNotExist) {
		old, ok := serveProcess(*f.port)
		switch {
		case ok:
			st = old
		case portServed(*f.port):
			fail(notStarted(*f.port))
		default:
			st = runState{Host: *f.host, Port: *f.port}
		}
		err = nil
	}
	if err != nil {
		fail(err)
	}
	host, model, logPath := restartFlags(st, f)
	if running(st.PID) {
		stopServer(st)
	} else {
		_ = os.Remove(statePath(*f.port))
	}
	detachServe(host, *f.port, model, logPath)
}

// restartFlags are the saved server's host, model and log, each replaced by
// the flag given on the command line.
func restartFlags(st runState, f serveFlags) (host, model, logPath string) {
	host, model, logPath = st.Host, st.Model, st.Log
	f.fs.Visit(func(fl *flag.Flag) {
		switch fl.Name {
		case "host":
			host = *f.host
		case "model":
			model = f.modelName()
		case "log":
			logPath = *f.log
		}
	})
	return host, model, logPath
}

// statusCmd is `janis status`: the pid, the model, the uptime, and whether
// GET /health answers within healthWait. Exits 1 unless it does.
func statusCmd(args []string) {
	port := portFlag("status", args)
	st, err := liveState(port)
	if err != nil {
		fail(err)
	}
	model := st.Model
	if model == "" {
		model = "the first a request names"
	}
	base := "http://" + net.JoinHostPort(dialHost(st.Host), fmt.Sprint(port))
	up := "unknown: started without a state file (janis v0.1.15 or older)"
	if !st.Started.IsZero() {
		up = uptime(time.Since(st.Started))
	}
	fmt.Printf("janis on %s/v1\n  pid     %d\n  model   %s\n  up      %s\n", base, st.PID, model, up)
	if st.Log != "" {
		fmt.Printf("  log     %s\n", st.Log)
	}
	t0 := time.Now()
	resp, err := (&http.Client{Timeout: healthWait}).Get(base + "/health")
	if err != nil {
		fmt.Printf("  health  no answer in %s\n", healthWait)
		os.Exit(1)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		fmt.Printf("  health  %s\n", resp.Status)
		os.Exit(1)
	}
	fmt.Printf("  health  ok (%d ms)\n", time.Since(t0).Milliseconds())
}

// dialHost is the address to reach a server bound to host from this machine.
func dialHost(host string) string {
	if host == "0.0.0.0" || host == "::" || host == "" {
		return defaultHost
	}
	return host
}

func uptime(d time.Duration) string {
	d = d.Round(time.Second)
	switch {
	case d >= 24*time.Hour:
		return fmt.Sprintf("%dd %dh", int(d.Hours())/24, int(d.Hours())%24)
	case d >= time.Hour:
		return fmt.Sprintf("%dh %dm", int(d.Hours()), int(d.Minutes())%60)
	default:
		return d.String()
	}
}
