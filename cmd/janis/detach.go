package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"time"
)

// detachServe is `serve --detach`, for a CI job or a script: the model pulled
// when it is missing, the server started in the background on the same flags,
// and the command returns once that server has answered one request with the
// model. Its first answer lays the weights out for the CPU's kernels (half a
// minute for an 8B on four vCPUs): paid here, not by the first client. The
// server's output goes to logPath; a server that dies prints it and fails.
func detachServe(host string, port int, model, logPath string) {
	// Another server already on the port would answer for this one.
	addr := net.JoinHostPort(host, fmt.Sprint(port))
	if l, err := net.Listen("tcp", addr); err != nil {
		fail(fmt.Errorf("janis serve: %s is taken: %w", addr, err))
	} else {
		l.Close()
	}
	if model != "" {
		pull([]string{model})
	}
	if logPath == "" {
		logPath = filepath.Join(os.TempDir(), fmt.Sprintf("janis-%d.log", port))
	}
	log, err := os.Create(logPath)
	if err != nil {
		fail(err)
	}
	self, err := os.Executable()
	if err != nil {
		fail(err)
	}
	args := []string{"serve", "--host", host, "--port", fmt.Sprint(port)}
	if model != "" {
		args = append(args, "--model", model)
	}
	cmd := exec.Command(self, args...)
	cmd.Stdout, cmd.Stderr = log, log
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	if err := cmd.Start(); err != nil {
		fail(err)
	}
	exited := make(chan struct{})
	go func() { _ = cmd.Wait(); close(exited) }()
	dead := func(what string) {
		out, _ := os.ReadFile(logPath)
		fmt.Fprintf(os.Stderr, "janis serve %s:\n%s", what, out)
		os.Exit(1)
	}

	if host == "0.0.0.0" || host == "::" || host == "" {
		host = "127.0.0.1"
	}
	base := "http://" + net.JoinHostPort(host, fmt.Sprint(port)) + "/v1"
	probe := &http.Client{Timeout: 2 * time.Second}
	ready := time.Now().Add(10 * time.Minute)
	for {
		select {
		case <-exited:
			dead("stopped before it answered")
		default:
		}
		if resp, err := probe.Get(base + "/models"); err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				break
			}
		}
		if time.Now().After(ready) {
			_ = cmd.Process.Kill()
			dead("did not answer in 10 minutes")
		}
		time.Sleep(time.Second)
	}
	if model != "" {
		body, _ := json.Marshal(map[string]any{"model": model, "max_tokens": 1,
			"messages": []map[string]string{{"role": "user", "content": "ok"}}})
		client := &http.Client{Timeout: 30 * time.Minute}
		resp, err := client.Post(base+"/chat/completions", "application/json", bytes.NewReader(body))
		if err != nil {
			dead("failed its first answer: " + err.Error())
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			dead("failed its first answer: " + resp.Status)
		}
	}
	pid := cmd.Process.Pid
	_ = cmd.Process.Release()
	served := model
	if served == "" {
		served = "the models a request names"
	}
	fmt.Printf("✓ janis serving %s on %s (pid %d, log %s)\n", served, base, pid, logPath)
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, err)
	os.Exit(1)
}
