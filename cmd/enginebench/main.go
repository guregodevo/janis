// Command enginebench compares memdoor's in-process engine against Ollama (and
// optionally a llama.cpp server) on the SAME MoE coding model, under a GROWING-
// CONTEXT agentic tool-loop — the workload that actually decides the wedge:
// decode speed as an agent's context grows turn over turn, not a clean one-shot
// prompt. (A clean prompt flatters every backend equally; agentic loops are where
// an expert-offload engine either holds up or falls apart.)
//
// Built for the 64GB target box running Qwen3-Coder-30B-A3B (qwen3-moe, which the
// memdoor engine supports natively). Smoke-test it on 16GB with a small model to
// prove the harness works — but the DECISION is measured on the 64GB machine.
//
// Usage:
//
//	enginebench [flags]
//	  -model    mlx-community model name substring memdoor loads (default Qwen3-Coder-30B-A3B-4bit)
//	  -turns    agentic turns; context grows each turn (default 6)
//	  -ctxchunk synthetic "tool result" tokens appended per turn (default 2000)
//	  -gen      tokens generated per turn (default 256)
//	  -ollama        Ollama base URL, "" to skip (default http://localhost:11434)
//	  -ollama-model  Ollama model tag (default qwen3-coder:30b)
//	  -llamacpp      llama.cpp server base URL, "" to skip
//	  -out      dir to write each backend's last-turn output for quality diff (default /tmp)
//
// What it reports per turn, per backend: input context tokens, prefill time
// (time-to-first-token), and DECODE tok/s — so you see the degradation CURVE as
// context grows, which is the real question (does memdoor beat Ollama under load,
// or only on a cold prompt?).
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"memdoor/llm"
)

type turnStat struct {
	backend    string
	turn       int
	ctxTok     int
	prefillMs  float64
	decodeTokS float64
	outTok     int
	totalMs    float64
}

func main() {
	model := flag.String("model", "Qwen3-Coder-30B-A3B-4bit", "mlx-community model name substring for memdoor")
	turns := flag.Int("turns", 6, "agentic turns (context grows each turn)")
	ctxChunk := flag.Int("ctxchunk", 2000, "synthetic tool-result tokens appended per turn")
	gen := flag.Int("gen", 256, "tokens generated per turn")
	ollamaURL := flag.String("ollama", "http://localhost:11434", "Ollama base URL (\"\" to skip)")
	ollamaModel := flag.String("ollama-model", "qwen3-coder:30b", "Ollama model tag")
	llamacppURL := flag.String("llamacpp", "", "llama.cpp server base URL (\"\" to skip)")
	outDir := flag.String("out", "/tmp", "dir for each backend's last-turn output")
	flag.Parse()

	// The agentic loop: a fixed coding task, then each turn appends a chunk of
	// synthetic "file contents" (a tool result) and asks the agent to continue —
	// exactly how a real coding agent's context balloons.
	system := "You are a coding agent. You are given file contents from tool calls. " +
		"Analyze them, find bugs, and propose concrete fixes. Be specific and cite the file."
	seed := "Task: review the service for a goroutine leak and a missing error check, then propose the fix."
	chunk := makeCodeChunk(*ctxChunk)

	var all []turnStat

	// ---- memdoor (in-process) ----
	fmt.Fprintf(os.Stderr, "== memdoor engine (%s) ==\n", *model)
	if stats, err := runMemdoor(*model, system, seed, chunk, *turns, *gen, *outDir); err != nil {
		fmt.Fprintf(os.Stderr, "memdoor: %v\n", err)
	} else {
		all = append(all, stats...)
	}

	// ---- Ollama ----
	if *ollamaURL != "" {
		fmt.Fprintf(os.Stderr, "== ollama (%s @ %s) ==\n", *ollamaModel, *ollamaURL)
		if stats, err := runOllama(*ollamaURL, *ollamaModel, system, seed, chunk, *turns, *gen, *outDir); err != nil {
			fmt.Fprintf(os.Stderr, "ollama skipped: %v\n", err)
		} else {
			all = append(all, stats...)
		}
	}

	// ---- llama.cpp (OpenAI-compatible) ----
	if *llamacppURL != "" {
		fmt.Fprintf(os.Stderr, "== llama.cpp (@ %s) ==\n", *llamacppURL)
		if stats, err := runLlamaCpp(*llamacppURL, system, seed, chunk, *turns, *gen, *outDir); err != nil {
			fmt.Fprintf(os.Stderr, "llama.cpp skipped: %v\n", err)
		} else {
			all = append(all, stats...)
		}
	}

	printTable(all)
}

// makeCodeChunk returns ~nTok tokens of plausible Go source (repeated), the
// per-turn "tool result" that grows the context. Same text for every backend, so
// the comparison is fair (tokenizers differ slightly; the bytes don't).
func makeCodeChunk(nTok int) string {
	const unit = `
func (s *Service) handle(ctx context.Context, req *Request) (*Response, error) {
	ch := make(chan result) // BUG: never closed, goroutine can leak
	go func() {
		out := s.process(req)        // BUG: error ignored
		ch <- result{value: out}
	}()
	select {
	case r := <-ch:
		return &Response{Value: r.value}, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}
`
	// ~4 chars per token is the usual rough ratio; size by chars.
	target := nTok * 4
	var b strings.Builder
	for b.Len() < target {
		b.WriteString(unit)
	}
	return b.String()
}

// buildTurns returns the message slice for turn i (0-based): system + seed, then
// i tool-result chunks appended, then "continue".
func buildTurns(system, seed, chunk string, turn int) []llm.Message {
	msgs := []llm.Message{
		{Role: "system", Content: system},
		{Role: "user", Content: seed},
	}
	for k := 0; k <= turn; k++ {
		msgs = append(msgs, llm.Message{
			Role:    "user",
			Content: fmt.Sprintf("[tool result: contents of file_%d.go]\n%s\nContinue your analysis using this file.", k, chunk),
		})
	}
	return msgs
}

func runMemdoor(modelSub, system, seed, chunk string, turns, gen int, outDir string) ([]turnStat, error) {
	home, _ := os.UserHomeDir()
	dirs, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--mlx-community--"+modelSub+"*/snapshots/*"))
	if len(dirs) == 0 {
		return nil, fmt.Errorf("model %q not cached (memdoor llm auto --model %s)", modelSub, modelSub)
	}
	eng, err := llm.Open(dirs[0])
	if err != nil {
		return nil, err
	}
	defer eng.Close()

	var stats []turnStat
	var lastOut string
	var prevIOBytes int64
	var prevIODur time.Duration
	for t := 0; t < turns; t++ {
		msgs := buildTurns(system, seed, chunk, t)
		ctxTok := 0
		for _, m := range msgs {
			ctxTok += eng.NumTokens(m.Content)
		}
		start := time.Now()
		var tFirst time.Time
		firstSet := false
		reply := eng.ChatStream(msgs, llm.Options{Temp: 0, MaxTokens: gen}, func(string) {
			if !firstSet {
				tFirst = time.Now()
				firstSet = true
			}
		})
		end := time.Now()
		outTok := eng.NumTokens(reply)
		st := turnStat{backend: "memdoor", turn: t, ctxTok: ctxTok, outTok: outTok, totalMs: end.Sub(start).Seconds() * 1000}
		if firstSet {
			st.prefillMs = tFirst.Sub(start).Seconds() * 1000
			if d := end.Sub(tFirst).Seconds(); d > 0 && outTok > 1 {
				st.decodeTokS = float64(outTok-1) / d
			}
		}
		stats = append(stats, st)
		lastOut = reply
		if ioB, ioD, ok := eng.ExpertIOStats(); ok {
			dB, dD := ioB-prevIOBytes, ioD-prevIODur
			prevIOBytes, prevIODur = ioB, ioD
			fmt.Fprintf(os.Stderr, "  memdoor turn %d: ctx=%d prefill=%.0fms decode=%.1f tok/s  expert-io=%dMB/%.0fms\n",
				t, ctxTok, st.prefillMs, st.decodeTokS, dB>>20, dD.Seconds()*1000)
		} else {
			fmt.Fprintf(os.Stderr, "  memdoor turn %d: ctx=%d prefill=%.0fms decode=%.1f tok/s\n", t, ctxTok, st.prefillMs, st.decodeTokS)
		}
	}
	_ = os.WriteFile(filepath.Join(outDir, "enginebench-memdoor.txt"), []byte(lastOut), 0o644)
	return stats, nil
}

// ----- Ollama (/api/chat, non-streaming → returns prompt/eval timings) -----

type ollamaMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ollamaReq struct {
	Model    string         `json:"model"`
	Messages []ollamaMsg    `json:"messages"`
	Stream   bool           `json:"stream"`
	Options  map[string]any `json:"options"`
}

type ollamaResp struct {
	Message            ollamaMsg `json:"message"`
	PromptEvalCount    int       `json:"prompt_eval_count"`
	PromptEvalDuration int64     `json:"prompt_eval_duration"` // ns
	EvalCount          int       `json:"eval_count"`
	EvalDuration       int64     `json:"eval_duration"` // ns
	Error              string    `json:"error"`
}

func runOllama(baseURL, model, system, seed, chunk string, turns, gen int, outDir string) ([]turnStat, error) {
	client := &http.Client{Timeout: 30 * time.Minute}
	// reachability probe
	if _, err := client.Get(strings.TrimRight(baseURL, "/") + "/api/tags"); err != nil {
		return nil, fmt.Errorf("not reachable at %s: %w", baseURL, err)
	}
	var stats []turnStat
	var lastOut string
	for t := 0; t < turns; t++ {
		gmsgs := buildTurns(system, seed, chunk, t)
		oms := make([]ollamaMsg, len(gmsgs))
		for i, m := range gmsgs {
			oms[i] = ollamaMsg{Role: m.Role, Content: m.Content}
		}
		body, _ := json.Marshal(ollamaReq{
			Model: model, Messages: oms, Stream: false,
			Options: map[string]any{"temperature": 0, "num_predict": gen},
		})
		resp, err := client.Post(strings.TrimRight(baseURL, "/")+"/api/chat", "application/json", bytes.NewReader(body))
		if err != nil {
			return stats, err
		}
		var r ollamaResp
		_ = json.NewDecoder(resp.Body).Decode(&r)
		resp.Body.Close()
		if r.Error != "" {
			return stats, fmt.Errorf("%s (pull it: ollama pull %s)", r.Error, model)
		}
		st := turnStat{backend: "ollama", turn: t, ctxTok: r.PromptEvalCount, outTok: r.EvalCount}
		st.prefillMs = float64(r.PromptEvalDuration) / 1e6
		if r.EvalDuration > 0 {
			st.decodeTokS = float64(r.EvalCount) / (float64(r.EvalDuration) / 1e9)
		}
		st.totalMs = float64(r.PromptEvalDuration+r.EvalDuration) / 1e6
		stats = append(stats, st)
		lastOut = r.Message.Content
		fmt.Fprintf(os.Stderr, "  ollama turn %d: ctx=%d prefill=%.0fms decode=%.1f tok/s\n", t, st.ctxTok, st.prefillMs, st.decodeTokS)
	}
	_ = os.WriteFile(filepath.Join(outDir, "enginebench-ollama.txt"), []byte(lastOut), 0o644)
	return stats, nil
}

// ----- llama.cpp server (/v1/chat/completions; reads "timings") -----

func runLlamaCpp(baseURL, system, seed, chunk string, turns, gen int, outDir string) ([]turnStat, error) {
	client := &http.Client{Timeout: 30 * time.Minute}
	var stats []turnStat
	var lastOut string
	for t := 0; t < turns; t++ {
		gmsgs := buildTurns(system, seed, chunk, t)
		oms := make([]ollamaMsg, len(gmsgs))
		for i, m := range gmsgs {
			oms[i] = ollamaMsg{Role: m.Role, Content: m.Content}
		}
		body, _ := json.Marshal(map[string]any{
			"messages": oms, "temperature": 0, "max_tokens": gen, "stream": false,
		})
		resp, err := client.Post(strings.TrimRight(baseURL, "/")+"/v1/chat/completions", "application/json", bytes.NewReader(body))
		if err != nil {
			return stats, err
		}
		var r struct {
			Choices []struct {
				Message ollamaMsg `json:"message"`
			} `json:"choices"`
			Timings struct {
				PromptN            int     `json:"prompt_n"`
				PromptMs           float64 `json:"prompt_ms"`
				PredictedN         int     `json:"predicted_n"`
				PredictedPerSecond float64 `json:"predicted_per_second"`
			} `json:"timings"`
		}
		_ = json.NewDecoder(resp.Body).Decode(&r)
		resp.Body.Close()
		st := turnStat{backend: "llamacpp", turn: t, ctxTok: r.Timings.PromptN, outTok: r.Timings.PredictedN,
			prefillMs: r.Timings.PromptMs, decodeTokS: r.Timings.PredictedPerSecond}
		st.totalMs = r.Timings.PromptMs
		stats = append(stats, st)
		if len(r.Choices) > 0 {
			lastOut = r.Choices[0].Message.Content
		}
		fmt.Fprintf(os.Stderr, "  llamacpp turn %d: ctx=%d prefill=%.0fms decode=%.1f tok/s\n", t, st.ctxTok, st.prefillMs, st.decodeTokS)
	}
	_ = os.WriteFile(filepath.Join(outDir, "enginebench-llamacpp.txt"), []byte(lastOut), 0o644)
	return stats, nil
}

func printTable(stats []turnStat) {
	fmt.Println()
	fmt.Printf("%-10s %5s %9s %12s %14s %10s\n", "BACKEND", "TURN", "CTX_TOK", "PREFILL_MS", "DECODE_TOK/S", "OUT_TOK")
	fmt.Println(strings.Repeat("─", 64))
	for _, s := range stats {
		fmt.Printf("%-10s %5d %9d %12.0f %14.1f %10d\n", s.backend, s.turn, s.ctxTok, s.prefillMs, s.decodeTokS, s.outTok)
	}
	// Per-backend decode averages (the headline number — does memdoor hold up?).
	sum := map[string]float64{}
	cnt := map[string]int{}
	for _, s := range stats {
		if s.decodeTokS > 0 {
			sum[s.backend] += s.decodeTokS
			cnt[s.backend]++
		}
	}
	fmt.Println(strings.Repeat("─", 64))
	for b, n := range cnt {
		fmt.Printf("avg decode  %-10s %.1f tok/s over %d turns\n", b, sum[b]/float64(n), n)
	}
	fmt.Println("\noutputs written to <out>/enginebench-<backend>.txt for quality diff")
}
