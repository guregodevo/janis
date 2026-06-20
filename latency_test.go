package llm

import (
	"os"
	"strings"
	"testing"
	"time"
)

// TestChatLatency measures the prefill (time-to-first-token) vs decode split for
// a RAG-sized prompt, so we can see where a ~2-minute wiki answer actually goes.
// Gated MLX_LATENCY=1; set MLX_MODEL_DIR and optionally MLX_PREFILL_CHUNK.
//
//	MLX_LATENCY=1 MLX_MODEL_DIR=<chat> go test ./llm/ -run TestChatLatency -v
func TestChatLatency(t *testing.T) {
	if os.Getenv("MLX_LATENCY") != "1" {
		t.Skip("set MLX_LATENCY=1 to run")
	}
	dir := os.Getenv("MLX_MODEL_DIR")
	if dir == "" {
		t.Skip("set MLX_MODEL_DIR")
	}
	eng, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer eng.Close()

	ctxText := strings.Repeat("Under Article 22 GDPR a data subject has the right not to be subject "+
		"to a decision based solely on automated processing which produces legal effects; the controller "+
		"must provide meaningful information about the logic involved. ", 130)
	msgs := []Message{
		{Role: "system", Content: "Answer only from the context. " + ctxText},
		{Role: "user", Content: "What must we disclose for an automated loan denial?"},
	}
	promptTok := eng.NumTokens(msgs[0].Content) + eng.NumTokens(msgs[1].Content)

	start := time.Now()
	var ttft time.Duration
	first := true
	reply := eng.ChatStream(msgs, Options{MaxTokens: 800}, func(string) {
		if first {
			ttft = time.Since(start)
			first = false
		}
	})
	total := time.Since(start)
	outTok := eng.NumTokens(reply)
	decode := total - ttft

	t.Logf("chunk=%s prompt=%d tok", os.Getenv("MLX_PREFILL_CHUNK"), promptTok)
	t.Logf("  PREFILL (ttft) = %5.1fs  (%.0f tok/s)", ttft.Seconds(), float64(promptTok)/ttft.Seconds())
	t.Logf("  DECODE         = %5.1fs  (%d tok, %.1f tok/s)", decode.Seconds(), outTok, float64(outTok)/decode.Seconds())
	t.Logf("  TOTAL          = %5.1fs", total.Seconds())
}
