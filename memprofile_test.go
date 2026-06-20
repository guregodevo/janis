package llm

import (
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"memdoor/llm/mlxc"
)

// TestGatewayMemoryProfile reproduces the gateway's exact MLX workload in
// isolation — load the chat model, load the bge embedder, embed a query, then
// run a RAG-sized generation — logging MLX active/cache/peak + process RSS at
// each step so we can see precisely where the memory goes. Gated MLX_PROFILE=1.
//
//	MLX_PROFILE=1 MLX_MODEL_DIR=<chat-snapshot> go test ./llm/ -run TestGatewayMemoryProfile -v
func TestGatewayMemoryProfile(t *testing.T) {
	if os.Getenv("MLX_PROFILE") != "1" {
		t.Skip("set MLX_PROFILE=1 to run")
	}
	step := func(label string) {
		a, c, p := mlxc.MemoryStatsMB()
		t.Logf("%-22s MLX active=%6.0fMB cache=%5.0fMB peak=%6.0fMB | procRSS=%5.0fMB",
			label, a, c, p, selfRSSMB())
	}

	step("start")
	chatDir := os.Getenv("MLX_MODEL_DIR")
	if chatDir == "" {
		t.Skip("set MLX_MODEL_DIR to a chat snapshot")
	}
	eng, err := Open(chatDir)
	if err != nil {
		t.Fatal(err)
	}
	defer eng.Close()
	step("after chat load")

	emb, err := OpenEmbedder(bgeDir(t))
	if err != nil {
		t.Fatal(err)
	}
	defer emb.Close()
	step("after embed load")

	_ = emb.Embed("automated decision making profiling right to explanation GDPR article 22")
	step("after query embed")

	// RAG-sized prompt: ~3000 tokens of injected legal context + a question.
	ctxText := strings.Repeat("Under Article 22 GDPR, a data subject has the right not to be "+
		"subject to a decision based solely on automated processing which produces legal "+
		"effects. The controller must provide meaningful information about the logic involved. ", 120)
	msgs := []Message{
		{Role: "system", Content: "Answer only from the context. " + ctxText},
		{Role: "user", Content: "What must we disclose for an automated loan denial?"},
	}
	t.Logf("prompt tokens ~= %d", eng.NumTokens(msgs[0].Content)+eng.NumTokens(msgs[1].Content))
	reply := eng.Chat(msgs, Options{MaxTokens: 64})
	step("after RAG generation")
	t.Logf("reply: %.80q", reply)
}

func selfRSSMB() float64 {
	out, err := exec.Command("ps", "-o", "rss=", "-p", strconv.Itoa(os.Getpid())).Output()
	if err != nil {
		return -1
	}
	kb, _ := strconv.Atoi(strings.TrimSpace(string(out)))
	return float64(kb) / 1024
}

var _ = filepath.Dir
