package llm

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestQwen35MultiTurnLive drives the hybrid linear-attention engine through
// the two session paths that differ from pure-KV models: prefix EXTENSION
// (turn 2 reuses turn 1's conv/recurrent state) and prefix DIVERGENCE (a new
// conversation cannot truncate a recurrent state, so the session must reset
// and re-prefill from scratch). Gated on the real snapshot being installed.
//
//	MLX_QWEN35_E2E=1 go test ./llm/ -run TestQwen35MultiTurnLive -v
func TestQwen35MultiTurnLive(t *testing.T) {
	if os.Getenv("MLX_QWEN35_E2E") != "1" {
		t.Skip("set MLX_QWEN35_E2E=1 to run (loads the 9B snapshot)")
	}
	home, _ := os.UserHomeDir()
	dirs, _ := filepath.Glob(filepath.Join(home,
		".cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/*"))
	if len(dirs) == 0 {
		t.Skip("Qwen3.5-9B-MLX-4bit not installed")
	}
	eng, err := Open(dirs[0])
	if err != nil {
		t.Fatal(err)
	}
	defer eng.Close()
	if eng.ModelType() != "qwen3_5" {
		t.Fatalf("ModelType = %q, want qwen3_5", eng.ModelType())
	}

	opts := Options{Temp: 0, MaxTokens: 40}
	turn1 := []Message{{Role: "user", Content: "My favorite number is 17. Reply with just OK."}}
	r1 := eng.Chat(turn1, opts)
	if strings.TrimSpace(r1) == "" {
		t.Fatal("turn 1 produced an empty reply")
	}

	// Extension: the grown conversation shares the full turn-1 prefix, so the
	// recurrent state is reused; recall proves the state carried real content.
	turn2 := append(turn1,
		Message{Role: "assistant", Content: r1},
		Message{Role: "user", Content: "What is my favorite number? Answer with just the number."})
	r2 := eng.Chat(turn2, opts)
	if !strings.Contains(r2, "17") {
		t.Errorf("turn 2 should recall 17 from the reused prefix, got %q", r2)
	}

	// Divergence: a fresh conversation shares almost no prefix; the hybrid
	// session must reset (LinearCache can't truncate) and still answer.
	r3 := eng.Chat([]Message{{Role: "user", Content: "What is 6 times 7? Answer with just the number."}}, opts)
	if !strings.Contains(r3, "42") {
		t.Errorf("post-reset turn should answer 42, got %q", r3)
	}
}
