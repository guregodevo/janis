package chat

import (
	"strings"
	"testing"
)

// Both Qwen3 variants must prime the closed think block by default — a
// reasoning model without it spends the whole decode budget inside
// <think>…</think>, which the provider strips to an empty (silent) reply.
func TestApplyTemplateSuppressesThinkingForQwen3Family(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "hi"}}
	for _, mt := range []string{"qwen3", "qwen3_moe"} {
		out := ApplyTemplate(mt, msgs)
		if !strings.Contains(out, "<think>\n\n</think>") {
			t.Errorf("%s: prompt must prime the closed think block, got tail %q", mt, out[len(out)-40:])
		}
	}
	for _, mt := range []string{"qwen2", "llama", "gemma2"} {
		if out := ApplyTemplate(mt, msgs); strings.Contains(out, "<think>") {
			t.Errorf("%s: non-reasoning template must not carry a think block", mt)
		}
	}
}

// MLX_ENABLE_THINKING=1 re-enables reasoning for both variants.
func TestApplyTemplateThinkingOptIn(t *testing.T) {
	t.Setenv("MLX_ENABLE_THINKING", "1")
	msgs := []Message{{Role: "user", Content: "hi"}}
	for _, mt := range []string{"qwen3", "qwen3_moe"} {
		if out := ApplyTemplate(mt, msgs); strings.Contains(out, "<think>\n\n</think>") {
			t.Errorf("%s: opt-in thinking must not pre-close the think block", mt)
		}
	}
}

// History assistant turns must render with the same empty think block the live
// turn was primed with — otherwise a follow-up prompt diverges from the cached
// token sequence at the first assistant turn and destroys prefix reuse (total
// re-prefill on hybrid qwen3_5 sessions, whose state cannot truncate).
func TestChatmlHistoryAssistantTurnsMatchPriming(t *testing.T) {
	turn1 := []Message{{Role: "user", Content: "hi"}}
	p1 := ApplyTemplate("qwen3_5", turn1)
	reply := "hello"
	turn2 := append(turn1,
		Message{Role: "assistant", Content: reply},
		Message{Role: "user", Content: "again"})
	p2 := ApplyTemplate("qwen3_5", turn2)
	if !strings.HasPrefix(p2, p1+reply+"<|im_end|>\n") {
		t.Errorf("turn-2 prompt must extend turn-1 prompt + reply exactly:\np1=%q\np2=%q", p1, p2)
	}
	// A reply that already carries its own think block must not be double-primed.
	turn2[1].Content = "<think>\nx\n</think>\n\nhello"
	if got := ApplyTemplate("qwen3", turn2); strings.Count(got, "<think>") != 2 {
		t.Errorf("existing think block double-primed: %q", got)
	}
}
