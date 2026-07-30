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
