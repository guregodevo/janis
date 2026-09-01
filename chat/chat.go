// Package chat renders OpenAI-style messages into each model family's prompt
// format (the strings then tokenize into the right special-token IDs).
package chat

import (
	"os"
	"strings"
)

// thinkingEnabled reports whether Qwen3-style reasoning blocks are allowed.
// Default OFF: for grounded wiki/RAG answers the reasoning is wasted decode (a
// ~370-token <think> block at long-context decode rates costs ~30s and doesn't
// improve answers already present in the retrieved context). Re-enable with
// MLX_ENABLE_THINKING=1.
func thinkingEnabled() bool { return os.Getenv("MLX_ENABLE_THINKING") == "1" }

// Message is one chat turn.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ApplyTemplate renders messages into the prompt for the given model_type, with
// the assistant turn opened.
func ApplyTemplate(modelType string, msgs []Message) string {
	switch modelType {
	case "gemma2":
		return gemma(msgs)
	case "llama":
		return llama3(msgs)
	case "qwen3", "qwen3_moe", "qwen3_5":
		// Qwen3/3.5 (dense, MoE, hybrid) are reasoning families: prime the
		// assistant turn with an empty think block to skip the <think>…</think>
		// generation (the official enable_thinking=false behavior — Qwen3.5's
		// chat_template.jinja emits the same '<think>\n\n</think>\n\n' priming)
		// unless thinking is explicitly enabled. Without this, a 30B-A3B agent
		// turn spent its entire decode budget inside <think>, the provider
		// stripped it, and the agent went silent ("empty final text").
		return chatml(msgs, !thinkingEnabled())
	default: // qwen2 -> ChatML (no reasoning)
		return chatml(msgs, false)
	}
}

// StopMarker is the string that ends an assistant turn for the model family;
// the server tokenizes it to derive the stop token id.
func StopMarker(modelType string) string {
	switch modelType {
	case "gemma2":
		return "<end_of_turn>"
	case "llama":
		return "<|eot_id|>"
	default:
		return "<|im_end|>"
	}
}

func chatml(msgs []Message, noThink bool) string {
	var b strings.Builder
	for _, m := range msgs {
		b.WriteString("<|im_start|>")
		b.WriteString(m.Role)
		b.WriteByte('\n')
		if noThink && m.Role == "assistant" && !strings.Contains(m.Content, "<think>") {
			// HISTORY assistant turns carry the same empty think block the live
			// turn was primed with, so a follow-up prompt EXTENDS the cached
			// token sequence instead of diverging at the first assistant turn.
			// Without this, every multi-turn prompt truncated the KV prefix back
			// to turn 1 — and a hybrid (qwen3_5) session, whose recurrent state
			// cannot truncate at all, re-prefilled the whole conversation every
			// turn (measured: turn 2 "reused=0" instead of the full prefix).
			b.WriteString("<think>\n\n</think>\n\n")
		}
		b.WriteString(m.Content)
		b.WriteString("<|im_end|>\n")
	}
	b.WriteString("<|im_start|>assistant\n")
	if noThink {
		// An empty, already-closed think block tells Qwen3 reasoning is done, so
		// it generates the answer directly.
		b.WriteString("<think>\n\n</think>\n\n")
	}
	return b.String()
}

func gemma(msgs []Message) string {
	var b strings.Builder
	b.WriteString("<bos>")
	for _, m := range foldSystem(msgs) {
		role := m.Role
		if role == "assistant" {
			role = "model"
		}
		b.WriteString("<start_of_turn>")
		b.WriteString(role)
		b.WriteByte('\n')
		b.WriteString(m.Content)
		b.WriteString("<end_of_turn>\n")
	}
	b.WriteString("<start_of_turn>model\n")
	return b.String()
}

func llama3(msgs []Message) string {
	var b strings.Builder
	b.WriteString("<|begin_of_text|>")
	for _, m := range msgs {
		b.WriteString("<|start_header_id|>")
		b.WriteString(m.Role)
		b.WriteString("<|end_header_id|>\n\n")
		b.WriteString(m.Content)
		b.WriteString("<|eot_id|>")
	}
	b.WriteString("<|start_header_id|>assistant<|end_header_id|>\n\n")
	return b.String()
}

// foldSystem merges a leading system message into the first user turn (Gemma has
// no system role).
func foldSystem(msgs []Message) []Message {
	if len(msgs) == 0 || msgs[0].Role != "system" {
		return msgs
	}
	sys, rest := msgs[0].Content, msgs[1:]
	out := make([]Message, 0, len(rest))
	if len(rest) > 0 && rest[0].Role == "user" {
		out = append(out, Message{Role: "user", Content: sys + "\n\n" + rest[0].Content})
		out = append(out, rest[1:]...)
	} else {
		out = append(out, Message{Role: "user", Content: sys})
		out = append(out, rest...)
	}
	return out
}
