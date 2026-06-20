// Package chat renders OpenAI-style messages into each model family's prompt
// format (the strings then tokenize into the right special-token IDs).
package chat

import "strings"

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
	default: // qwen2, qwen3 -> ChatML
		return chatml(msgs)
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

func chatml(msgs []Message) string {
	var b strings.Builder
	for _, m := range msgs {
		b.WriteString("<|im_start|>")
		b.WriteString(m.Role)
		b.WriteByte('\n')
		b.WriteString(m.Content)
		b.WriteString("<|im_end|>\n")
	}
	b.WriteString("<|im_start|>assistant\n")
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
