package chat

import (
	"encoding/json"
	"os"
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

// INKLING 2'S PROMPT IS WHAT ITS OWN TEMPLATE RENDERS.
//
// testdata/inkling2_template.json holds, for a few conversations, the output
// of the chat_template.jinja published with the checkpoints: with
// reasoning_effort "none" (thinking off, janis's default) and with the
// template's default effort (thinking on).
func TestInklingPromptIsWhatItsTemplateRenders(t *testing.T) {
	raw, err := os.ReadFile("testdata/inkling2_template.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases map[string]struct {
		Messages []Message `json:"messages"`
		None     string    `json:"none"`
		Default  string    `json:"default"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases) == 0 {
		t.Fatal("no cases")
	}
	for name, c := range cases {
		t.Setenv("MLX_ENABLE_THINKING", "")
		if got := ApplyTemplate("inkling2_loop", c.Messages); got != c.None {
			t.Errorf("%s, thinking off:\n got %q\nwant %q", name, got, c.None)
		}
		t.Setenv("MLX_ENABLE_THINKING", "1")
		if got := ApplyTemplate("inkling2_loop", c.Messages); got != c.Default {
			t.Errorf("%s, thinking on:\n got %q\nwant %q", name, got, c.Default)
		}
	}
}

// INKLING 2'S REPLY IS READ MESSAGE BY MESSAGE.
//
// The thinking goes apart, a call becomes the <tool_call> block the server
// parses, the text stays. Token ids stand for their text here.
func TestInklingReplyIsReadMessageByMessage(t *testing.T) {
	m := InklingMarks{Model: 1, Thinking: 2, Text: 3, Call: 4, End: 5}
	words := map[int32]string{
		10: "The user asks.", 11: "Paris", 12: "read_file",
		13: `{"name":"read_file","args":{"path":"go.mod"}}`, 14: "On it.", 15: `{"name":"read_`,
	}
	decode := func(ids []int32) string {
		var b strings.Builder
		for _, id := range ids {
			b.WriteString(words[id])
		}
		return b.String()
	}
	call := "<tool_call>\n" + `{"name":"read_file","arguments":{"path":"go.mod"}}` + "\n</tool_call>"
	for _, tc := range []struct {
		name            string
		ids             []int32
		reply, thinking string
	}{
		{"thinking then text", []int32{1, 2, 10, 5, 1, 3, 11, 5}, "Paris", "The user asks."},
		{"a thought still open", []int32{1, 2, 10}, "", "The user asks."},
		{"text still open", []int32{1, 3, 11}, "Paris", ""},
		{"a call", []int32{1, 12, 4, 13, 5}, call, ""},
		{"text then a call", []int32{1, 3, 14, 5, 1, 12, 4, 13, 5}, "On it.\n" + call, ""},
		{"a call still open says nothing", []int32{1, 3, 14, 5, 1, 12, 4, 15}, "On it.", ""},
		{"nothing", nil, "", ""},
	} {
		reply, thinking := InklingReply(tc.ids, m, decode)
		if reply != tc.reply || thinking != tc.thinking {
			t.Errorf("%s: reply %q thinking %q, want %q and %q", tc.name, reply, thinking, tc.reply, tc.thinking)
		}
	}
}

// A request's reasoning_effort turns Qwen's thinking on or off; with none
// the server's MLX_ENABLE_THINKING decides, as before.
func TestQwenThinksAsTheRequestSays(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "hi"}}
	primed := func(out string) bool { return strings.HasSuffix(out, "<think>\n\n</think>\n\n") }
	cases := []struct {
		env, effort string
		thinks      bool
	}{
		{"", "", false},
		{"1", "", true},
		{"", "high", true},
		{"", "low", true},
		{"1", "none", false},
	}
	for _, c := range cases {
		t.Setenv("MLX_ENABLE_THINKING", c.env)
		for _, mt := range []string{"qwen3", "qwen3_moe", "qwen3_5"} {
			if out := ApplyTemplateEffort(mt, msgs, c.effort); primed(out) == c.thinks {
				t.Errorf("%s env=%q effort=%q: thinks %v, want %v", mt, c.env, c.effort, !primed(out), c.thinks)
			}
		}
	}
	t.Setenv("MLX_ENABLE_THINKING", "")
	if out := ApplyTemplateEffort("llama", msgs, "high"); strings.Contains(out, "<think>") {
		t.Errorf("llama took a think block: %q", out)
	}
}

// Qwen's <think> block leaves the reply as its thinking; a cut reply that
// never closed it is all thinking. Read token by token, both only grow, so
// a stream can send each as deltas.
func TestSplitThinkingPartsQwensReply(t *testing.T) {
	cases := []struct{ mt, reply, text, reasoning string }{
		{"qwen3_5", "<think>\nLet me see.\n</think>\n\nThe answer.", "The answer.", "Let me see.\n"},
		{"qwen3", "<think>\nStill going", "", "Still going"},
		{"qwen3_5", "The answer, no thinking.", "The answer, no thinking.", ""},
		{"qwen3_5", "<think>\n\n</think>\n\n<tool_call>{}</tool_call>", "<tool_call>{}</tool_call>", ""},
		{"llama", "<think>x</think>y", "<think>x</think>y", ""},
	}
	for _, c := range cases {
		if text, reasoning := SplitThinking(c.mt, c.reply); text != c.text || reasoning != c.reasoning {
			t.Errorf("%s %q: (%q, %q), want (%q, %q)", c.mt, c.reply, text, reasoning, c.text, c.reasoning)
		}
	}
	tokens := []string{"<think>", "\n", "Let", " me", " see", ".\n", "</think>", "\n\n", "The", " answer", "."}
	var reply, lastText, lastReasoning string
	for _, tok := range tokens {
		reply += tok
		text, reasoning := SplitThinking("qwen3_5", reply)
		if !strings.HasPrefix(text, lastText) || !strings.HasPrefix(reasoning, lastReasoning) {
			t.Fatalf("after %q: (%q, %q) does not extend (%q, %q)", reply, text, reasoning, lastText, lastReasoning)
		}
		lastText, lastReasoning = text, reasoning
	}
}
