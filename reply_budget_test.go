package llm

import "testing"

func TestAToWindowReplyGetsWhatThePromptLeavesOfTheWindow(t *testing.T) {
	told := 0
	e := &Engine{window: 100}
	if got := e.replyBudget(Options{ToWindow: true, OnBudget: func(n int) { told = n }}, 512, 30); got != 70 || told != 70 {
		t.Fatalf("budget %d, told %d, want 70", got, told)
	}
	if got := e.replyBudget(Options{ToWindow: true}, 512, 100); got != 1 {
		t.Fatalf("a prompt that fills the window: %d, want 1", got)
	}
	if got := e.replyBudget(Options{ToWindow: true, MaxTokens: 50}, 50, 30); got != 50 {
		t.Fatalf("a named max_tokens is kept: %d", got)
	}
	if got := (&Engine{}).replyBudget(Options{ToWindow: true}, 512, 30); got != windowlessBudget {
		t.Fatalf("no window in the config: %d, want %d", got, windowlessBudget)
	}
	if got := e.replyBudget(Options{}, 512, 30); got != 512 {
		t.Fatalf("an internal call without ToWindow: %d, want 512", got)
	}
}
