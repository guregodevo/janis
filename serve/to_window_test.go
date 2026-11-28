package serve

import "testing"

// A request that names no max_tokens is not cut at a default: the engine is
// asked to run to the end of the window, and a reply that fills the budget the
// engine settled on still ends "length", streamed or not.
func TestARequestWithNoMaxTokensRunsToTheWindow(t *testing.T) {
	installFake(t, "Qwen3-8B-4bit", 40960)
	for _, c := range []struct {
		name, reply, extra, finish string
		toWindow                   bool
	}{
		{"no max_tokens, stops by itself", "one two three", ``, "stop", true},
		{"no max_tokens, fills the window", "one two three four", ``, "length", true},
		{"max_tokens named", "one two", `,"max_tokens":7`, "stop", false},
	} {
		eng := &fakeEngine{reply: c.reply, budget: 4}
		srv := newTestServer(t, eng)
		body := `{"model":"Qwen3-8B-4bit","messages":[{"role":"user","content":"hi"}]` + c.extra
		_, v := post(t, srv.URL+"/v1/chat/completions", body+`}`)
		if eng.opts.ToWindow != c.toWindow {
			t.Fatalf("%s: ToWindow = %v, want %v", c.name, eng.opts.ToWindow, c.toWindow)
		}
		if c.toWindow && eng.opts.MaxTokens != 0 {
			t.Fatalf("%s: MaxTokens = %d with none asked: a default cap", c.name, eng.opts.MaxTokens)
		}
		if got := v["choices"].([]any)[0].(map[string]any)["finish_reason"]; got != c.finish {
			t.Errorf("%s: finish %v, want %s", c.name, got, c.finish)
		}
		resp, _ := post(t, srv.URL+"/v1/chat/completions", body+`,"stream":true}`)
		if _, finish, _, _ := readStream(t, resp); finish != c.finish {
			t.Errorf("%s streamed: finish %q, want %s", c.name, finish, c.finish)
		}
	}
}
