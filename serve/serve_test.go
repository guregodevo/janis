package serve

import (
	"bufio"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	llm "github.com/guregodevo/janis"
)

// fakeEngine answers with a script and records the rendered prompt.
type fakeEngine struct {
	reply     string
	got       []llm.Message
	opts      llm.Options
	state     string
	questions []llm.Question
}

func (f *fakeEngine) ModelType() string { return "qwen3" }
func (f *fakeEngine) ChatStream(msgs []llm.Message, opts llm.Options, onDelta func(string)) string {
	f.got, f.opts = msgs, opts
	if onDelta != nil {
		for i := 0; i < len(f.reply); i += 3 { // three bytes at a time, like tokens
			end := min(i+3, len(f.reply))
			onDelta(f.reply[i:end])
		}
	}
	return f.reply
}
func (f *fakeEngine) NumTokens(text string) int { return len(strings.Fields(text)) }

// Decide answers every question with a fixed distribution: the first option
// at 0.8, the rest sharing 0.2; the state and questions are kept.
func (f *fakeEngine) Decide(state string, qs []llm.Question) ([]llm.Decision, error) {
	f.state, f.questions = state, qs
	out := make([]llm.Decision, len(qs))
	for i, q := range qs {
		probs := make([]float64, len(q.Options))
		probs[0] = 0.8
		for j := 1; j < len(probs); j++ {
			probs[j] = 0.2 / float64(len(probs)-1)
		}
		exp := 0.0
		for j, p := range probs {
			if q.Kind == llm.KindScore {
				exp += p * q.Values[j]
			}
		}
		out[i] = llm.Decision{Options: q.Options, Probs: probs, Index: 0, Confidence: 0.8, Expected: exp}
	}
	return out, nil
}
func (f *fakeEngine) Close() {}

// installFake puts one model in a scratch HuggingFace cache so the server
// has something to list and load.
func installFake(t *testing.T, name string, window int) {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	dir := filepath.Join(home, ".cache/huggingface/hub/models--mlx-community--"+name, "snapshots", "abc")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	cfg := `{"model_type":"qwen3","max_position_embeddings":` + itoa(window) + `}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
}

func itoa(n int) string { b, _ := json.Marshal(n); return string(b) }

func newTestServer(t *testing.T, eng *fakeEngine) *httptest.Server {
	t.Helper()
	s := NewWithLoader(func(dir string) (Generator, error) { return eng, nil })
	s.logf = t.Logf
	srv := httptest.NewServer(s.Handler())
	t.Cleanup(srv.Close)
	return srv
}

func post(t *testing.T, url, body string) (*http.Response, map[string]any) {
	t.Helper()
	resp, err := http.Post(url, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	var v map[string]any
	if !strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream") {
		_ = json.NewDecoder(resp.Body).Decode(&v)
	}
	return resp, v
}

func TestModelsListTheInstalledOnesWithTheirWindow(t *testing.T) {
	installFake(t, "Qwen3-8B-4bit", 40960)
	// A snapshot without weights (a download that stopped) is not served.
	home, _ := os.UserHomeDir()
	partial := filepath.Join(home, ".cache/huggingface/hub/models--mlx-community--Half-7B/snapshots/abc")
	_ = os.MkdirAll(partial, 0o755)
	_ = os.WriteFile(filepath.Join(partial, "config.json"), []byte(`{"max_position_embeddings":1}`), 0o644)
	srv := newTestServer(t, &fakeEngine{})
	resp, err := http.Get(srv.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	var v struct {
		Data []modelCard `json:"data"`
	}
	_ = json.NewDecoder(resp.Body).Decode(&v)
	if len(v.Data) != 1 || v.Data[0].ID != "Qwen3-8B-4bit" || v.Data[0].ContextLength != 40960 || v.Data[0].OwnedBy != "janis" {
		t.Fatalf("models: %+v", v.Data)
	}
}

func TestAPlainAnswerStreamedAndNot(t *testing.T) {
	installFake(t, "Qwen3-8B-4bit", 40960)
	eng := &fakeEngine{reply: "The capital of France is Paris."}
	srv := newTestServer(t, eng)

	resp, v := post(t, srv.URL+"/v1/chat/completions", `{"model":"Qwen3-8B-4bit","messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"Capital of France?"}],"temperature":0}`)
	if resp.StatusCode != 200 {
		t.Fatalf("status %d: %v", resp.StatusCode, v)
	}
	choice := v["choices"].([]any)[0].(map[string]any)
	msg := choice["message"].(map[string]any)
	if msg["content"] != "The capital of France is Paris." || choice["finish_reason"] != "stop" || msg["tool_calls"] != nil {
		t.Fatalf("answer: %v", choice)
	}
	if u := v["usage"].(map[string]any); u["completion_tokens"].(float64) != 6 {
		t.Fatalf("usage: %v", u)
	}
	if len(eng.got) != 2 || eng.got[0].Role != "system" || eng.got[0].Content != "Be brief." || eng.got[1].Content != "Capital of France?" || eng.opts.Temp != 0 || eng.opts.ToolCallGrammar {
		t.Fatalf("the prompt the engine saw: %+v %+v", eng.got, eng.opts)
	}

	resp, _ = post(t, srv.URL+"/v1/chat/completions", `{"model":"Qwen3-8B-4bit","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"Capital of France?"}]}`)
	text, finish, sawUsage, sawDone := readStream(t, resp)
	if text != "The capital of France is Paris." || finish != "stop" || !sawUsage || !sawDone {
		t.Fatalf("stream: %q %q usage=%v done=%v", text, finish, sawUsage, sawDone)
	}
}

func TestAStreamNeverSplitsACharacter(t *testing.T) {
	installFake(t, "Qwen3-8B-4bit", 40960)
	reply := "apple — a fruit; café, naïve, 日本 — done"
	srv := newTestServer(t, &fakeEngine{reply: reply})
	resp, _ := post(t, srv.URL+"/v1/chat/completions", `{"model":"Qwen3-8B-4bit","stream":true,"messages":[{"role":"user","content":"Words?"}]}`)
	if text, _, _, _ := readStream(t, resp); text != reply {
		t.Fatalf("stream: %q, want %q", text, reply)
	}
}

func TestAToolCallRoundTrip(t *testing.T) {
	installFake(t, "Qwen3-8B-4bit", 40960)
	eng := &fakeEngine{reply: "Let me look.\n<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"main.go\"}}\n</tool_call>"}
	srv := newTestServer(t, eng)
	body := `{"model":"Qwen3-8B-4bit","tools":[{"type":"function","function":{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}],"messages":[{"role":"user","content":"What is in main.go?"}]}`

	resp, v := post(t, srv.URL+"/v1/chat/completions", body)
	if resp.StatusCode != 200 {
		t.Fatalf("status %d: %v", resp.StatusCode, v)
	}
	choice := v["choices"].([]any)[0].(map[string]any)
	msg := choice["message"].(map[string]any)
	calls := msg["tool_calls"].([]any)
	fn := calls[0].(map[string]any)["function"].(map[string]any)
	if choice["finish_reason"] != "tool_calls" || len(calls) != 1 || fn["name"] != "read_file" || fn["arguments"] != `{"path": "main.go"}` || msg["content"] != "Let me look." {
		t.Fatalf("tool call: %v", choice)
	}
	sys := eng.got[0].Content
	if eng.got[0].Role != "system" || !strings.Contains(sys, "<tools>") || !strings.Contains(sys, `"name":"read_file"`) || !strings.Contains(sys, "<tool_call></tool_call>") || !eng.opts.ToolCallGrammar || eng.opts.ForceToolCall {
		t.Fatalf("the tools are rendered into the system prompt and the grammar is on: %q %+v", sys, eng.opts)
	}

	// The next turn replays the call and its result in the model's own form.
	eng.reply = "main.go starts the server."
	followUp := `{"model":"Qwen3-8B-4bit","stream":true,"tool_choice":"required","tools":[{"type":"function","function":{"name":"read_file","parameters":{"type":"object"}}}],"messages":[{"role":"user","content":"What is in main.go?"},{"role":"assistant","content":"Let me look.","tool_calls":[{"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"main.go\"}"}}]},{"role":"tool","tool_call_id":"call_1","content":"package main\nfunc main() { serve() }"}]}`
	resp, _ = post(t, srv.URL+"/v1/chat/completions", followUp)
	text, finish, _, _ := readStream(t, resp)
	if text != "main.go starts the server." || finish != "stop" {
		t.Fatalf("follow-up: %q %q", text, finish)
	}
	if len(eng.got) != 4 || eng.got[2].Role != "assistant" || !strings.Contains(eng.got[2].Content, "<tool_call>\n{\"arguments\":{\"path\":\"main.go\"},\"name\":\"read_file\"}\n</tool_call>") || eng.got[3].Role != "user" || !strings.HasPrefix(eng.got[3].Content, "<tool_response>\npackage main") || !eng.opts.ForceToolCall {
		t.Fatalf("the transcript replayed in the Hermes form: %+v %+v", eng.got, eng.opts)
	}

	// Streamed, a tool call never leaks its tag as text.
	eng.reply = "<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.go\"}}\n</tool_call>"
	resp, _ = post(t, srv.URL+"/v1/chat/completions", strings.Replace(body, `"model"`, `"stream":true,"model"`, 1))
	text, finish, _, _ = readStream(t, resp)
	if text != "" || finish != "tool_calls" {
		t.Fatalf("streamed tool call: text=%q finish=%q", text, finish)
	}
}

func TestAnUnknownModelIsRefusedInWords(t *testing.T) {
	installFake(t, "Qwen3-8B-4bit", 40960)
	srv := newTestServer(t, &fakeEngine{reply: "x"})
	resp, v := post(t, srv.URL+"/v1/chat/completions", `{"model":"nope","messages":[{"role":"user","content":"hi"}]}`)
	if resp.StatusCode != 404 || !strings.Contains(v["error"].(map[string]any)["message"].(string), "not installed") {
		t.Fatalf("%d %v", resp.StatusCode, v)
	}
}

// readStream collects the text deltas, the finish reason, and whether a
// usage chunk and [DONE] came.
func readStream(t *testing.T, resp *http.Response) (text, finish string, sawUsage, sawDone bool) {
	t.Helper()
	defer resp.Body.Close()
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			sawDone = true
			continue
		}
		var c struct {
			Choices []struct {
				Delta struct {
					Content   string     `json:"content"`
					ToolCalls []toolCall `json:"tool_calls"`
				} `json:"delta"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
			Usage *usage `json:"usage"`
		}
		if err := json.Unmarshal([]byte(payload), &c); err != nil {
			t.Fatalf("bad chunk %q: %v", payload, err)
		}
		if c.Usage != nil {
			sawUsage = true
		}
		for _, ch := range c.Choices {
			text += ch.Delta.Content
			if ch.FinishReason != nil {
				finish = *ch.FinishReason
			}
		}
	}
	return
}

// The System One wire on the engine's Decide: a noul, a choice and a score
// about one state, answered with probabilities; the client's own model
// name is accepted and the loaded model answers.
func TestDecisionsAnswerTheSystemOneWire(t *testing.T) {
	installFake(t, "Qwen3-8B-4bit", 40960)
	eng := &fakeEngine{}
	srv := newTestServer(t, eng)
	body := `{"model":"jev-latest","state":"func add(a, b int) int { return a + b }","questions":{
		"relevant":{"type":"noul","instructions":"Is this hit relevant to the task?","criteria":{"true":"it is about the add function","false":"it is about something else"}},
		"kind":{"type":"choice","instructions":"What kind of code is this?","criteria":{"test":"a test","impl":"an implementation","doc":"documentation"}},
		"quality":{"type":"score","instructions":"How clear is it?","criteria":["unclear","fine","clear"]}}}`
	resp, v := post(t, srv.URL+"/v1/systemone", body)
	if resp.StatusCode != 200 {
		t.Fatalf("status %d: %v", resp.StatusCode, v)
	}
	if v["model"] != "Qwen3-8B-4bit" || eng.state != "func add(a, b int) int { return a + b }" || len(eng.questions) != 3 {
		t.Fatalf("answered by the loaded model on the state: %v %+v", v, eng.questions)
	}
	answers := v["answers"].(map[string]any)
	rel := answers["relevant"].(map[string]any)
	if rel["type"] != "noul" || rel["noul"].(float64) != 0.8 || rel["confidence"].(float64) != 0.8 {
		t.Fatalf("noul: %v", rel)
	}
	kind := answers["kind"].(map[string]any)
	if kind["choice"] != "doc" || kind["probabilities"].(map[string]any)["doc"].(float64) != 0.8 || len(kind["probabilities"].(map[string]any)) != 3 {
		t.Fatalf("choice (labels in a fixed order, the first favoured by the fake): %v", kind)
	}
	q := answers["quality"].(map[string]any)
	if sc := q["score"].(float64); sc < 0.299 || sc > 0.301 || q["probabilities"].(map[string]any)["0"].(float64) != 0.8 {
		t.Fatalf("score: a zero-based expected level and a probability per level: %v", q)
	}
	// The questions the engine saw carry the criteria in their text.
	for _, eq := range eng.questions {
		switch eq.Kind {
		case llm.KindYesNo:
			if !strings.Contains(eq.Text, "Yes means: it is about the add function") {
				t.Fatalf("noul criteria: %q", eq.Text)
			}
		case llm.KindChoice:
			if !strings.Contains(eq.Text, "impl: an implementation") || strings.Join(eq.Options, ",") != "doc,impl,test" {
				t.Fatalf("choice: %q %v", eq.Text, eq.Options)
			}
		case llm.KindScore:
			if strings.Join(eq.Options, ",") != "unclear,fine,clear" || eq.Values[2] != 2 {
				t.Fatalf("score levels: %v %v", eq.Options, eq.Values)
			}
		}
	}
	resp, v = post(t, srv.URL+"/v1/systemone", `{"state":"x","questions":{"q":{"type":"choice","instructions":"?","criteria":{"only":"one"}}}}`)
	if resp.StatusCode != 422 || !strings.Contains(v["error"].(map[string]any)["message"].(string), "options") {
		t.Fatalf("a one-option choice is refused in words: %d %v", resp.StatusCode, v)
	}
}

// The loaded model is listed first, so a client taking the first entry (a
// probe, a default) gets the one answering now instead of forcing a swap.
func TestTheLoadedModelIsListedFirst(t *testing.T) {
	installFake(t, "Qwen3-8B-4bit", 40960)
	home, _ := os.UserHomeDir()
	for _, n := range []string{"A-first-1B", "Z-last-1B"} {
		d := filepath.Join(home, ".cache/huggingface/hub/models--mlx-community--"+n+"/snapshots/abc")
		_ = os.MkdirAll(d, 0o755)
		_ = os.WriteFile(filepath.Join(d, "config.json"), []byte(`{}`), 0o644)
		_ = os.WriteFile(filepath.Join(d, "model.safetensors"), []byte("x"), 0o644)
	}
	srv := newTestServer(t, &fakeEngine{reply: "ok"})
	post(t, srv.URL+"/v1/chat/completions", `{"model":"Z-last-1B","messages":[{"role":"user","content":"hi"}]}`)
	resp, _ := http.Get(srv.URL + "/v1/models")
	var v struct {
		Data []modelCard `json:"data"`
	}
	_ = json.NewDecoder(resp.Body).Decode(&v)
	if len(v.Data) != 3 || v.Data[0].ID != "Z-last-1B" {
		t.Fatalf("loaded first: %+v", v.Data)
	}
}
