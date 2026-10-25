// Package serve is janis as a server: the OpenAI-compatible endpoints a
// client such as Memdoor, or any OpenAI SDK, speaks to llama.cpp, Ollama or
// vLLM — GET /v1/models and POST /v1/chat/completions, streamed or not,
// with tools. One model is loaded at a time; a request naming another
// installed model swaps it. Generation is serialized: the engine is one GPU.
//
// Tools travel the Hermes way, the single format the engine's grammar
// enforces: the schemas are rendered into the system prompt, the model
// answers <tool_call>{"name":…,"arguments":{…}}</tool_call>, and the server
// hands them back as OpenAI tool_calls. A prior tool call and its result are
// replayed in the same form, so the transcript teaches the model its own
// protocol.
package serve

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	llm "github.com/guregodevo/janis"
	"github.com/guregodevo/janis/hostfit"
)

// Generator is what the server needs from a loaded model: the engine, or a
// fake in tests. llm.Engine satisfies it.
type Generator interface {
	ModelType() string
	ChatStream(msgs []llm.Message, opts llm.Options, onDelta func(string)) string
	Decide(state string, qs []llm.Question) ([]llm.Decision, error)
	NumTokens(text string) int
	Close()
}

// Loader opens an installed model's snapshot directory. The default wraps
// llm.Open; tests inject a fake.
type Loader func(modelDir string) (Generator, error)

// Server serves one engine at a time behind the OpenAI endpoints.
type Server struct {
	load    Loader
	mu      sync.Mutex // one generation at a time, and the swap under it
	current Generator
	name    string // the installed model's name the engine was opened from
	logf    func(string, ...any)
}

// New makes a server with the default loader. Name preloads that installed
// model (""), the first request's model is loaded otherwise.
func New(preload string) (*Server, error) {
	s := NewWithLoader(func(dir string) (Generator, error) { return llm.Open(dir) })
	if preload != "" {
		if err := s.ensure(preload); err != nil {
			return nil, err
		}
	}
	return s, nil
}

// NewWithLoader is New with the loader injected.
func NewWithLoader(load Loader) *Server {
	return &Server{load: load, logf: log.Printf}
}

// Handler routes the endpoints.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/models", s.handleModels)
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	mux.HandleFunc("POST /v1/chat/completions", s.handleChat)
	mux.HandleFunc("POST /v1/systemone", s.handleDecide)
	return mux
}

// Close releases the loaded engine.
func (s *Server) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.current != nil {
		s.current.Close()
		s.current, s.name = nil, ""
	}
}

// ---- models ----

type modelCard struct {
	ID            string `json:"id"`
	Object        string `json:"object"`
	Created       int64  `json:"created"`
	OwnedBy       string `json:"owned_by"`
	ContextLength int    `json:"context_length,omitempty"`
}

func (s *Server) handleModels(w http.ResponseWriter, _ *http.Request) {
	installed, err := installedModels()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	cards := make([]modelCard, 0, len(installed))
	for _, m := range installed {
		cards = append(cards, modelCard{ID: m.Name, Object: "model", Created: time.Now().Unix(), OwnedBy: "janis", ContextLength: contextLength(m.Path)})
	}
	// The loaded model first: a client that takes the first listed (a probe,
	// a default) gets the one that answers now, not a swap to another.
	s.mu.Lock()
	loaded := s.name
	s.mu.Unlock()
	for i, c := range cards {
		if c.ID == loaded && i > 0 {
			cards = append([]modelCard{c}, append(cards[:i:i], cards[i+1:]...)...)
			break
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": cards})
}

// installedModels is the cache's chat models whose weights are there
// (hostfit leaves out a snapshot a download stopped short of).
func installedModels() ([]hostfit.InstalledModel, error) { return hostfit.ListInstalled() }

// contextLength is the model's own window, from its config.json; 0 when the
// file does not say.
func contextLength(dir string) int {
	b, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return 0
	}
	var cfg struct {
		MaxPositionEmbeddings int `json:"max_position_embeddings"`
	}
	_ = json.Unmarshal(b, &cfg)
	return cfg.MaxPositionEmbeddings
}

// ensure loads name (an installed model's name, or "" for whatever is
// loaded, else the first installed), swapping the engine when it differs.
// Called with s.mu held.
func (s *Server) ensure(name string) error {
	if name == "" || name == s.name {
		if s.current != nil {
			return nil
		}
	}
	installed, err := installedModels()
	if err != nil {
		return err
	}
	if len(installed) == 0 {
		return fmt.Errorf("no model installed: janis serves the mlx-community models in ~/.cache/huggingface/hub")
	}
	pick := installed[0]
	if name != "" {
		found := false
		for _, m := range installed {
			if m.Name == name {
				pick, found = m, true
				break
			}
		}
		if !found {
			return fmt.Errorf("model %q is not installed; GET /v1/models lists the ones that are", name)
		}
	}
	if s.current != nil && s.name == pick.Name {
		return nil
	}
	eng, err := s.load(pick.Path)
	if err != nil {
		return fmt.Errorf("load %s: %w", pick.Name, err)
	}
	if s.current != nil {
		s.current.Close()
	}
	s.current, s.name = eng, pick.Name
	s.logf("loaded %s (%s)", pick.Name, eng.ModelType())
	return nil
}

// ---- chat completions ----

type chatRequest struct {
	Model               string          `json:"model"`
	Messages            []chatMessage   `json:"messages"`
	Tools               []chatTool      `json:"tools"`
	ToolChoice          json.RawMessage `json:"tool_choice"`
	Stream              bool            `json:"stream"`
	StreamOptions       *streamOptions  `json:"stream_options"`
	Temperature         *float64        `json:"temperature"`
	TopP                *float64        `json:"top_p"`
	MaxTokens           int             `json:"max_tokens"`
	MaxCompletionTokens int             `json:"max_completion_tokens"`
	Seed                *uint64         `json:"seed"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type chatMessage struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content"` // a string, or parts [{type:text,text}]
	ToolCalls  []toolCall      `json:"tool_calls,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
	Name       string          `json:"name,omitempty"`
}

type chatTool struct {
	Type     string `json:"type"`
	Function struct {
		Name        string          `json:"name"`
		Description string          `json:"description"`
		Parameters  json.RawMessage `json:"parameters"`
	} `json:"function"`
}

type toolCall struct {
	Index    *int   `json:"index,omitempty"`
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

const (
	defaultTemp      = 0.6
	defaultTopP      = 0.95
	defaultMaxTokens = 4096
)

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	var req chatRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, 64<<20)).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "a JSON body: "+err.Error())
		return
	}
	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "messages is empty")
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ensure(req.Model); err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}
	msgs := renderChat(req)
	opts := llm.Options{Temp: float32(defaultTemp), TopP: float32(defaultTopP), MaxTokens: defaultMaxTokens}
	if req.Temperature != nil {
		opts.Temp = float32(*req.Temperature)
	}
	if req.TopP != nil {
		opts.TopP = float32(*req.TopP)
	}
	if req.MaxCompletionTokens > 0 {
		opts.MaxTokens = req.MaxCompletionTokens
	} else if req.MaxTokens > 0 {
		opts.MaxTokens = req.MaxTokens
	}
	if req.Seed != nil {
		opts.Seed = *req.Seed
	}
	ctx := r.Context()
	opts.Cancel = func() bool { return ctx.Err() != nil }
	if len(req.Tools) > 0 {
		opts.ToolCallGrammar = true
		opts.ForceToolCall = string(req.ToolChoice) == `"required"`
	}
	inTokens := 0
	for _, m := range msgs {
		inTokens += s.current.NumTokens(m.Content)
	}
	id := "chatcmpl-" + fmt.Sprint(time.Now().UnixNano())
	created := time.Now().Unix()
	if !req.Stream {
		reply := s.current.ChatStream(msgs, opts, nil)
		text, calls := parseCalls(reply)
		finish := "stop"
		if len(calls) > 0 {
			finish = "tool_calls"
		}
		u := usage{PromptTokens: inTokens, CompletionTokens: s.current.NumTokens(reply)}
		u.TotalTokens = u.PromptTokens + u.CompletionTokens
		msg := map[string]any{"role": "assistant", "content": text}
		if len(calls) > 0 {
			msg["tool_calls"] = calls
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"id": id, "object": "chat.completion", "created": created, "model": s.name,
			"choices": []map[string]any{{"index": 0, "message": msg, "finish_reason": finish}},
			"usage":   u,
		})
		return
	}
	s.streamChat(w, ctx, req, msgs, opts, id, created, inTokens)
}

// streamChat writes server-sent events: the role, text deltas as they come
// (held back by the length of "<tool_call>" so an opening tag is never
// half-sent), then the tool calls, the finish and, when asked, the usage.
func (s *Server) streamChat(w http.ResponseWriter, ctx context.Context, req chatRequest, msgs []llm.Message, opts llm.Options, id string, created int64, inTokens int) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	flusher, _ := w.(http.Flusher)
	send := func(v any) {
		b, _ := json.Marshal(v)
		_, _ = w.Write([]byte("data: "))
		_, _ = w.Write(b)
		_, _ = w.Write([]byte("\n\n"))
		if flusher != nil {
			flusher.Flush()
		}
	}
	chunk := func(delta map[string]any, finish any) map[string]any {
		return map[string]any{"id": id, "object": "chat.completion.chunk", "created": created, "model": s.name,
			"choices": []map[string]any{{"index": 0, "delta": delta, "finish_reason": finish}}}
	}
	send(chunk(map[string]any{"role": "assistant", "content": ""}, nil))
	var full strings.Builder
	emitted, inCall := 0, false
	onDelta := func(d string) {
		full.WriteString(d)
		if inCall {
			return
		}
		text := full.String()
		if i := strings.Index(text[emitted:], openTag); i >= 0 {
			if i > 0 {
				send(chunk(map[string]any{"content": text[emitted : emitted+i]}, nil))
			}
			emitted += i
			inCall = true
			return
		}
		safe := len(text) - len(openTag) + 1
		if safe > emitted {
			send(chunk(map[string]any{"content": text[emitted:safe]}, nil))
			emitted = safe
		}
	}
	reply := s.current.ChatStream(msgs, opts, onDelta)
	_, calls := parseCalls(reply)
	finish := "stop"
	switch {
	case len(calls) > 0:
		finish = "tool_calls"
		for i := range calls {
			idx := i
			calls[i].Index = &idx
		}
		send(chunk(map[string]any{"tool_calls": calls}, nil))
	case !inCall && emitted < len(full.String()):
		// The tail a plain answer held back against an opening tag.
		send(chunk(map[string]any{"content": full.String()[emitted:]}, nil))
	}
	send(chunk(map[string]any{}, finish))
	if req.StreamOptions != nil && req.StreamOptions.IncludeUsage {
		u := usage{PromptTokens: inTokens, CompletionTokens: s.current.NumTokens(reply)}
		u.TotalTokens = u.PromptTokens + u.CompletionTokens
		send(map[string]any{"id": id, "object": "chat.completion.chunk", "created": created, "model": s.name, "choices": []any{}, "usage": u})
	}
	_, _ = w.Write([]byte("data: [DONE]\n\n"))
	if flusher != nil {
		flusher.Flush()
	}
}

// ---- the Hermes protocol ----

const (
	openTag  = "<tool_call>"
	closeTag = "</tool_call>"
)

var callBlock = regexp.MustCompile(`(?s)<tool_call>\s*(\{.*?\})\s*</tool_call>`)
var thinkBlock = regexp.MustCompile(`(?s)<think>.*?</think>\s*`)

// renderChat turns the OpenAI messages into the engine's role/content turns:
// the tool schemas go into the system prompt, a prior assistant tool call is
// replayed as its <tool_call> block, a tool result as a <tool_response>
// under the user role (the Qwen template's own form).
func renderChat(req chatRequest) []llm.Message {
	var out []llm.Message
	system := ""
	for _, m := range req.Messages {
		if m.Role == "system" || m.Role == "developer" {
			system += contentText(m.Content) + "\n"
		}
	}
	if len(req.Tools) > 0 {
		system = strings.TrimSpace(system)
		if system == "" {
			system = "You are a helpful assistant."
		}
		system += "\n\n" + renderTools(req.Tools)
	}
	if strings.TrimSpace(system) != "" {
		out = append(out, llm.Message{Role: "system", Content: strings.TrimSpace(system)})
	}
	for _, m := range req.Messages {
		switch m.Role {
		case "system", "developer":
			continue
		case "assistant":
			text := contentText(m.Content)
			for _, c := range m.ToolCalls {
				args := json.RawMessage(c.Function.Arguments)
				if !json.Valid(args) {
					b, _ := json.Marshal(c.Function.Arguments)
					args = b
				}
				b, _ := json.Marshal(map[string]any{"name": c.Function.Name, "arguments": args})
				text += "\n" + openTag + "\n" + string(b) + "\n" + closeTag
			}
			out = append(out, llm.Message{Role: "assistant", Content: strings.TrimSpace(text)})
		case "tool":
			body := "<tool_response>\n" + contentText(m.Content) + "\n</tool_response>"
			if n := len(out); n > 0 && out[n-1].Role == "user" && strings.HasPrefix(out[n-1].Content, "<tool_response>") {
				out[n-1].Content += "\n" + body // several results answer one turn
				continue
			}
			out = append(out, llm.Message{Role: "user", Content: body})
		default:
			out = append(out, llm.Message{Role: "user", Content: contentText(m.Content)})
		}
	}
	return out
}

// renderTools is the Hermes/Qwen tools block of the system prompt.
func renderTools(tools []chatTool) string {
	var b strings.Builder
	b.WriteString("# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n")
	for _, t := range tools {
		params := t.Function.Parameters
		if len(params) == 0 {
			params = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		j, _ := json.Marshal(map[string]any{"type": "function", "function": map[string]any{"name": t.Function.Name, "description": t.Function.Description, "parameters": params}})
		b.Write(j)
		b.WriteByte('\n')
	}
	b.WriteString("</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>")
	return b.String()
}

// contentText is the text of an OpenAI content field: a string, or the text
// parts of an array.
func contentText(raw json.RawMessage) string {
	if len(raw) == 0 || string(raw) == "null" {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if json.Unmarshal(raw, &parts) == nil {
		var b strings.Builder
		for _, p := range parts {
			if p.Type == "text" || p.Type == "input_text" {
				b.WriteString(p.Text)
			}
		}
		return b.String()
	}
	return string(raw)
}

// parseCalls splits a reply into its text and its tool calls; a think block
// is dropped from the text.
func parseCalls(reply string) (string, []toolCall) {
	var calls []toolCall
	for i, m := range callBlock.FindAllStringSubmatch(reply, -1) {
		var c struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		}
		if json.Unmarshal([]byte(m[1]), &c) != nil || c.Name == "" {
			continue
		}
		args := string(c.Arguments)
		if args == "" || args == "null" {
			args = "{}"
		}
		var s string
		if json.Unmarshal(c.Arguments, &s) == nil { // arguments given as a JSON string
			args = s
		}
		tc := toolCall{ID: fmt.Sprintf("call_%d", i+1), Type: "function"}
		tc.Function.Name, tc.Function.Arguments = c.Name, args
		calls = append(calls, tc)
	}
	text := callBlock.ReplaceAllString(reply, "")
	text = thinkBlock.ReplaceAllString(text, "")
	return strings.TrimSpace(text), calls
}

// ---- decisions: the System One wire, on the engine's Decide ----
//
// POST /v1/systemone takes the request a Jev / TypeSafe client sends — one
// state and typed questions about it (noul: P(yes); choice: one of
// labelled options; score: a level on an ordered rubric) — and answers each
// from one prefill of the loaded model, the probability read off the
// next-token distribution over the option letters. Memdoor's decision
// model runs here, on this machine, with no key.

type wireQuestion struct {
	Type         string          `json:"type"`
	Instructions string          `json:"instructions"`
	Criteria     json.RawMessage `json:"criteria,omitempty"`
}

type wireRequest struct {
	Model     string                  `json:"model"`
	State     string                  `json:"state"`
	Questions map[string]wireQuestion `json:"questions"`
}

type wireAnswer struct {
	Type          string             `json:"type"`
	Noul          *float64           `json:"noul,omitempty"`
	Choice        string             `json:"choice,omitempty"`
	Score         *float64           `json:"score,omitempty"`
	Probabilities map[string]float64 `json:"probabilities,omitempty"`
	Confidence    float64            `json:"confidence"`
}

func (s *Server) handleDecide(w http.ResponseWriter, r *http.Request) {
	var req wireRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, 16<<20)).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "a JSON body: "+err.Error())
		return
	}
	if len(req.Questions) == 0 {
		writeError(w, http.StatusBadRequest, "questions is empty")
		return
	}
	ids := make([]string, 0, len(req.Questions))
	for id := range req.Questions {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	qs := make([]llm.Question, 0, len(ids))
	for _, id := range ids {
		q, err := toQuestion(req.Questions[id])
		if err != nil {
			writeError(w, http.StatusUnprocessableEntity, id+": "+err.Error())
			return
		}
		qs = append(qs, q)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	// A decision client names its own model (jev-latest): the loaded model
	// answers unless the name is one installed here.
	model := req.Model
	if !installed(model) {
		model = ""
	}
	if err := s.ensure(model); err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}
	ds, err := s.current.Decide(req.State, qs)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	answers := make(map[string]wireAnswer, len(ids))
	for i, id := range ids {
		answers[id] = toAnswer(req.Questions[id].Type, qs[i], ds[i])
	}
	writeJSON(w, http.StatusOK, map[string]any{"model": s.name, "answers": answers})
}

func installed(name string) bool {
	if name == "" {
		return false
	}
	list, _ := installedModels()
	for _, m := range list {
		if m.Name == name {
			return true
		}
	}
	return false
}

// toQuestion maps a wire question onto the engine's: a noul becomes a
// yes/no with the criteria spelled out, a score a rubric whose levels are
// numbered from zero (the client's "zero-based position"), a choice its
// labels in a fixed order with each description in the text.
func toQuestion(wq wireQuestion) (llm.Question, error) {
	text := strings.TrimSpace(wq.Instructions)
	switch wq.Type {
	case "noul", "boolean", "yesno":
		var crit map[string]string
		if len(wq.Criteria) > 0 && json.Unmarshal(wq.Criteria, &crit) == nil {
			if t := strings.TrimSpace(crit["true"]); t != "" {
				text += "\nYes means: " + t
			}
			if f := strings.TrimSpace(crit["false"]); f != "" {
				text += "\nNo means: " + f
			}
		}
		return llm.NewYesNo(text)
	case "score":
		var levels []string
		if err := json.Unmarshal(wq.Criteria, &levels); err != nil {
			return llm.Question{}, fmt.Errorf("a score question needs its levels, low to high")
		}
		values := make([]float64, len(levels))
		for i := range values {
			values[i] = float64(i)
		}
		return llm.NewScore(text, levels, values)
	case "choice", "":
		var crit map[string]string
		if err := json.Unmarshal(wq.Criteria, &crit); err != nil || len(crit) == 0 {
			var labels []string
			if json.Unmarshal(wq.Criteria, &labels) != nil || len(labels) == 0 {
				return llm.Question{}, fmt.Errorf("a choice question needs its options")
			}
			return llm.NewChoice(text, labels)
		}
		labels := make([]string, 0, len(crit))
		for l := range crit {
			labels = append(labels, l)
		}
		sort.Strings(labels)
		for _, l := range labels {
			if d := strings.TrimSpace(crit[l]); d != "" && d != l {
				text += "\n" + l + ": " + d
			}
		}
		return llm.NewChoice(text, labels)
	}
	return llm.Question{}, fmt.Errorf("unknown question type %q (noul, choice, score)", wq.Type)
}

func toAnswer(wireType string, q llm.Question, d llm.Decision) wireAnswer {
	a := wireAnswer{Type: wireType, Confidence: d.Confidence, Probabilities: map[string]float64{}}
	switch q.Kind {
	case llm.KindYesNo:
		p := d.Probs[0]
		a.Noul = &p
		a.Probabilities = nil
	case llm.KindScore:
		e := d.Expected
		a.Score = &e
		for i, p := range d.Probs {
			a.Probabilities[fmt.Sprint(i)] = p
		}
	default:
		a.Choice = d.Answer()
		for i, p := range d.Probs {
			a.Probabilities[d.Options[i]] = p
		}
	}
	return a
}

// ---- plumbing ----

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, code int, msg string) {
	writeJSON(w, code, map[string]any{"error": map[string]any{"message": msg, "type": "invalid_request_error", "code": code}})
}
