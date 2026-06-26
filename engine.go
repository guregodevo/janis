// Package llm is the embeddable high-level API: load a model once and chat with
// it in-process. This is what an app (e.g. memdoor's gateway) imports instead of
// spawning a separate inference server — no subprocess, no HTTP, no llama-server.
package llm

import (
	"fmt"

	"memdoor/llm/chat"
	"memdoor/llm/engine"
	"memdoor/llm/gemma"
	"memdoor/llm/qwen"
	"memdoor/llm/safetensors"
	"memdoor/llm/tokenizer"
)

// Message is re-exported so callers don't import the chat package directly.
type Message = chat.Message

// Options controls a chat request.
type Options struct {
	Temp      float32
	TopP      float32
	MaxTokens int
	Seed      uint64
	// Cancel, if set, is polled during generation; returning true stops early and
	// returns the partial reply. Used to interrupt an in-flight chat.
	Cancel func() bool
}

type sessionMaker interface {
	NewSession() *qwen.Session
}

// Engine is a loaded model + tokenizer with a persistent prefix-reusing session.
// It is safe for concurrent callers (generation is serialized internally).
type Engine struct {
	bk engine.Backend
	// session is the main conversational session (prefix-reused across an
	// agent's turns). utilSession serves one-off utility generations — query
	// reformulation, wiki summarize/synthesize — so they don't overwrite the
	// agent session's KV prefix and destroy its reuse (each utility call would
	// otherwise drop the next real prefill's longest-common-prefix to ~1 token).
	session     *qwen.Session
	utilSession *qwen.Session
	tok         *tokenizer.Tokenizer
	mtype       string
	stops       map[int32]bool
}

// Open loads the model from a local snapshot directory (config.json,
// model.safetensors, tokenizer.json).
func Open(modelDir string) (*Engine, error) {
	cfg, err := qwen.LoadConfig(modelDir)
	if err != nil {
		return nil, err
	}
	e := &Engine{bk: newBackend(), mtype: cfg.ModelType, stops: map[int32]bool{}}
	b := e.bk

	st, err := safetensors.OpenModel(modelDir)
	if err != nil {
		return nil, err
	}
	var sm sessionMaker
	switch cfg.ModelType {
	case "qwen2", "qwen3", "qwen3_moe", "llama":
		m, err := qwen.LoadModel(b, st, cfg)
		if err != nil {
			return nil, err
		}
		sm = m
	case "gemma2":
		m, err := gemma.LoadModel(b, st, cfg)
		if err != nil {
			return nil, err
		}
		sm = m
	default:
		return nil, fmt.Errorf("unsupported model_type %q", cfg.ModelType)
	}
	pinAll(e.bk) // protect the weights across requests (no-op on GC backends)
	e.session = sm.NewSession()
	e.utilSession = sm.NewSession()

	if e.tok, err = tokenizer.New(modelDir); err != nil {
		return nil, err
	}
	for _, id := range e.tok.Encode(chat.StopMarker(cfg.ModelType)) {
		e.stops[id] = true
	}
	for _, id := range cfg.EosTokens {
		e.stops[id] = true
	}
	return e, nil
}

// ModelType reports the loaded architecture (qwen2/qwen3/llama/gemma2).
func (e *Engine) ModelType() string { return e.mtype }

// Chat generates a full reply for the conversation, on the main prefix-reusing
// session.
func (e *Engine) Chat(msgs []Message, opts Options) string {
	return e.chat(msgs, opts, nil, e.session)
}

// ChatStream generates a reply, calling onDelta for each incremental text chunk;
// it also returns the full reply.
func (e *Engine) ChatStream(msgs []Message, opts Options, onDelta func(string)) string {
	return e.chat(msgs, opts, onDelta, e.session)
}

// ChatUtil generates a reply on the throwaway utility session — for one-off
// calls (query reformulation, wiki ops) that must NOT pollute the agent
// session's KV prefix and break its cross-turn reuse.
func (e *Engine) ChatUtil(msgs []Message, opts Options) string {
	return e.chat(msgs, opts, nil, e.utilSession)
}

func (e *Engine) chat(msgs []Message, opts Options, onDelta func(string), sess *qwen.Session) string {
	ids := e.tok.Encode(chat.ApplyTemplate(e.mtype, msgs))

	maxTok := opts.MaxTokens
	if maxTok <= 0 {
		maxTok = 512
	}
	topP := opts.TopP
	if topP <= 0 {
		topP = 1.0
	}
	p := qwen.SampleParams{Temp: opts.Temp, TopP: topP, Seed: opts.Seed, Stop: e.stops, Cancel: opts.Cancel}

	if onDelta != nil {
		var gen []int32
		var emitted string
		p.OnToken = func(tok int32) {
			gen = append(gen, tok)
			full := e.tok.Decode(gen)
			if len(full) > len(emitted) {
				onDelta(full[len(emitted):])
				emitted = full
			}
		}
	}

	mlxComputeMu.Lock()
	out := sess.Generate(e.bk, ids, maxTok, p)
	mlxComputeMu.Unlock()
	return e.tok.Decode(out)
}

// NumTokens returns the token count of text (for usage stats).
func (e *Engine) NumTokens(text string) int { return len(e.tok.Encode(text)) }

// Close releases the tokenizer and backend.
// Close releases the tokenizer and backend. It first acquires the shared MLX
// compute lock so it never frees the backend out from under an in-flight
// mlx_eval (which segfaults). If a generation is still running — e.g. the HTTP
// drain timed out before the request finished — it skips the explicit free and
// lets process exit reclaim the GPU buffers: bounded and safe, rather than
// either blocking shutdown or crashing.
func (e *Engine) Close() {
	if !mlxComputeMu.TryLock() {
		return
	}
	defer mlxComputeMu.Unlock()
	if e.tok != nil {
		e.tok.Close()
	}
	if e.bk != nil {
		e.bk.Close()
	}
}
