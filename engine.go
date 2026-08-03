// Package llm is the embeddable high-level API: load a model once and chat with
// it in-process. This is what an app (e.g. memdoor's gateway) imports instead of
// spawning a separate inference server — no subprocess, no HTTP, no llama-server.
package llm

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"memdoor/llm/chat"
	"memdoor/llm/engine"
	"memdoor/llm/gemma"
	"memdoor/llm/grammar"
	"memdoor/llm/qwen"
	"memdoor/llm/safetensors"
	"memdoor/llm/tokenizer"
)

// Repetition-penalty defaults for local decoding, tunable via env. 1.1 over the
// last 64 tokens is the llama.cpp default — mild enough not to distort code, strong
// enough to break a runaway repeat loop. MEMDOOR_REPEAT_PENALTY=1 disables it.
var (
	repeatPenalty = envFloat32("MEMDOOR_REPEAT_PENALTY", 1.1)
	repeatLastN   = envInt("MEMDOOR_REPEAT_LAST_N", 64)
)

func envFloat32(key string, def float32) float32 {
	if v := os.Getenv(key); v != "" {
		if f, err := strconv.ParseFloat(v, 32); err == nil {
			return float32(f)
		}
	}
	return def
}

func envInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

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
	// ToolCallGrammar constrains decoding to the <tool_call>{json}</tool_call>
	// envelope (grammar-constrained decoding): plain text stays free, but once the
	// model opens a tool call the JSON is forced well-formed and correctly closed.
	// Set it for tool-using turns so the provider parser never sees malformed calls.
	ToolCallGrammar bool
	// ForceToolCall additionally REQUIRES the turn to open a <tool_call> — the model
	// cannot answer in prose or a ```json fence. Only meaningful with ToolCallGrammar
	// and tools present; set it for a turn where a tool call is expected.
	ForceToolCall bool
	// OnToken, if set, observes every committed token (id and its decoded text,
	// which is empty for special tokens). Diagnostic hook: callers use it to
	// reconstruct WHAT a model emitted when the decoded reply is empty or
	// mangled — token-level ground truth the text reply cannot show.
	OnToken func(id int32, text string)
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
	vocab       int
	modelDir    string            // snapshot dir the model loaded from — KV-snapshot identity
	experts     *qwen.ExpertStore // non-nil for MoE models (streamed experts)
	// tokTable is a lazily-built id→text table for grammar-constrained decoding:
	// the sampler tests the grammar against many candidate tokens each step, so it
	// needs O(1) token text, not a per-candidate cgo tokenizer call. Built once,
	// only when a grammar is first used (non-tool chats never pay for it).
	tokTable     []string
	tokTableOnce sync.Once
}

// Open loads the model from a local snapshot directory (config.json,
// model.safetensors, tokenizer.json).
func Open(modelDir string) (*Engine, error) {
	cfg, err := qwen.LoadConfig(modelDir)
	if err != nil {
		return nil, err
	}
	e := &Engine{bk: newBackend(), mtype: cfg.ModelType, stops: map[int32]bool{}, vocab: cfg.Vocab, modelDir: modelDir}
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
		e.experts = m.Experts
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

// ExpertIOStats reports cumulative streamed-expert reads for MoE models:
// total bytes fetched and wall time blocked awaiting them. ok is false for
// dense models, which have no streamed experts.
func (e *Engine) ExpertIOStats() (bytes int64, dur time.Duration, ok bool) {
	if e.experts == nil {
		return 0, 0, false
	}
	bytes, dur = e.experts.IOStats()
	return bytes, dur, true
}

// tokenText returns the decoded text of a single token via a table built once on
// first use. Grammar-constrained decoding vets many candidate tokens per step, so
// this must be an O(1) lookup rather than a per-candidate tokenizer (cgo) call.
func (e *Engine) tokenText(id int32) string {
	e.tokTableOnce.Do(func() {
		e.tokTable = make([]string, e.vocab)
		for i := 0; i < e.vocab; i++ {
			e.tokTable[i] = e.tok.Decode([]int32{int32(i)})
		}
	})
	if id < 0 || int(id) >= len(e.tokTable) {
		return ""
	}
	return e.tokTable[id]
}

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
	p := qwen.SampleParams{
		Temp: opts.Temp, TopP: topP, Seed: opts.Seed, Stop: e.stops, Cancel: opts.Cancel,
		// Repetition penalty is on by default so a local model can't lock into a
		// decode loop on free text (the classic "of of of" / echo-forever failure).
		// Tunable via env; 1.0 disables. Skipped automatically during grammar-
		// constrained tool calls (see sampleToken).
		RepeatPenalty: repeatPenalty,
		RepeatLastN:   repeatLastN,
	}
	if opts.ToolCallGrammar {
		// A fresh grammar per generation (it carries per-turn state). It decodes
		// candidate tokens via the O(1) token table (built once) and constrains the
		// tool-call JSON to be well-formed. ForceToolCall additionally requires the
		// turn to OPEN a call (the model can't drift into malformed free-text JSON).
		if opts.ForceToolCall {
			p.Grammar = grammar.NewToolCallForced(e.tokenText)
		} else {
			p.Grammar = grammar.NewToolCall(e.tokenText)
		}
	}

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

	if opts.OnToken != nil {
		obs, prev := opts.OnToken, p.OnToken
		p.OnToken = func(tok int32) {
			obs(tok, e.tokenText(tok))
			if prev != nil {
				prev(tok)
			}
		}
	}

	if opts.ToolCallGrammar {
		// Whitespace-runaway guard: grammar-forced decoding on some models
		// (live: Qwen3-Coder-30B) emits a complete tool call and then endless
		// whitespace instead of the closing tag, burning the whole token budget
		// in silence. 16 consecutive whitespace-only tokens is pathological in
		// any real output — stop there. Free-text turns are unaffected (the
		// repetition penalty covers those).
		var wsRun int
		prevOnToken := p.OnToken
		p.OnToken = func(tok int32) {
			if strings.TrimSpace(e.tokenText(tok)) == "" {
				wsRun++
			} else {
				wsRun = 0
			}
			if prevOnToken != nil {
				prevOnToken(tok)
			}
		}
		prevCancel := p.Cancel
		p.Cancel = func() bool {
			return wsRun >= 16 || (prevCancel != nil && prevCancel())
		}
	}

	mlxComputeMu.Lock()
	// Generate-nothing calls (KV warmup, MaxTokens<=1) always prefill chunked:
	// chunk=128 fetches each MoE layer's expert union once per chunk (~30%
	// faster prefill, measured 2.5 vs 1.9 tok/s cold on Qwen3-30B-A3B), and
	// the reason chunking is NOT the general default — it leaves the decode
	// slot cache unwarmed (0.10 tok/s decode measured after a from-scratch
	// chunked prefill) — cannot bite when nothing meaningful is decoded.
	// Dense models ignore PrefillChunk overrides above the global chunk size.
	restoreChunk := false
	if maxTok <= 1 && sess.PrefillChunk == 1 {
		sess.PrefillChunk = 128
		restoreChunk = true
	}
	out := sess.Generate(e.bk, ids, maxTok, p)
	if restoreChunk {
		sess.PrefillChunk = 1
	}
	mlxComputeMu.Unlock()
	return e.tok.Decode(out)
}

// NumTokens returns the token count of text (for usage stats).
func (e *Engine) NumTokens(text string) int { return len(e.tok.Encode(text)) }

// SaveMainSession snapshots the main session's materialized KV prefix to path
// (EXPERT_STREAMING.md step 0: a warmed prefix is deterministic in (model,
// prompt bytes) — recomputing it on a streamed-MoE model costs 20-60 min per
// restart; loading it costs seconds). Takes the compute lock: downloading
// cache tensors while a generation mutates them would serialize garbage.
// Returns the number of persisted prefix tokens.
func (e *Engine) SaveMainSession(path string) (int, error) {
	mlxComputeMu.Lock()
	defer mlxComputeMu.Unlock()
	n := len(e.session.IDs)
	if err := e.session.Save(e.bk, path, e.modelDir); err != nil {
		return 0, err
	}
	if c := e.session.Caches[0].Len(); c < n {
		n = c
	}
	return n, nil
}

// LoadMainSession restores a SaveMainSession snapshot into the (fresh) main
// session. Safe to call only between Open and the first Chat. A stale or
// mismatched snapshot returns an error and leaves the session untouched —
// the caller just proceeds cold. Returns the number of restored prefix tokens.
func (e *Engine) LoadMainSession(path string) (int, error) {
	mlxComputeMu.Lock()
	defer mlxComputeMu.Unlock()
	return e.session.Restore(e.bk, path, e.modelDir)
}

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
