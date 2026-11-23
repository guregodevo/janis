// Package llm is the embeddable high-level API: load a model once and chat with
// it in-process. This is what an app (e.g. memdoor's gateway) imports instead of
// spawning a separate inference server — no subprocess, no HTTP, no llama-server.
package llm

import (
	"crypto/sha256"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/guregodevo/janis/chat"
	"github.com/guregodevo/janis/engine"
	"github.com/guregodevo/janis/gemma"
	"github.com/guregodevo/janis/grammar"
	"github.com/guregodevo/janis/hostfit"
	"github.com/guregodevo/janis/inkling2"
	"github.com/guregodevo/janis/qwen"
	"github.com/guregodevo/janis/qwen35"
	"github.com/guregodevo/janis/safetensors"
	"github.com/guregodevo/janis/tokenizer"
)

// Repetition-penalty defaults for local decoding, tunable via env. 1.1 over the
// last 64 tokens is the llama.cpp default — mild enough not to distort code, strong
// enough to break a runaway repeat loop. JANIS_REPEAT_PENALTY=1 disables it.
var (
	repeatPenalty = envFloat32("JANIS_REPEAT_PENALTY", 1.1)
	repeatLastN   = envInt("JANIS_REPEAT_LAST_N", 64)
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
	// OnReasoning, if set, hears the model's thinking as it comes, for a
	// family that writes it apart from the reply (Inkling 2). The reply
	// never carries it.
	OnReasoning func(delta string)
	// ReasoningEffort names how much to think, for a family that takes it
	// (Inkling 2: none, minimal, low, medium, high, xhigh, max). "" is the
	// family's default.
	ReasoningEffort string
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
	// q35 is the loaded Qwen3.5 text stack (nil for other families).
	q35 *qwen35.Model

	// slots are the sessions BeginChat decodes on together (engine_batch.go),
	// made on the first BeginChat: slotBusy marks a slot holding a reply,
	// slotUsed when its reply began (slotClock's count), for least recently used.
	slots       []*qwen.Session
	slotBusy    []bool
	slotUsed    []int
	slotClock   int
	slotDecoder *qwen35.BatchDecoder
	slotStep    []int       // the slots of the last StepChats, in its order
	slotStats   *batchStats // JANIS_BATCH_STATS=1, else nil
	// ink2 is the loaded Inkling 2 text stack (nil for other families).
	ink2 *inkling2.Model
	// vision is set for a model with a vision tower (nil otherwise).
	vision *visionState
	// ink holds Inkling 2's marker ids: its reply is a run of messages read
	// by chat.InklingReply, not one text. Nil for the other families.
	ink *chat.InklingMarks

	decideState
	editState
}

// Open loads the model from a local snapshot directory (config.json,
// model.safetensors, tokenizer.json).
func Open(modelDir string) (*Engine, error) {
	// Inkling 2's config nests under text_config in a layout of its own; the
	// shared loader refuses it, so it is read first.
	var cfg qwen.Config
	var ink inkling2.Config
	var err error
	if inkling2.Handles(inkling2.ModelType(modelDir)) {
		if ink, err = inkling2.LoadConfig(modelDir); err != nil {
			return nil, err
		}
		cfg = qwen.Config{ModelType: ink.ModelType, Vocab: ink.UnpaddedVocab, EosTokens: ink.EosTokens}
	} else if cfg, err = qwen.LoadConfig(modelDir); err != nil {
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
	case "qwen3_5":
		m, err := qwen35.LoadModel(b, st, cfg)
		if err != nil {
			return nil, err
		}
		sm = m
		e.q35 = m
	case inkling2.TypeLoop:
		ink.StreamExperts = streamExperts(inkling2.ExpertBytes(st))
		ink.ExpertCacheBytes = expertCacheBytes()
		m, err := inkling2.LoadModel(b, st, ink)
		if err != nil {
			return nil, err
		}
		// The draft head is used when asked for: its checkpoint is 0.64 GB
		// more in memory.
		if os.Getenv("JANIS_DRAFT") == "1" {
			dcfg, ok, err := inkling2.LoadDraftConfig(modelDir, ink)
			if err != nil {
				return nil, err
			}
			if !ok {
				return nil, fmt.Errorf("JANIS_DRAFT=1: %s has no %s or no mtp_config", modelDir, inkling2.DraftFile)
			}
			if err := m.LoadDraft(modelDir, dcfg); err != nil {
				return nil, fmt.Errorf("the draft head: %w", err)
			}
			m.SetDraftCount(envInt("JANIS_DRAFT_TOKENS", 3))
		}
		sm = m
		e.ink2 = m
		if vcfg, ok, err := inkling2.LoadVisionConfig(modelDir); err != nil {
			return nil, err
		} else if ok {
			e.vision = &visionState{cfg: vcfg, st: st, seen: map[*qwen.Session]string{}}
		}
	default:
		return nil, fmt.Errorf("unsupported model_type %q", cfg.ModelType)
	}
	e.st = st
	if pl, ok := sm.(paramLister); ok {
		e.indexParams(pl.Params())
	}
	pinAll(e.bk) // protect the weights across requests (no-op on GC backends)
	e.session = sm.NewSession()
	e.utilSession = sm.NewSession()
	e.decideSession = sm.NewSession()

	if e.tok, err = tokenizer.New(modelDir); err != nil {
		return nil, err
	}
	for _, marker := range append([]string{chat.StopMarker(cfg.ModelType)}, chat.ControlMarkers(cfg.ModelType)...) {
		for _, id := range e.tok.Encode(marker) {
			e.stops[id] = true
		}
	}
	for _, id := range cfg.EosTokens {
		e.stops[id] = true
	}
	if inkling2.Handles(cfg.ModelType) {
		var m chat.InklingMarks
		for marker, id := range map[string]*int32{
			chat.InklingModel: &m.Model, chat.InklingThinking: &m.Thinking, chat.InklingText: &m.Text,
			chat.InklingCall: &m.Call, chat.InklingEndMessage: &m.End,
		} {
			ids := e.tok.Encode(marker)
			if len(ids) != 1 {
				return nil, fmt.Errorf("%s: the tokenizer has no single token for %s", modelDir, marker)
			}
			*id = ids[0]
		}
		e.ink = &m
	}
	e.padForSlots()
	return e, nil
}

// reply is the generated tokens as text, and the thinking apart for a
// family that writes it as a message of its own.
func (e *Engine) reply(ids []int32) (text, reasoning string) {
	if e.ink == nil {
		return e.tok.Decode(ids), ""
	}
	return chat.InklingReply(ids, *e.ink, e.tok.Decode)
}

// ModelType reports the loaded architecture (qwen2/qwen3/qwen3_5/llama/gemma2/inkling2_loop).
func (e *Engine) ModelType() string { return e.mtype }

// ModelDir is the snapshot directory this engine's model loaded from — the
// identity the KV-snapshot files are keyed by.
func (e *Engine) ModelDir() string { return e.modelDir }

// ExpertIOStats reports cumulative streamed-expert reads for MoE models:
// total bytes fetched and wall time blocked awaiting them. ok is false for
// dense models, which have no streamed experts.
func (e *Engine) ExpertIOStats() (bytes int64, dur time.Duration, ok bool) {
	if e.ink2 != nil && e.ink2.Cfg.StreamExperts {
		bytes, dur = e.ink2.ExpertIO()
		return bytes, dur, true
	}
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

// sampleParams is a reply's sampling and its token budget, from opts: the
// repetition penalty, the tool-call grammar, onDelta and opts.OnToken on
// each committed token, the whitespace-runaway guard.
func (e *Engine) sampleParams(opts Options, onDelta func(string)) (qwen.SampleParams, int) {
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
	if e.ink != nil {
		// Inkling 2 writes a call as a message of its own, which the reply
		// reader turns into the <tool_call> block: the grammar of that block
		// would forbid what the model was trained to write.
		opts.ToolCallGrammar = false
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

	if onDelta != nil || opts.OnReasoning != nil {
		var gen []int32
		var emitted, thought string
		p.OnToken = func(tok int32) {
			gen = append(gen, tok)
			full, thinking := e.reply(gen)
			if onDelta != nil && len(full) > len(emitted) {
				onDelta(full[len(emitted):])
				emitted = full
			}
			if opts.OnReasoning != nil && len(thinking) > len(thought) {
				opts.OnReasoning(thinking[len(thought):])
				thought = thinking
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
	return p, maxTok
}

func (e *Engine) chat(msgs []Message, opts Options, onDelta func(string), sess *qwen.Session) string {
	ids := e.tok.Encode(chat.ApplyTemplateEffort(e.mtype, msgs, opts.ReasoningEffort))
	var images [][]byte
	for _, m := range msgs {
		images = append(images, m.Images...)
	}

	p, maxTok := e.sampleParams(opts, onDelta)

	mlxComputeMu.Lock()
	if fixedRows && e.slots != nil && sess == e.session {
		e.settleSlots() // slot 0 decodes alone: every row back from the batch
	}
	if e.vision != nil {
		ids = e.see(sess, ids, images)
	}
	// Generate-nothing calls (KV warmup, MaxTokens<=1) always prefill chunked:
	// chunk=128 fetches each MoE layer's expert union once per chunk (~30%
	// faster prefill, measured 2.5 vs 1.9 tok/s cold on Qwen3-30B-A3B), and
	// the reason chunking is NOT the general default — it leaves the decode
	// slot cache unwarmed (0.10 tok/s decode measured after a from-scratch
	// chunked prefill) — cannot bite when nothing meaningful is decoded.
	// Dense models ignore PrefillChunk overrides above the global chunk size.
	var out []int32
	if maxTok <= 1 {
		withGenerateNothingChunking(sess, func() { out = sess.Generate(e.bk, ids, maxTok, p) })
	} else {
		out = sess.Generate(e.bk, ids, maxTok, p)
	}
	if e.vision != nil {
		// The next prompt read may be another session's, with no pictures.
		e.ink2.SetImages(nil, nil)
	}
	mlxComputeMu.Unlock()
	text, _ := e.reply(out)
	return text
}

// NumTokens returns the token count of text (for usage stats).
func (e *Engine) NumTokens(text string) int { return len(e.tok.Encode(text)) }

// SaveMainSession snapshots the main session's materialized KV prefix to path
// (a warmed prefix is deterministic in (model,
// prompt bytes) — recomputing it on a streamed-MoE model costs 20-60 min per
// restart; loading it costs seconds). Takes the compute lock: downloading
// cache tensors while a generation mutates them would serialize garbage.
// Returns the number of persisted prefix tokens.
func (e *Engine) SaveMainSession(path string) (int, error) {
	mlxComputeMu.Lock()
	defer mlxComputeMu.Unlock()
	n := len(e.session.IDs)
	if err := e.session.Save(e.bk, path, e.modelID()); err != nil {
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
	return e.session.Restore(e.bk, path, e.modelID())
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

// streamExperts decides whether a model's experts stay on disk: when they
// would take more than 60% of the host's memory, which leaves too little
// for the rest of the model, its caches and the system (the 15.8 GB of
// Inkling 2 Mini on a 16 GB Mac swapped 10 GB and answered nothing in nine
// minutes). JANIS_MOE_RESIDENT=1 keeps them on the device, JANIS_MOE_STREAM=1
// on disk, whatever the host.
func streamExperts(expertBytes int64) bool {
	switch {
	case os.Getenv("JANIS_MOE_RESIDENT") == "1":
		return false
	case os.Getenv("JANIS_MOE_STREAM") == "1":
		return true
	}
	host, err := hostfit.Detect()
	if err != nil {
		return false
	}
	return expertBytes > int64(host.RAMGiB)<<30*6/10
}

// expertCacheBytes bounds the streamed experts kept on the device
// (JANIS_MOE_CACHE_MB, as for the Qwen3 MoE store): three eighths of the
// host's memory by default. The cache and the file cache share that
// memory, and a miss is cheap only from the file cache. Measured on
// Inkling-2-Mini-4bit, a 16 GB Mac, one 71-step request: 30 s with no
// cache, 20 s at 1.5 GB, 26 s at 4 GB, 19 s at 6 GB, 17 s at 8 GB with
// 1 GB more swap.
func expertCacheBytes() int64 {
	if mb, _ := strconv.Atoi(os.Getenv("JANIS_MOE_CACHE_MB")); mb > 0 {
		return int64(mb) << 20
	}
	host, err := hostfit.Detect()
	if err != nil {
		return 2500 << 20
	}
	return int64(host.RAMGiB) << 30 * 3 / 8
}

// visionState is a model's vision tower, loaded when the first image
// comes, and what each session's prompt last held.
type visionState struct {
	cfg   inkling2.VisionConfig
	st    *safetensors.File
	tower *inkling2.Vision
	// seen is, by session, the digests of the images of its last prompt.
	// Two prompts with the same tokens and other pictures share no prefix.
	seen map[*qwen.Session]string
}

// CheckImages says why the messages' images cannot be read by this model,
// nil when they can (or there are none).
func (e *Engine) CheckImages(msgs []Message) error {
	for _, m := range msgs {
		for i, img := range m.Images {
			if e.vision == nil {
				return fmt.Errorf("%s takes no images", e.mtype)
			}
			// A tower that does not load is said here, not found out as a
			// reply that has seen no picture.
			mlxComputeMu.Lock()
			err := e.loadTower()
			mlxComputeMu.Unlock()
			if err != nil {
				return fmt.Errorf("the vision tower does not load: %w", err)
			}
			if _, _, _, err := inkling2.PrepareImage(img, e.vision.cfg.Block()); err != nil {
				return fmt.Errorf("image %d of a %s message: %w", i+1, m.Role, err)
			}
		}
	}
	return nil
}

// see encodes the prompt's images and tells the model where they go: each
// image token of ids becomes one token for every block of its picture, and
// the model reads the picture's rows at those places. An image that cannot
// be read loses its token. The session forgets its prefix when the
// pictures before it changed. Called with the compute lock held.
func (e *Engine) see(sess *qwen.Session, ids []int32, images [][]byte) []int32 {
	v := e.vision
	key := ""
	for _, img := range images {
		key += fmt.Sprintf("%x.", sha256.Sum256(img))
	}
	if prev := v.seen[sess]; !strings.HasPrefix(key, prev) {
		sess.IDs = nil
	}
	v.seen[sess] = key
	if len(images) == 0 {
		e.ink2.SetImages(nil, nil)
		return ids
	}
	if err := e.loadTower(); err != nil {
		e.ink2.SetImages(nil, nil)
		return ids
	}
	out := make([]int32, 0, len(ids))
	var rows engine.Tensor
	var at []int
	next := 0
	for _, id := range ids {
		if id != v.cfg.ImageToken {
			out = append(out, id)
			continue
		}
		if next >= len(images) {
			continue
		}
		img := images[next]
		next++
		pixels, h, w, err := inkling2.PrepareImage(img, v.cfg.Block())
		if err != nil {
			continue
		}
		r, err := v.tower.Encode(pixels, h, w)
		if err != nil {
			continue
		}
		for i := 0; i < r.Shape()[0]; i++ {
			at = append(at, len(out))
			out = append(out, id)
		}
		if rows == nil {
			rows = r
		} else {
			rows = e.bk.Concat(rows, r, 0)
		}
	}
	if rows != nil {
		e.bk.Eval(rows)
	}
	e.ink2.SetImages(rows, at)
	return out
}

// loadTower loads the vision tower once. Called with the compute lock held.
func (e *Engine) loadTower() error {
	v := e.vision
	if v.tower != nil {
		return nil
	}
	tower, err := e.ink2.LoadVision(v.st, v.cfg)
	if err != nil {
		return err
	}
	if sw, ok := e.bk.(engine.Sweeper); ok {
		sw.PinAll() // the tower's weights, like the model's at the load
	}
	v.tower = tower
	return nil
}

// DraftStats is how the main session's draft head has done: the tokens it
// proposed and the ones the model kept. Zero without a draft head.
func (e *Engine) DraftStats() (drafted, accepted int) {
	return e.session.Drafted, e.session.Accepted
}
