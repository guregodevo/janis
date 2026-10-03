package llm

import (
	"github.com/guregodevo/janis/bge"
	"github.com/guregodevo/janis/engine"
	"github.com/guregodevo/janis/safetensors"
	"github.com/guregodevo/janis/tokenizer"
)

// Embedder is a loaded embedding model (bge-m3 / XLM-RoBERTa) that turns text
// into an L2-normalized vector in-process — the embedding sibling of Engine, so
// the gateway needs no separate embedding server. Safe for concurrent callers
// (work is serialized internally; the MLX backend is single-threaded).
type Embedder struct {
	bk     engine.Backend
	model  *bge.Model
	tok    *tokenizer.Tokenizer
	dim    int
	maxTok int
}

// OpenEmbedder loads an embedding model from a local snapshot directory
// (config.json, model.safetensors, tokenizer.json).
func OpenEmbedder(modelDir string) (*Embedder, error) {
	cfg, err := bge.LoadConfig(modelDir)
	if err != nil {
		return nil, err
	}
	e := &Embedder{bk: newBackend(), dim: cfg.Hidden, maxTok: cfg.MaxPos - 2}
	b := e.bk

	st, err := safetensors.OpenModel(modelDir)
	if err != nil {
		return nil, err
	}
	if e.model, err = bge.LoadModel(b, st, cfg); err != nil {
		return nil, err
	}
	pinAll(e.bk) // protect the weights across requests

	if e.tok, err = tokenizer.New(modelDir); err != nil {
		return nil, err
	}
	return e, nil
}

// Dim returns the embedding dimension (e.g. 1024 for bge-m3).
func (e *Embedder) Dim() int { return e.dim }

// Embed returns the L2-normalized embedding for a single text.
func (e *Embedder) Embed(text string) []float32 {
	mlxComputeMu.Lock()
	defer mlxComputeMu.Unlock()
	return e.embedLocked(text)
}

// EmbedBatch embeds a slice of texts (serialized).
func (e *Embedder) EmbedBatch(texts []string) [][]float32 {
	mlxComputeMu.Lock()
	defer mlxComputeMu.Unlock()
	out := make([][]float32, len(texts))
	for i, t := range texts {
		out[i] = e.embedLocked(t)
	}
	return out
}

func (e *Embedder) embedLocked(text string) []float32 {
	ids := e.tok.EncodeSpecial(text)
	if len(ids) > e.maxTok {
		ids = ids[:e.maxTok]
	}
	vec := e.model.Embed(e.bk, ids) // Floats copies to host before we sweep
	sweep(e.bk)                     // free this call's transient arrays
	return vec
}

// Close releases the model and backend resources. Like Engine.Close it acquires
// the shared MLX compute lock first so the backend is never freed under an
// in-flight embed; if one is running it skips the free and lets process exit
// reclaim the buffers.
func (e *Embedder) Close() {
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
