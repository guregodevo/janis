package llm

import (
	"os"
	"strconv"

	"github.com/guregodevo/janis/engine"
	"github.com/guregodevo/janis/reranker"
	"github.com/guregodevo/janis/safetensors"
	"github.com/guregodevo/janis/tokenizer"
)

// XLM-RoBERTa special token ids (shared by bge-m3 and bge-reranker).
const (
	tokBOS = 0 // <s> / [CLS]
	tokEOS = 2 // </s> / [SEP]

	// defaultRerankMaxTok is the cross-encoder pair length cap (incl. specials).
	// 512 is the standard reranker context; the page head carries the relevance
	// signal, so longer windows cost latency without precision.
	defaultRerankMaxTok = 512
)

// rerankMaxTokEnv returns the pair-length cap, MEMDOOR_RERANK_MAXTOK overriding
// the 512 default. Zero or invalid falls back to the default.
func rerankMaxTokEnv() int {
	if v := os.Getenv("MEMDOOR_RERANK_MAXTOK"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return defaultRerankMaxTok
}

// Reranker is a loaded bge cross-encoder. Score(query, passage) returns a
// relevance logit by jointly encoding the pair — far more discriminative than
// the bi-encoder cosine (Embedder), so it both reorders candidates and exposes
// "nothing actually answers this" gaps. Serialized through the shared MLX lock.
type Reranker struct {
	bk     engine.Backend
	model  *reranker.Model
	tok    *tokenizer.Tokenizer
	maxTok int
}

// OpenReranker loads bge-reranker-base from a local snapshot dir.
func OpenReranker(modelDir string) (*Reranker, error) {
	cfg, err := reranker.LoadConfig(modelDir)
	if err != nil {
		return nil, err
	}
	// Cap the pair length at the standard cross-encoder context (512), not the
	// model's full position budget: v2-m3 advertises 8194, but scoring 8k-token
	// passages is ~16x slower for no precision gain — relevance is decided by the
	// page head (title/summary/opening), and a uniform cap keeps models
	// comparable. Override with MEMDOOR_RERANK_MAXTOK. Room kept for <s> </s></s> </s>.
	maxTok := cfg.MaxPos - 4
	if cap := rerankMaxTokEnv(); cap > 0 && cap-4 < maxTok {
		maxTok = cap - 4
	}
	r := &Reranker{bk: newBackend(), maxTok: maxTok}
	b := r.bk
	st, err := safetensors.OpenModel(modelDir)
	if err != nil {
		return nil, err
	}
	if r.model, err = reranker.LoadModel(b, st, cfg); err != nil {
		return nil, err
	}
	pinAll(r.bk)
	if r.tok, err = tokenizer.New(modelDir); err != nil {
		return nil, err
	}
	return r, nil
}

// Score returns the relevance logit for (query, passage). Higher = more relevant.
func (r *Reranker) Score(query, passage string) float32 {
	mlxComputeMu.Lock()
	defer mlxComputeMu.Unlock()
	s := r.scoreLocked(query, passage)
	sweep(r.bk)
	return s
}

// scoreLocked assembles the XLM-RoBERTa pair (<s> query </s></s> passage </s>),
// truncating the passage to fit the model's position budget, and runs the
// cross-encoder. Caller holds mlxComputeMu.
func (r *Reranker) scoreLocked(query, passage string) float32 {
	q := r.tok.Encode(query)
	p := r.tok.Encode(passage)
	// 4 special tokens (<s>, </s>, </s>, </s>); keep the query whole and
	// truncate the passage — the query is short, the passage is the page.
	if budget := r.maxTok - len(q); len(p) > budget {
		if budget < 0 {
			budget = 0
		}
		p = p[:budget]
	}
	ids := make([]int32, 0, len(q)+len(p)+4)
	ids = append(ids, tokBOS)
	ids = append(ids, q...)
	ids = append(ids, tokEOS, tokEOS)
	ids = append(ids, p...)
	ids = append(ids, tokEOS)
	return r.model.Score(r.bk, ids)
}

// Close releases the model and backend.
func (r *Reranker) Close() {
	if r.tok != nil {
		r.tok.Close()
	}
	r.bk.Close()
}
