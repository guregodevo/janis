package llm

import (
	"path/filepath"
	"sort"

	"memdoor/llm/engine"
	"memdoor/llm/mlxc"
	"memdoor/llm/reranker"
	"memdoor/llm/safetensors"
	"memdoor/llm/tokenizer"
)

// XLM-RoBERTa special token ids (shared by bge-m3 and bge-reranker).
const (
	tokBOS = 0 // <s> / [CLS]
	tokEOS = 2 // </s> / [SEP]
)

// Reranker is a loaded bge cross-encoder. Score(query, passage) returns a
// relevance logit by jointly encoding the pair — far more discriminative than
// the bi-encoder cosine (Embedder), so it both reorders candidates and exposes
// "nothing actually answers this" gaps. Serialized through the shared MLX lock.
type Reranker struct {
	bk     *mlxc.Backend
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
	r := &Reranker{bk: mlxc.New(), maxTok: cfg.MaxPos - 4} // room for <s> </s></s> </s>
	var b engine.Backend = r.bk
	st, err := safetensors.Open(filepath.Join(modelDir, "model.safetensors"))
	if err != nil {
		return nil, err
	}
	if r.model, err = reranker.LoadModel(b, st, cfg); err != nil {
		return nil, err
	}
	r.bk.PinAll()
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
	r.bk.Sweep()
	return s
}

// Rank scores every passage against the query and returns indices sorted best
// first, paired with their scores.
func (r *Reranker) Rank(query string, passages []string) (order []int, scores []float32) {
	mlxComputeMu.Lock()
	defer mlxComputeMu.Unlock()
	scores = make([]float32, len(passages))
	order = make([]int, len(passages))
	for i, p := range passages {
		scores[i] = r.scoreLocked(query, p)
		order[i] = i
		r.bk.Sweep()
	}
	sort.SliceStable(order, func(a, b int) bool { return scores[order[a]] > scores[order[b]] })
	return order, scores
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
