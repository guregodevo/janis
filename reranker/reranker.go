// Package reranker implements the BGE cross-encoder reranker (bge-reranker-base,
// an XLM-RoBERTa sequence classifier): it jointly encodes a (query, passage)
// pair and emits a single relevance logit. Unlike the bge-m3 bi-encoder (which
// embeds query and passage independently, then cosines them — too compressed to
// separate corpus gaps from real matches), the cross-encoder attends across the
// pair, so its score is far more discriminative. Used to rerank bi-encoder
// candidates and to detect "no page actually answers this" gaps.
//
// Same XLM-RoBERTa stack as package bge (word/position/token_type embeddings,
// bidirectional encoder layers, exact-GELU FFN) — weights carry a "roberta."
// prefix and the model pools the [CLS] token through a dense+tanh+out_proj head
// instead of mean-pooling.
package reranker

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"

	"github.com/guregodevo/janis/engine"
	"github.com/guregodevo/janis/qwen"
	"github.com/guregodevo/janis/safetensors"
)

// Config holds the XLM-RoBERTa dimensions.
type Config struct {
	Hidden, Layers, Heads, Intermediate, Vocab, MaxPos int
	Eps                                                float32
	PadID                                              int
}

func LoadConfig(dir string) (Config, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return Config{}, err
	}
	var r struct {
		HiddenSize            int     `json:"hidden_size"`
		NumHiddenLayers       int     `json:"num_hidden_layers"`
		NumAttentionHeads     int     `json:"num_attention_heads"`
		IntermediateSize      int     `json:"intermediate_size"`
		VocabSize             int     `json:"vocab_size"`
		MaxPositionEmbeddings int     `json:"max_position_embeddings"`
		LayerNormEps          float64 `json:"layer_norm_eps"`
		PadTokenID            int     `json:"pad_token_id"`
	}
	if err := json.Unmarshal(raw, &r); err != nil {
		return Config{}, err
	}
	return Config{
		Hidden: r.HiddenSize, Layers: r.NumHiddenLayers, Heads: r.NumAttentionHeads,
		Intermediate: r.IntermediateSize, Vocab: r.VocabSize, MaxPos: r.MaxPositionEmbeddings,
		Eps: float32(r.LayerNormEps), PadID: r.PadTokenID,
	}, nil
}

// linear: y = x @ Wᵀ + b (weight stored [out,in], transposed once at load).
type linear struct{ wt, bias engine.Tensor }

func loadLinear(b engine.Backend, st *safetensors.File, prefix string) (*linear, error) {
	w, err := qwen.LoadTensor(b, st, prefix+".weight")
	if err != nil {
		return nil, err
	}
	bias, err := qwen.LoadTensor(b, st, prefix+".bias")
	if err != nil {
		return nil, err
	}
	return &linear{wt: b.Transpose(w, 1, 0), bias: bias}, nil
}

func (l *linear) forward(b engine.Backend, x engine.Tensor) engine.Tensor {
	return b.Add(b.MatMul(x, l.wt), l.bias)
}

func layerNormPair(b engine.Backend, st *safetensors.File, prefix string) (w, bias engine.Tensor, err error) {
	if w, err = qwen.LoadTensor(b, st, prefix+".weight"); err != nil {
		return
	}
	bias, err = qwen.LoadTensor(b, st, prefix+".bias")
	return
}

type embeddings struct {
	word, position, tokenType engine.Tensor
	lnW, lnB                  engine.Tensor
	eps                       float32
	padID                     int
}

func (e *embeddings) forward(b engine.Backend, ids []int32) engine.Tensor {
	L := len(ids)
	idsT := b.FromInt32(ids, 1, L)
	pos := make([]int32, L)
	for i := range pos {
		pos[i] = int32(e.padID + 1 + i) // RoBERTa position offset (no padding)
	}
	posT := b.FromInt32(pos, 1, L)
	we := b.TakeAxis(e.word, idsT, 0)
	pe := b.TakeAxis(e.position, posT, 0)
	tte := b.Slice(e.tokenType, 0, 0, 1) // token_type[0], broadcasts over L
	emb := b.Add(b.Add(we, pe), tte)
	return b.LayerNorm(emb, e.lnW, e.lnB, e.eps)
}

type encoderLayer struct {
	q, k, v, attnOut *linear
	attnLNW, attnLNB engine.Tensor
	inter, out       *linear
	outLNW, outLNB   engine.Tensor
	nHeads, headDim  int
	eps, scale       float32
}

func (l *encoderLayer) forward(b engine.Backend, x engine.Tensor) engine.Tensor {
	B, L := x.Shape()[0], x.Shape()[1]
	q := b.Transpose(b.Reshape(l.q.forward(b, x), B, L, l.nHeads, l.headDim), 0, 2, 1, 3)
	k := b.Transpose(b.Reshape(l.k.forward(b, x), B, L, l.nHeads, l.headDim), 0, 2, 1, 3)
	v := b.Transpose(b.Reshape(l.v.forward(b, x), B, L, l.nHeads, l.headDim), 0, 2, 1, 3)
	attn := b.SDPA(q, k, v, l.scale, false) // bidirectional
	attn = b.Reshape(b.Transpose(attn, 0, 2, 1, 3), B, L, l.nHeads*l.headDim)
	attn = l.attnOut.forward(b, attn)
	h := b.LayerNorm(b.Add(x, attn), l.attnLNW, l.attnLNB, l.eps)
	ff := l.out.forward(b, geluExact(b, l.inter.forward(b, h)))
	return b.LayerNorm(b.Add(h, ff), l.outLNW, l.outLNB, l.eps)
}

func geluExact(b engine.Backend, x engine.Tensor) engine.Tensor {
	const invSqrt2 = 0.7071067811865476
	return b.ScalarMul(b.Mul(x, b.ScalarAdd(b.Erf(b.ScalarMul(x, invSqrt2)), 1.0)), 0.5)
}

// Model is a bge cross-encoder reranker.
type Model struct {
	emb      *embeddings
	layers   []*encoderLayer
	clsDense *linear // classifier.dense  (hidden -> hidden), then tanh
	clsOut   *linear // classifier.out_proj (hidden -> 1)
	cfg      Config
}

func LoadModel(b engine.Backend, st *safetensors.File, cfg Config) (*Model, error) {
	const p = "roberta."
	e := &embeddings{eps: cfg.Eps, padID: cfg.PadID}
	var err error
	if e.word, err = qwen.LoadTensor(b, st, p+"embeddings.word_embeddings.weight"); err != nil {
		return nil, err
	}
	if e.position, err = qwen.LoadTensor(b, st, p+"embeddings.position_embeddings.weight"); err != nil {
		return nil, err
	}
	if e.tokenType, err = qwen.LoadTensor(b, st, p+"embeddings.token_type_embeddings.weight"); err != nil {
		return nil, err
	}
	if e.lnW, e.lnB, err = layerNormPair(b, st, p+"embeddings.LayerNorm"); err != nil {
		return nil, err
	}

	layers := make([]*encoderLayer, cfg.Layers)
	headDim := cfg.Hidden / cfg.Heads
	for i := range layers {
		lp := p + "encoder.layer." + itoa(i)
		l := &encoderLayer{nHeads: cfg.Heads, headDim: headDim, eps: cfg.Eps,
			scale: float32(1.0 / math.Sqrt(float64(headDim)))}
		if l.q, err = loadLinear(b, st, lp+".attention.self.query"); err != nil {
			return nil, err
		}
		if l.k, err = loadLinear(b, st, lp+".attention.self.key"); err != nil {
			return nil, err
		}
		if l.v, err = loadLinear(b, st, lp+".attention.self.value"); err != nil {
			return nil, err
		}
		if l.attnOut, err = loadLinear(b, st, lp+".attention.output.dense"); err != nil {
			return nil, err
		}
		if l.attnLNW, l.attnLNB, err = layerNormPair(b, st, lp+".attention.output.LayerNorm"); err != nil {
			return nil, err
		}
		if l.inter, err = loadLinear(b, st, lp+".intermediate.dense"); err != nil {
			return nil, err
		}
		if l.out, err = loadLinear(b, st, lp+".output.dense"); err != nil {
			return nil, err
		}
		if l.outLNW, l.outLNB, err = layerNormPair(b, st, lp+".output.LayerNorm"); err != nil {
			return nil, err
		}
		layers[i] = l
	}

	m := &Model{emb: e, layers: layers, cfg: cfg}
	if m.clsDense, err = loadLinear(b, st, "classifier.dense"); err != nil {
		return nil, err
	}
	if m.clsOut, err = loadLinear(b, st, "classifier.out_proj"); err != nil {
		return nil, err
	}
	return m, nil
}

// Score returns the relevance logit for the tokenized (query+passage) pair.
// Higher = more relevant. ids must already carry the special tokens
// (<s> query </s></s> passage </s>).
func (m *Model) Score(b engine.Backend, ids []int32) float32 {
	h := m.emb.forward(b, ids)
	for _, l := range m.layers {
		h = l.forward(b, h)
	}
	// XLMRobertaClassificationHead: take the <s>/[CLS] token (position 0),
	// dense -> tanh -> out_proj -> single logit.
	cls := b.Slice(h, 1, 0, 1)              // [1, 1, hidden]
	cls = b.Reshape(cls, 1, m.cfg.Hidden)   // [1, hidden]
	d := b.Tanh(m.clsDense.forward(b, cls)) // dense + tanh
	logit := m.clsOut.forward(b, d)         // -> [1, 1]
	return b.Floats(logit)[0]
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var buf [4]byte
	n := len(buf)
	for i > 0 {
		n--
		buf[n] = byte('0' + i%10)
		i /= 10
	}
	return string(buf[n:])
}
