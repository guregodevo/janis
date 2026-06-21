// Package bge implements the BGE-M3 embedding model (an XLM-RoBERTa encoder)
// over engine.Backend: token+position+type embeddings, bidirectional attention,
// LayerNorm, exact-GELU feed-forward, mean pooling + L2 normalize.
package bge

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"memdoor/llm/engine"
	"memdoor/llm/qwen"
	"memdoor/llm/safetensors"
)

// Config holds the XLM-RoBERTa dimensions.
type Config struct {
	Hidden       int
	Layers       int
	Heads        int
	Intermediate int
	Vocab        int
	MaxPos       int
	TypeVocab    int
	Eps          float32
	PadID        int
}

// LoadConfig reads config.json from a model dir.
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
		TypeVocabSize         int     `json:"type_vocab_size"`
		LayerNormEps          float64 `json:"layer_norm_eps"`
		PadTokenID            int     `json:"pad_token_id"`
	}
	if err := json.Unmarshal(raw, &r); err != nil {
		return Config{}, err
	}
	return Config{
		Hidden: r.HiddenSize, Layers: r.NumHiddenLayers, Heads: r.NumAttentionHeads,
		Intermediate: r.IntermediateSize, Vocab: r.VocabSize, MaxPos: r.MaxPositionEmbeddings,
		TypeVocab: r.TypeVocabSize, Eps: float32(r.LayerNormEps), PadID: r.PadTokenID,
	}, nil
}

// linear is a plain (non-quantized) Linear: y = x @ Wᵀ + b. The weight is stored
// [out, in]; we transpose once at load.
type linear struct {
	wt, bias engine.Tensor
}

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
	// RoBERTa position ids: (no padding) [padID+1 .. padID+L].
	pos := make([]int32, L)
	for i := range pos {
		pos[i] = int32(e.padID + 1 + i)
	}
	posT := b.FromInt32(pos, 1, L)

	we := b.TakeAxis(e.word, idsT, 0)     // [1, L, h]
	pe := b.TakeAxis(e.position, posT, 0) // [1, L, h]
	tte := b.Slice(e.tokenType, 0, 0, 1)  // token_type[0]: [1, h], broadcasts over L

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

	attn := b.SDPA(q, k, v, l.scale, false) // bidirectional (no causal mask)
	attn = b.Reshape(b.Transpose(attn, 0, 2, 1, 3), B, L, l.nHeads*l.headDim)
	attn = l.attnOut.forward(b, attn)
	h := b.LayerNorm(b.Add(x, attn), l.attnLNW, l.attnLNB, l.eps)

	ff := l.out.forward(b, geluExact(b, l.inter.forward(b, h)))
	return b.LayerNorm(b.Add(h, ff), l.outLNW, l.outLNB, l.eps)
}

// geluExact: 0.5*x*(1 + erf(x/√2)).
func geluExact(b engine.Backend, x engine.Tensor) engine.Tensor {
	const invSqrt2 = 0.7071067811865476
	return b.ScalarMul(b.Mul(x, b.ScalarAdd(b.Erf(b.ScalarMul(x, invSqrt2)), 1.0)), 0.5)
}

// Model is a BGE-M3 encoder.
type Model struct {
	emb    *embeddings
	layers []*encoderLayer
	cfg    Config
}

// LoadModel loads the encoder weights.
func LoadModel(b engine.Backend, st *safetensors.File, cfg Config) (*Model, error) {
	e := &embeddings{eps: cfg.Eps, padID: cfg.PadID}
	var err error
	if e.word, err = qwen.LoadTensor(b, st, "embeddings.word_embeddings.weight"); err != nil {
		return nil, err
	}
	if e.position, err = qwen.LoadTensor(b, st, "embeddings.position_embeddings.weight"); err != nil {
		return nil, err
	}
	if e.tokenType, err = qwen.LoadTensor(b, st, "embeddings.token_type_embeddings.weight"); err != nil {
		return nil, err
	}
	if e.lnW, e.lnB, err = layerNormPair(b, st, "embeddings.LayerNorm"); err != nil {
		return nil, err
	}

	layers := make([]*encoderLayer, cfg.Layers)
	headDim := cfg.Hidden / cfg.Heads
	for i := range layers {
		p := fmt.Sprintf("encoder.layer.%d", i)
		l := &encoderLayer{nHeads: cfg.Heads, headDim: headDim, eps: cfg.Eps,
			scale: float32(1.0 / math.Sqrt(float64(headDim)))}
		if l.q, err = loadLinear(b, st, p+".attention.self.query"); err != nil {
			return nil, err
		}
		if l.k, err = loadLinear(b, st, p+".attention.self.key"); err != nil {
			return nil, err
		}
		if l.v, err = loadLinear(b, st, p+".attention.self.value"); err != nil {
			return nil, err
		}
		if l.attnOut, err = loadLinear(b, st, p+".attention.output.dense"); err != nil {
			return nil, err
		}
		if l.attnLNW, l.attnLNB, err = layerNormPair(b, st, p+".attention.output.LayerNorm"); err != nil {
			return nil, err
		}
		if l.inter, err = loadLinear(b, st, p+".intermediate.dense"); err != nil {
			return nil, err
		}
		if l.out, err = loadLinear(b, st, p+".output.dense"); err != nil {
			return nil, err
		}
		if l.outLNW, l.outLNB, err = layerNormPair(b, st, p+".output.LayerNorm"); err != nil {
			return nil, err
		}
		layers[i] = l
	}
	return &Model{emb: e, layers: layers, cfg: cfg}, nil
}

// Embed returns the L2-normalized mean-pooled embedding for the token ids.
func (m *Model) Embed(b engine.Backend, ids []int32) []float32 {
	h := m.emb.forward(b, ids)
	for _, l := range m.layers {
		h = l.forward(b, h)
	}
	vec := b.Floats(b.Mean(h, 1)) // mean over sequence -> [hidden]

	var ss float64
	for _, x := range vec {
		ss += float64(x) * float64(x)
	}
	n := float32(math.Sqrt(ss))
	if n > 0 {
		for i := range vec {
			vec[i] /= n
		}
	}
	return vec
}
