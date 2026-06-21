package qwen

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// Config holds the architecture dimensions needed to build a Qwen2 model.
type Config struct {
	ModelType         string
	Hidden            int
	Layers            int
	NHeads            int
	NKVHeads          int
	HeadDim           int
	Vocab             int
	Intermediate      int
	RopeBase          float32
	RMSEps            float32
	GroupSize         int
	Bits              int
	TieWordEmbeddings bool
	AttentionBias     bool // qwen2 q/k/v projections carry a bias
	QKNorm            bool // qwen3 per-head Q/K RMSNorm

	// gemma2-specific (zero/unset for qwen2)
	AttnSoftcap        float32
	FinalSoftcap       float32
	QueryPreAttnScalar float32

	// llama3 rope scaling ("" = none)
	RopeScaling                           string
	RopeFactor, RopeLowFreq, RopeHighFreq float32
	RopeOrigMaxPos                        int

	EosTokens []int32 // end-of-sequence token ids (stop tokens)
}

// rawConfig mirrors the HF config.json fields we consume.
type rawConfig struct {
	ModelType             string          `json:"model_type"`
	HiddenSize            int             `json:"hidden_size"`
	NumHiddenLayers       int             `json:"num_hidden_layers"`
	NumAttentionHeads     int             `json:"num_attention_heads"`
	NumKeyValueHeads      int             `json:"num_key_value_heads"`
	HeadDim               int             `json:"head_dim"`
	VocabSize             int             `json:"vocab_size"`
	IntermediateSize      int             `json:"intermediate_size"`
	RopeTheta             float64         `json:"rope_theta"`
	RMSNormEps            float64         `json:"rms_norm_eps"`
	TieWordEmbeddings     *bool           `json:"tie_word_embeddings"`
	EosTokenID            json.RawMessage `json:"eos_token_id"`
	AttnLogitSoftcapping  float64         `json:"attn_logit_softcapping"`
	FinalLogitSoftcapping float64         `json:"final_logit_softcapping"`
	QueryPreAttnScalar    float64         `json:"query_pre_attn_scalar"`
	RopeScaling           *struct {
		RopeType                      string  `json:"rope_type"`
		Factor                        float64 `json:"factor"`
		LowFreqFactor                 float64 `json:"low_freq_factor"`
		HighFreqFactor                float64 `json:"high_freq_factor"`
		OriginalMaxPositionEmbeddings int     `json:"original_max_position_embeddings"`
	} `json:"rope_scaling"`
	Quantization struct {
		GroupSize int `json:"group_size"`
		Bits      int `json:"bits"`
	} `json:"quantization"`
}

// LoadConfig reads config.json from a model directory.
func LoadConfig(modelDir string) (Config, error) {
	raw, err := os.ReadFile(filepath.Join(modelDir, "config.json"))
	if err != nil {
		return Config{}, fmt.Errorf("read config.json: %w", err)
	}
	var r rawConfig
	if err := json.Unmarshal(raw, &r); err != nil {
		return Config{}, fmt.Errorf("parse config.json: %w", err)
	}

	headDim := r.HeadDim
	if headDim == 0 {
		headDim = r.HiddenSize / r.NumAttentionHeads
	}
	nkv := r.NumKeyValueHeads
	if nkv == 0 {
		nkv = r.NumAttentionHeads
	}
	tie := r.TieWordEmbeddings == nil || *r.TieWordEmbeddings

	cfg := Config{
		ModelType:          r.ModelType,
		Hidden:             r.HiddenSize,
		Layers:             r.NumHiddenLayers,
		NHeads:             r.NumAttentionHeads,
		NKVHeads:           nkv,
		HeadDim:            headDim,
		Vocab:              r.VocabSize,
		Intermediate:       r.IntermediateSize,
		RopeBase:           float32(r.RopeTheta),
		RMSEps:             float32(r.RMSNormEps),
		GroupSize:          r.Quantization.GroupSize,
		Bits:               r.Quantization.Bits,
		TieWordEmbeddings:  tie,
		AttentionBias:      r.ModelType == "qwen2", // qwen2 always has q/k/v bias
		QKNorm:             r.ModelType == "qwen3",
		AttnSoftcap:        float32(r.AttnLogitSoftcapping),
		FinalSoftcap:       float32(r.FinalLogitSoftcapping),
		QueryPreAttnScalar: float32(r.QueryPreAttnScalar),
	}
	if raw := r.EosTokenID; len(raw) > 0 {
		var single int32
		if json.Unmarshal(raw, &single) == nil {
			cfg.EosTokens = []int32{single}
		} else {
			var arr []int32
			_ = json.Unmarshal(raw, &arr)
			cfg.EosTokens = arr
		}
	}
	if rs := r.RopeScaling; rs != nil && rs.RopeType == "llama3" {
		cfg.RopeScaling = "llama3"
		cfg.RopeFactor = float32(rs.Factor)
		cfg.RopeLowFreq = float32(rs.LowFreqFactor)
		cfg.RopeHighFreq = float32(rs.HighFreqFactor)
		cfg.RopeOrigMaxPos = rs.OriginalMaxPositionEmbeddings
	}
	return cfg, nil
}

// Qwen25_3B returns the configuration for Qwen2.5-3B-Instruct (4-bit), kept for
// the rung-validation commands that diff against the 3B oracle.
func Qwen25_3B() Config {
	return Config{
		ModelType: "qwen2", Hidden: 2048, Layers: 36, NHeads: 16, NKVHeads: 2,
		HeadDim: 128, Vocab: 151936, Intermediate: 11008, RopeBase: 1e6, RMSEps: 1e-6,
		GroupSize: 64, Bits: 4, TieWordEmbeddings: true, AttentionBias: true,
	}
}
