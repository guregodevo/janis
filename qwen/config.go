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

	// qwen3_moe (zero/unset for dense models)
	NumExperts      int  // 0 = dense model
	TopK            int  // experts activated per token
	MoeIntermediate int  // per-expert FFN width
	SparseStep      int  // every Nth layer is MoE (1 = all)
	NormTopk        bool // renormalize the top-k router scores

	// qwen3_5 hybrid linear attention (zero/unset for pure-attention models).
	// Every FullAttnInterval-th layer is softmax attention with a sigmoid
	// output gate (q_proj emits [query, gate] per head); the rest are Gated
	// DeltaNet linear-attention layers with a depthwise causal conv front-end.
	FullAttnInterval int
	LinearKHeads     int     // linear_num_key_heads (q/k heads)
	LinearVHeads     int     // linear_num_value_heads
	LinearKDim       int     // linear_key_head_dim
	LinearVDim       int     // linear_value_head_dim
	ConvKernel       int     // linear_conv_kernel_dim
	PartialRotary    float32 // fraction of head_dim RoPE rotates (0 = all)

	// QuantOverrides maps a tensor name to its quantization when it differs
	// from the global GroupSize/Bits — mlx-community MoE checkpoints quantize
	// the per-layer routers at 8-bit while everything else is 4-bit.
	QuantOverrides map[string]QuantSpec

	EosTokens []int32 // end-of-sequence token ids (stop tokens)
}

// QuantSpec is one tensor's affine-quantization parameters.
type QuantSpec struct {
	GroupSize int `json:"group_size"`
	Bits      int `json:"bits"`
}

// QuantFor resolves the quantization for a named tensor: its override if the
// checkpoint declares one, else the global defaults.
func (c Config) QuantFor(name string) (groupSize, bits int) {
	if q, ok := c.QuantOverrides[name]; ok {
		return q.GroupSize, q.Bits
	}
	return c.GroupSize, c.Bits
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
	Quantization        json.RawMessage `json:"quantization"`
	NumExperts          int             `json:"num_experts"`
	NumExpertsPerTok    int             `json:"num_experts_per_tok"`
	MoeIntermediateSize int             `json:"moe_intermediate_size"`
	DecoderSparseStep   int             `json:"decoder_sparse_step"`
	NormTopkProb        *bool           `json:"norm_topk_prob"`

	// qwen3_5 multimodal layout: the text fields above live nested here, and
	// the hybrid linear-attention fields below live inside that nest.
	TextConfig            json.RawMessage `json:"text_config"`
	FullAttentionInterval int             `json:"full_attention_interval"`
	LinearNumKeyHeads     int             `json:"linear_num_key_heads"`
	LinearNumValueHeads   int             `json:"linear_num_value_heads"`
	LinearKeyHeadDim      int             `json:"linear_key_head_dim"`
	LinearValueHeadDim    int             `json:"linear_value_head_dim"`
	LinearConvKernelDim   int             `json:"linear_conv_kernel_dim"`
	RopeParameters        *struct {
		RopeTheta           float64 `json:"rope_theta"`
		PartialRotaryFactor float64 `json:"partial_rotary_factor"`
	} `json:"rope_parameters"`
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

	// Qwen3.5 is a multimodal checkpoint: the whole text stack nests under
	// "text_config" (rope inside "rope_parameters"), while quantization and
	// tie_word_embeddings stay top-level. Hoist the nest so the rest of the
	// loader sees one flat config. Other text_config families (no engine
	// support for their text stack) still refuse below.
	if r.ModelType == "qwen3_5" && len(r.TextConfig) > 0 {
		var tr rawConfig
		if err := json.Unmarshal(r.TextConfig, &tr); err != nil {
			return Config{}, fmt.Errorf("parse text_config: %w", err)
		}
		tr.ModelType = r.ModelType
		tr.Quantization = r.Quantization
		tr.TieWordEmbeddings = r.TieWordEmbeddings
		if tr.RopeParameters != nil {
			tr.RopeTheta = tr.RopeParameters.RopeTheta
		}
		r = tr
	}

	// The quantization object holds global group_size/bits plus optional
	// per-tensor overrides as sibling keys whose values are objects:
	//   {"group_size":64, "bits":4, "model.layers.0.mlp.gate":{"group_size":64,"bits":8}}
	var globalQ QuantSpec
	var quantOverrides map[string]QuantSpec
	if len(r.Quantization) > 0 {
		if err := json.Unmarshal(r.Quantization, &globalQ); err != nil {
			return Config{}, fmt.Errorf("parse quantization: %w", err)
		}
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(r.Quantization, &fields); err != nil {
			return Config{}, fmt.Errorf("parse quantization: %w", err)
		}
		for name, v := range fields {
			if len(v) == 0 || v[0] != '{' {
				continue
			}
			var q QuantSpec
			if err := json.Unmarshal(v, &q); err != nil || q.Bits == 0 {
				continue
			}
			if quantOverrides == nil {
				quantOverrides = map[string]QuantSpec{}
			}
			quantOverrides[name] = q
		}
	}

	// A field that decodes to zero is a config this engine cannot serve, not a
	// number to divide by. Qwen3.5's multimodal layout nests every text field
	// under "text_config", so the top-level heads read as zero and the old
	// division PANICKED THE GATEWAY at boot (live, 2026-08-31: a pinned
	// Qwen3.5-9B took the whole process down; the boot path degrades cleanly
	// on an error, and the panic was the only thing that could bypass it).
	if r.NumAttentionHeads == 0 || r.HiddenSize == 0 {
		if hasNestedTextConfig(modelDir) {
			return Config{}, fmt.Errorf("%s: multimodal config (fields nested under text_config) is not supported by this engine — pick a text-only model (`memdoor llm auto`)", modelDir)
		}
		return Config{}, fmt.Errorf("%s: config has no attention heads / hidden size — not a model this engine can load", modelDir)
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
		GroupSize:          globalQ.GroupSize,
		Bits:               globalQ.Bits,
		QuantOverrides:     quantOverrides,
		TieWordEmbeddings:  tie,
		AttentionBias:      r.ModelType == "qwen2", // qwen2 always has q/k/v bias
		QKNorm:             r.ModelType == "qwen3" || r.ModelType == "qwen3_moe" || r.ModelType == "qwen3_5",
		NumExperts:         r.NumExperts,
		TopK:               r.NumExpertsPerTok,
		MoeIntermediate:    r.MoeIntermediateSize,
		SparseStep:         r.DecoderSparseStep,
		NormTopk:           r.NormTopkProb == nil || *r.NormTopkProb,
		AttnSoftcap:        float32(r.AttnLogitSoftcapping),
		FinalSoftcap:       float32(r.FinalLogitSoftcapping),
		QueryPreAttnScalar: float32(r.QueryPreAttnScalar),
		FullAttnInterval:   r.FullAttentionInterval,
		LinearKHeads:       r.LinearNumKeyHeads,
		LinearVHeads:       r.LinearNumValueHeads,
		LinearKDim:         r.LinearKeyHeadDim,
		LinearVDim:         r.LinearValueHeadDim,
		ConvKernel:         r.LinearConvKernelDim,
	}
	if r.RopeParameters != nil {
		cfg.PartialRotary = float32(r.RopeParameters.PartialRotaryFactor)
	}
	if r.ModelType == "qwen3_5" {
		// Fail fast on a hybrid config the model builder can't serve, rather
		// than letting a zero dimension surface as a shape error mid-load.
		if cfg.FullAttnInterval <= 0 || cfg.LinearKHeads <= 0 || cfg.LinearVHeads <= 0 ||
			cfg.LinearKDim <= 0 || cfg.LinearVDim <= 0 || cfg.ConvKernel <= 0 {
			return Config{}, fmt.Errorf("%s: qwen3_5 config is missing linear-attention dimensions", modelDir)
		}
		if cfg.LinearVHeads%cfg.LinearKHeads != 0 {
			return Config{}, fmt.Errorf("%s: linear_num_value_heads (%d) not divisible by linear_num_key_heads (%d)", modelDir, cfg.LinearVHeads, cfg.LinearKHeads)
		}
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

// hasNestedTextConfig reports whether the model's config nests its text fields
// under "text_config" — the multimodal layout, named in the error so the
// reader learns why a real model from a real org refuses to load.
func hasNestedTextConfig(modelDir string) bool {
	b, err := os.ReadFile(filepath.Join(modelDir, "config.json"))
	if err != nil {
		return false
	}
	var probe struct {
		TextConfig map[string]json.RawMessage `json:"text_config"`
	}
	return json.Unmarshal(b, &probe) == nil && len(probe.TextConfig) > 0
}
