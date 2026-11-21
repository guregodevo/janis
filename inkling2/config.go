// Package inkling2 implements the text stack of Inkling 2 (model_type
// inkling2_loop), following the mlx-lm implementation. The
// decoder differs from the other families in every part: attention takes a
// learned relative-position bias and a sink logit in place of RoPE, each
// layer reads a softmax mix of the block outputs before it, every sublayer
// output passes a short causal convolution, the MLP is a routed set of
// experts, and the looped variant runs its core layers several times with a
// gated message from the pass before. The vision tower is never loaded.
package inkling2

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// TypeLoop is the model type this package loads. The reference also has a
// decoder with no loop ("inkling2"); no checkpoint of it is published, and
// it is refused until one can be checked.
const TypeLoop = "inkling2_loop"

// Config is the text configuration, from config.json's text_config.
type Config struct {
	ModelType string

	Hidden, Layers int
	Vocab          int // rows of the embedding and the head
	UnpaddedVocab  int // logits the model returns

	// Full attention layers, then sliding-window ("local") ones.
	Heads, KVHeads, HeadDim          int
	SWAHeads, SWAKVHeads, SWAHeadDim int
	Window                           int
	Local                            map[int]bool

	DRel, RelExtent int
	NFloor          int
	Alpha           float32
	Eps             float32
	QKClip, RelClip float32

	ConvKernel int
	BlockSize  int // layers per block of the depth attention

	Experts, TopK, Intermediate         int
	GateCap, UpCap, RouteScale, HeadCap float32

	// The loop: layers [CoreStart, CoreEnd) run Passes times.
	CoreStart, CoreEnd, Passes, GateRank int

	GroupSize, Bits int // affine quantization of the projections

	// StreamExperts leaves the experts on disk and reads, each step, the
	// ones its tokens were routed to. Set by the caller before LoadModel,
	// for a host with less memory than the model.
	StreamExperts bool
	// ExpertCacheBytes bounds the streamed experts kept on the device.
	// Zero keeps a step's own and no more.
	ExpertCacheBytes int64
	EosTokens        []int32
}

type rawText struct {
	HiddenSize      int     `json:"hidden_size"`
	NumHiddenLayers int     `json:"num_hidden_layers"`
	VocabSize       int     `json:"vocab_size"`
	UnpaddedVocab   int     `json:"unpadded_vocab_size"`
	Heads           int     `json:"num_attention_heads"`
	KVHeads         int     `json:"num_key_value_heads"`
	HeadDim         int     `json:"head_dim"`
	SWAHeads        int     `json:"swa_num_attention_heads"`
	SWAKVHeads      int     `json:"swa_num_key_value_heads"`
	SWAHeadDim      int     `json:"swa_head_dim"`
	Window          int     `json:"sliding_window_size"`
	Local           []int   `json:"local_layer_ids"`
	DRel            int     `json:"d_rel"`
	RelExtent       int     `json:"rel_extent"`
	NFloor          int     `json:"log_scaling_n_floor"`
	Alpha           float64 `json:"log_scaling_alpha"`
	Eps             float64 `json:"rms_norm_eps"`
	QKClip          float64 `json:"qk_l2_clip_threshold"`
	RelClip         float64 `json:"rel_l2_clip_threshold"`
	ConvKernel      int     `json:"sconv_kernel_size"`
	BlockSize       int     `json:"n_layers_per_attn_res_block"`
	Experts         int     `json:"n_routed_experts"`
	TopK            int     `json:"num_experts_per_tok"`
	Intermediate    int     `json:"intermediate_size"`
	GateCap         float64 `json:"mlp_gate_softcap"`
	UpCap           float64 `json:"mlp_up_softcap"`
	RouteScale      float64 `json:"route_scale"`
	HeadCap         float64 `json:"final_logit_softcapping"`
	Loop            *struct {
		CoreStart int  `json:"core_start"`
		CoreEnd   int  `json:"core_end"`
		Passes    int  `json:"num_passes"`
		GateRank  int  `json:"gate_rank"`
		Varloop   bool `json:"is_varloop"`
	} `json:"loop_config"`
}

type rawConfig struct {
	ModelType    string          `json:"model_type"`
	Text         *rawText        `json:"text_config"`
	EosTokenID   json.RawMessage `json:"eos_token_id"`
	Quantization *struct {
		GroupSize int    `json:"group_size"`
		Bits      int    `json:"bits"`
		Mode      string `json:"mode"`
	} `json:"quantization"`
}

// ModelType reads model_type from a model directory's config.json, "" when
// it cannot be read.
func ModelType(modelDir string) string {
	raw, err := os.ReadFile(filepath.Join(modelDir, "config.json"))
	if err != nil {
		return ""
	}
	var r struct {
		ModelType string `json:"model_type"`
	}
	if json.Unmarshal(raw, &r) != nil {
		return ""
	}
	return r.ModelType
}

// Handles reports whether this package loads the model type.
func Handles(modelType string) bool { return modelType == TypeLoop }

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
	if !Handles(r.ModelType) {
		return Config{}, fmt.Errorf("%s: model_type %q is not an Inkling 2 model", modelDir, r.ModelType)
	}
	t := r.Text
	if t == nil {
		return Config{}, fmt.Errorf("%s: config has no text_config", modelDir)
	}
	if r.Quantization == nil || r.Quantization.Bits == 0 {
		return Config{}, fmt.Errorf("%s: the projections are not quantized; janis loads the MLX conversions (affine 4-bit and up)", modelDir)
	}
	if m := r.Quantization.Mode; m != "" && m != "affine" {
		return Config{}, fmt.Errorf("%s: projections quantized in mode %q, janis reads affine", modelDir, m)
	}
	cfg := Config{
		ModelType: r.ModelType,
		Hidden:    t.HiddenSize, Layers: t.NumHiddenLayers,
		Vocab: t.VocabSize, UnpaddedVocab: t.UnpaddedVocab,
		Heads: t.Heads, KVHeads: t.KVHeads, HeadDim: t.HeadDim,
		SWAHeads: t.SWAHeads, SWAKVHeads: t.SWAKVHeads, SWAHeadDim: t.SWAHeadDim,
		Window: t.Window, Local: map[int]bool{},
		DRel: t.DRel, RelExtent: t.RelExtent,
		NFloor: t.NFloor, Alpha: float32(t.Alpha), Eps: float32(t.Eps),
		QKClip: float32(t.QKClip), RelClip: float32(t.RelClip),
		ConvKernel: t.ConvKernel, BlockSize: t.BlockSize,
		Experts: t.Experts, TopK: t.TopK, Intermediate: t.Intermediate,
		GateCap: float32(t.GateCap), UpCap: float32(t.UpCap),
		RouteScale: float32(t.RouteScale), HeadCap: float32(t.HeadCap),
		GroupSize: r.Quantization.GroupSize, Bits: r.Quantization.Bits,
	}
	for _, i := range t.Local {
		cfg.Local[i] = true
	}
	if cfg.UnpaddedVocab == 0 {
		cfg.UnpaddedVocab = cfg.Vocab
	}
	l := t.Loop
	if l == nil {
		return Config{}, fmt.Errorf("%s: text_config has no loop_config", modelDir)
	}
	if l.Varloop {
		return Config{}, fmt.Errorf("%s: a variable loop (is_varloop) is not supported", modelDir)
	}
	cfg.CoreStart, cfg.CoreEnd, cfg.Passes, cfg.GateRank = l.CoreStart, l.CoreEnd, l.Passes, l.GateRank
	// A zero here is a division or an empty shape later in the load.
	for name, v := range map[string]int{
		"hidden_size": cfg.Hidden, "num_hidden_layers": cfg.Layers, "vocab_size": cfg.Vocab,
		"num_attention_heads": cfg.Heads, "num_key_value_heads": cfg.KVHeads, "head_dim": cfg.HeadDim,
		"swa_num_attention_heads": cfg.SWAHeads, "swa_num_key_value_heads": cfg.SWAKVHeads,
		"swa_head_dim": cfg.SWAHeadDim, "sliding_window_size": cfg.Window,
		"d_rel": cfg.DRel, "rel_extent": cfg.RelExtent, "log_scaling_n_floor": cfg.NFloor,
		"sconv_kernel_size": cfg.ConvKernel, "n_layers_per_attn_res_block": cfg.BlockSize,
		"n_routed_experts": cfg.Experts, "num_experts_per_tok": cfg.TopK,
		"intermediate_size": cfg.Intermediate, "num_passes": cfg.Passes,
	} {
		if v <= 0 {
			return Config{}, fmt.Errorf("%s: text_config.%s is missing or zero", modelDir, name)
		}
	}
	if cfg.ConvKernel < 2 {
		return Config{}, fmt.Errorf("%s: sconv_kernel_size %d, need 2 or more", modelDir, cfg.ConvKernel)
	}
	if cfg.Heads%cfg.KVHeads != 0 || cfg.SWAHeads%cfg.SWAKVHeads != 0 {
		return Config{}, fmt.Errorf("%s: attention heads are not a multiple of the key/value heads", modelDir)
	}
	if cfg.CoreStart < 0 || cfg.CoreEnd < cfg.CoreStart || cfg.CoreEnd > cfg.Layers {
		return Config{}, fmt.Errorf("%s: loop core [%d, %d) is outside the %d layers", modelDir, cfg.CoreStart, cfg.CoreEnd, cfg.Layers)
	}
	if raw := r.EosTokenID; len(raw) > 0 {
		var single int32
		if json.Unmarshal(raw, &single) == nil {
			cfg.EosTokens = []int32{single}
		} else {
			_ = json.Unmarshal(raw, &cfg.EosTokens)
		}
	}
	return cfg, nil
}

// Visits is the number of layer calls of one forward pass: each has a cache
// of its own.
func (c Config) Visits() int {
	return c.Layers + (c.Passes-1)*(c.CoreEnd-c.CoreStart)
}
