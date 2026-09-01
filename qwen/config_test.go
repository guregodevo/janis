package qwen

import (
	"os"
	"path/filepath"
	"testing"
)

// TestQuantOverrides checks that per-tensor quantization overrides in
// config.json (mlx-community MoE checkpoints quantize routers at 8-bit) are
// parsed and resolved by QuantFor, with globals for everything else.
func TestQuantOverrides(t *testing.T) {
	dir := t.TempDir()
	cfg := `{
		"model_type": "qwen3_moe",
		"hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
		"num_key_value_heads": 2, "vocab_size": 100, "intermediate_size": 128,
		"rope_theta": 10000, "rms_norm_eps": 1e-6,
		"num_experts": 8, "num_experts_per_tok": 2,
		"quantization": {
			"group_size": 64, "bits": 4,
			"model.layers.0.mlp.gate": {"group_size": 64, "bits": 8},
			"model.layers.1.mlp.gate": {"group_size": 32, "bits": 8}
		}
	}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
	c, err := LoadConfig(dir)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if c.GroupSize != 64 || c.Bits != 4 {
		t.Errorf("globals = %d/%d, want 64/4", c.GroupSize, c.Bits)
	}
	if gs, bits := c.QuantFor("model.layers.0.mlp.gate"); gs != 64 || bits != 8 {
		t.Errorf("layer0 gate = %d/%d, want 64/8", gs, bits)
	}
	if gs, bits := c.QuantFor("model.layers.1.mlp.gate"); gs != 32 || bits != 8 {
		t.Errorf("layer1 gate = %d/%d, want 32/8", gs, bits)
	}
	if gs, bits := c.QuantFor("model.layers.0.self_attn.q_proj"); gs != 64 || bits != 4 {
		t.Errorf("non-overridden tensor = %d/%d, want globals 64/4", gs, bits)
	}
}

// TestQwen35NestedTextConfig checks that the multimodal Qwen3.5 layout — text
// fields nested under text_config, rope under rope_parameters, quantization and
// tie_word_embeddings top-level — parses into a flat hybrid config instead of
// being refused. Mirrors mlx-community/Qwen3.5-9B-MLX-4bit's real config.json.
func TestQwen35NestedTextConfig(t *testing.T) {
	dir := t.TempDir()
	cfg := `{
		"architectures": ["Qwen3_5ForConditionalGeneration"],
		"model_type": "qwen3_5",
		"tie_word_embeddings": false,
		"quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
		"text_config": {
			"model_type": "qwen3_5_text",
			"hidden_size": 4096, "num_hidden_layers": 32,
			"num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 256,
			"vocab_size": 248320, "intermediate_size": 12288,
			"rms_norm_eps": 1e-6, "eos_token_id": 248044,
			"full_attention_interval": 4,
			"linear_num_key_heads": 16, "linear_num_value_heads": 32,
			"linear_key_head_dim": 128, "linear_value_head_dim": 128,
			"linear_conv_kernel_dim": 4,
			"rope_parameters": {"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "mrope_interleaved": true, "mrope_section": [11, 11, 10]}
		},
		"vision_config": {"depth": 24}
	}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
	c, err := LoadConfig(dir)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if c.ModelType != "qwen3_5" {
		t.Errorf("ModelType = %q, want qwen3_5", c.ModelType)
	}
	if c.Hidden != 4096 || c.Layers != 32 || c.NHeads != 16 || c.NKVHeads != 4 || c.HeadDim != 256 {
		t.Errorf("attention dims = %d/%d/%d/%d/%d", c.Hidden, c.Layers, c.NHeads, c.NKVHeads, c.HeadDim)
	}
	if c.Vocab != 248320 || c.Intermediate != 12288 {
		t.Errorf("vocab/intermediate = %d/%d", c.Vocab, c.Intermediate)
	}
	if c.GroupSize != 64 || c.Bits != 4 {
		t.Errorf("quantization = %d/%d, want top-level 64/4", c.GroupSize, c.Bits)
	}
	if c.TieWordEmbeddings {
		t.Error("tie_word_embeddings should come from the top level (false)")
	}
	if c.RopeBase != 1e7 {
		t.Errorf("RopeBase = %v, want 1e7 from rope_parameters", c.RopeBase)
	}
	if c.PartialRotary != 0.25 {
		t.Errorf("PartialRotary = %v, want 0.25", c.PartialRotary)
	}
	if c.FullAttnInterval != 4 || c.LinearKHeads != 16 || c.LinearVHeads != 32 ||
		c.LinearKDim != 128 || c.LinearVDim != 128 || c.ConvKernel != 4 {
		t.Errorf("hybrid dims = %d/%d/%d/%d/%d/%d",
			c.FullAttnInterval, c.LinearKHeads, c.LinearVHeads, c.LinearKDim, c.LinearVDim, c.ConvKernel)
	}
	if !c.QKNorm {
		t.Error("qwen3_5 has per-head q/k norms; QKNorm should be true")
	}
	if len(c.EosTokens) != 1 || c.EosTokens[0] != 248044 {
		t.Errorf("EosTokens = %v, want [248044] from text_config", c.EosTokens)
	}
}
