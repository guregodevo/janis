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
