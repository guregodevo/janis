package qwen

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A config the engine cannot serve must come back as an ERROR, not a panic.
// Qwen3.5's multimodal config nests every text field under "text_config", so
// top-level num_attention_heads decodes to zero and LoadConfig divided by it —
// panicking the WHOLE GATEWAY at boot (live, 2026-08-31 20:49: a pinned
// Qwen3.5-9B took the gateway down before the "local engine not loaded"
// fallback could catch anything; the boot path handles errors fine, and the
// panic was the only thing that could bypass it).
func TestAMultimodalNestedConfigErrorsInsteadOfPanicking(t *testing.T) {
	dir := t.TempDir()
	cfg := `{
		"model_type": "qwen3_vl",
		"text_config": {"num_attention_heads": 16, "hidden_size": 4096},
		"vision_config": {}
	}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("LoadConfig panicked: %v — a bad model must be an error, not a gateway crash", r)
		}
	}()
	_, err := LoadConfig(dir)
	if err == nil {
		t.Fatal("an unsupported nested config loaded without error")
	}
	for _, want := range []string{"text_config", "not supported"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the error should name the cause (%q): %v", want, err)
		}
	}
}

// The degenerate case with no nesting at all — fields simply absent — must
// also error, not divide by zero.
func TestAConfigWithZeroHeadsErrors(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"),
		[]byte(`{"model_type":"mystery","hidden_size":4096}`), 0o644); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("LoadConfig panicked: %v", r)
		}
	}()
	if _, err := LoadConfig(dir); err == nil {
		t.Fatal("a config with no attention heads loaded without error")
	}
}
