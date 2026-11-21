package inkling2

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadConfigReadsTheTextConfig(t *testing.T) {
	cfg, err := LoadConfig(filepath.Join("testdata", "tiny-f32"))
	if err != nil {
		t.Fatal(err)
	}
	// 8 layers, the 3 of the core run 3 times.
	if got := cfg.Visits(); got != 14 {
		t.Errorf("Visits() = %d, want 14", got)
	}
	if !cfg.Local[0] || cfg.Local[5] || cfg.Local[7] {
		t.Errorf("local layers %v, want 5 and 7 full", cfg.Local)
	}
	if cfg.UnpaddedVocab != 120 || cfg.Vocab != 128 || cfg.Bits != 4 || cfg.GroupSize != 64 {
		t.Errorf("vocab %d of %d, %d bits in groups of %d", cfg.UnpaddedVocab, cfg.Vocab, cfg.Bits, cfg.GroupSize)
	}
	if len(cfg.EosTokens) != 1 || cfg.EosTokens[0] != 119 {
		t.Errorf("eos %v, want [119]", cfg.EosTokens)
	}
}

// A config janis cannot serve is refused at the load, with the reason, not
// run into a wrong answer or a shape error.
func TestLoadConfigRefusesWhatItCannotServe(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "tiny-f32", "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ name, from, to, want string }{
		{"a variable loop", `"is_varloop": false`, `"is_varloop": true`, "is_varloop"},
		{"another model type", `"model_type": "inkling2_loop"`, `"model_type": "inkling2"`, "not an Inkling 2 model"},
		{"no head dimension", `"head_dim": 16`, `"head_dim": 0`, "head_dim"},
		{"projections not affine", `"mode": "affine"`, `"mode": "mxfp4"`, "affine"},
	} {
		changed := strings.Replace(string(raw), tc.from, tc.to, 1)
		if changed == string(raw) {
			t.Fatalf("%s: the fixture has no %s", tc.name, tc.from)
		}
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(changed), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := LoadConfig(dir); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: error %v, want one naming %q", tc.name, err, tc.want)
		}
	}
}
