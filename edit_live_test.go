package llm

import (
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestWeightEditsLive damages rows of one MLP matrix the way the
// self-rewriting paper does (arXiv:2611.21885: Gaussian noise at 200 times
// the matrix's spread), and checks that reverting gives the checkpoint's
// reply back token for token, and that a saved edit loads back to the
// damaged reply.
//
//	JANIS_EDIT_E2E=1 go test . -run TestWeightEditsLive -v
func TestWeightEditsLive(t *testing.T) {
	if os.Getenv("JANIS_EDIT_E2E") != "1" {
		t.Skip("set JANIS_EDIT_E2E=1 to run (loads Qwen3.5-0.8B-MLX-4bit)")
	}
	home, _ := os.UserHomeDir()
	dirs, _ := filepath.Glob(filepath.Join(home,
		".cache/huggingface/hub/models--mlx-community--Qwen3.5-0.8B-MLX-4bit/snapshots/*"))
	if len(dirs) == 0 {
		t.Skip("Qwen3.5-0.8B-MLX-4bit not installed")
	}
	eng, err := Open(dirs[0])
	if err != nil {
		t.Fatal(err)
	}
	defer eng.Close()

	ask := []Message{{Role: "user", Content: "What is 17 + 25? Answer with just the number."}}
	opts := Options{Temp: 0, MaxTokens: 24, Seed: 1}
	base := eng.Chat(ask, opts)

	ws, err := eng.Weights()
	if err != nil {
		t.Fatal(err)
	}
	const target = "language_model.model.layers.12.mlp.down_proj.weight"
	var found bool
	for _, w := range ws {
		if w.Name == target {
			found = true
			if w.Bits != 4 || w.GroupSize != 64 || w.Dtype != "BF16" || len(w.Shape) != 2 {
				t.Fatalf("%s described as %+v", target, w)
			}
		}
		if strings.Contains(w.Name, "embed_tokens") {
			t.Fatalf("the embedding is listed: %s", w.Name)
		}
	}
	if !found {
		t.Fatalf("%s not in Weights (%d listed)", target, len(ws))
	}

	v, shape, err := eng.ReadWeight(target)
	if err != nil {
		t.Fatal(err)
	}
	var sum, sq float64
	for _, x := range v {
		sum += float64(x)
		sq += float64(x) * float64(x)
	}
	n := float64(len(v))
	sigma := math.Sqrt(sq/n - (sum/n)*(sum/n))
	damaged := append([]float32(nil), v...)
	rng := rand.New(rand.NewSource(7))
	for r := 500; r < 508; r++ {
		for c := 0; c < shape[1]; c++ {
			damaged[r*shape[1]+c] = float32(rng.NormFloat64() * 200 * sigma)
		}
	}
	if err := eng.EditWeight(target, damaged, 0); err != nil {
		t.Fatal(err)
	}
	if got := eng.Edits(); len(got) != 1 || got[0] != target {
		t.Fatalf("Edits = %v", got)
	}
	broken := eng.Chat(ask, opts)
	if broken == base {
		t.Fatalf("200σ noise in 8 rows left the reply as it was: %q", base)
	}
	t.Logf("base %q, damaged %q", base, broken)

	path := filepath.Join(t.TempDir(), "damage.safetensors")
	if err := eng.SaveEdits(path); err != nil {
		t.Fatal(err)
	}
	eng.RevertWeights()
	if len(eng.Edits()) != 0 {
		t.Fatal("edits left after RevertWeights")
	}
	if got := eng.Chat(ask, opts); got != base {
		t.Fatalf("reverted reply %q, the checkpoint's %q", got, base)
	}
	if err := eng.LoadEdits(path); err != nil {
		t.Fatal(err)
	}
	if got := eng.Chat(ask, opts); got != broken {
		t.Fatalf("loaded edit replied %q, the edit made %q", got, broken)
	}
	eng.RevertWeights()

	// A plain tensor written back with the values read is the same bytes.
	const norm = "language_model.model.layers.12.post_attention_layernorm.weight"
	nv, _, err := eng.ReadWeight(norm)
	if err != nil {
		t.Fatal(err)
	}
	if err := eng.EditWeight(norm, nv, 0); err != nil {
		t.Fatal(err)
	}
	if got := eng.Chat(ask, opts); got != base {
		t.Fatalf("a norm written back unchanged replied %q, the checkpoint's %q", got, base)
	}
	eng.RevertWeights()

	// The matrix kept at 8 bits.
	if err := eng.EditWeight(target, v, 8); err != nil {
		t.Fatal(err)
	}
	ws, _ = eng.Weights()
	for _, w := range ws {
		if w.Name == target && (w.Bits != 8 || !w.Edited || w.Shape[1] != shape[1]) {
			t.Fatalf("after an 8-bit edit: %+v", w)
		}
	}
	t.Logf("8-bit copy of the matrix replies %q", eng.Chat(ask, opts))
	eng.RevertWeights()

	for _, bad := range []struct {
		name string
		v    []float32
		bits int
	}{
		{"model.layers.0.nothing", v, 0},
		{target, v[:10], 0},
		{target, v, 3},
		{norm, nv, 4},
		{norm, append([]float32{float32(math.Inf(1))}, nv[1:]...), 0},
	} {
		if err := eng.EditWeight(bad.name, bad.v, bad.bits); err == nil {
			t.Errorf("EditWeight(%s, %d values, %d bits) took it", bad.name, len(bad.v), bad.bits)
		}
	}
	if len(eng.Edits()) != 0 {
		t.Fatalf("a refused edit left %v", eng.Edits())
	}
}
