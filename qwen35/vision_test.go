package qwen35

import (
	"encoding/binary"
	"encoding/json"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"memdoor/llm/engine"
	"memdoor/llm/mlxc"
	"memdoor/llm/safetensors"
)

func qwen35Dir(t *testing.T) string {
	home, _ := os.UserHomeDir()
	m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/*/config.json"))
	if len(m) == 0 {
		t.Skip("Qwen3.5-9B not downloaded")
	}
	return filepath.Dir(m[0])
}

func readF32(t *testing.T, name string) []float32 {
	raw, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Skip(err)
	}
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}

type visionOracle struct {
	GridTHW      [][]int `json:"grid_thw"`
	InputIDs     []int   `json:"input_ids"`
	NImageTokens int     `json:"n_image_tokens"`
	FirstToken   int     `json:"first_token"`
	Text         string  `json:"text"`
}

func loadVisionOracle(t *testing.T) (visionOracle, Patches) {
	raw, err := os.ReadFile("testdata/vision_oracle.json")
	if err != nil {
		t.Skip(err)
	}
	var o visionOracle
	if err := json.Unmarshal(raw, &o); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open("testdata/vision_frame.png")
	if err != nil {
		t.Skip(err)
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	return o, Preprocess(img, Qwen35Vision)
}

// rowCosine returns the minimum cosine similarity over rows of width d.
func rowCosine(got, want []float32, d int) (minCos float64, maxAbs float64) {
	minCos = 1
	rows := len(want) / d
	for r := 0; r < rows; r++ {
		var dot, na, nb float64
		for i := 0; i < d; i++ {
			g, w := float64(got[r*d+i]), float64(want[r*d+i])
			dot += g * w
			na += g * g
			nb += w * w
			maxAbs = math.Max(maxAbs, math.Abs(g-w))
		}
		if c := dot / math.Sqrt(na*nb); c < minCos {
			minCos = c
		}
	}
	return minCos, maxAbs
}

// Milestone 1: the Go preprocessing produces the processor's patches, in its
// order, to float precision (the fixture image needs no resize).
func TestVisionPreprocessMatchesOracle(t *testing.T) {
	o, p := loadVisionOracle(t)
	want := readF32(t, "vision_patches.f32")
	if g := o.GridTHW[0]; p.T != g[0] || p.H != g[1] || p.W != g[2] {
		t.Fatalf("grid (%d,%d,%d), oracle %v", p.T, p.H, p.W, g)
	}
	if len(p.Data) != len(want) {
		t.Fatalf("%d values, oracle %d", len(p.Data), len(want))
	}
	maxErr := 0.0
	for i := range want {
		maxErr = math.Max(maxErr, math.Abs(float64(p.Data[i]-want[i])))
	}
	t.Logf("patches: %d × %d, max abs err %.2e", p.N(), Qwen35Vision.PatchDim(), maxErr)
	if maxErr > 1e-5 {
		t.Fatalf("patch values differ from the processor's by %.2e", maxErr)
	}
}

// Milestone 2: patch embedding and the whole tower match mlx-vlm's.
func TestVisionTowerMatchesOracle(t *testing.T) {
	o, p := loadVisionOracle(t)
	wantPE := readF32(t, "vision_patch_embed.f32")
	wantOut := readF32(t, "vision_tower_out.f32")
	b := mlxc.New()
	defer b.Close()
	st, err := safetensors.OpenModel(qwen35Dir(t))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	t0 := time.Now()
	tower, err := LoadVisionTower(b, st, Qwen35Vision)
	if err != nil {
		t.Fatal(err)
	}
	if sw, ok := any(b).(engine.Sweeper); ok {
		sw.PinAll()
	}
	t.Logf("tower loaded in %s", time.Since(t0).Round(time.Millisecond))

	pe := b.Floats(b.Cast(tower.PatchEmbed(b, p), engine.F32))
	c, maxAbs := rowCosine(pe, wantPE, Qwen35Vision.Hidden)
	t.Logf("patch embed vs oracle: min row cosine %.5f, max abs %.4f", c, maxAbs)
	if c < 0.999 {
		t.Fatalf("patch embedding min row cosine %.5f", c)
	}

	t0 = time.Now()
	out := tower.Forward(b, p)
	got := b.Floats(b.Cast(out, engine.F32))
	t.Logf("tower forward in %s: %v", time.Since(t0).Round(time.Millisecond), out.Shape())
	if len(got) != len(wantOut) {
		t.Fatalf("%d values, oracle %d (%d tokens expected)", len(got), len(wantOut), o.NImageTokens)
	}
	c, maxAbs = rowCosine(got, wantOut, Qwen35Vision.OutHidden)
	t.Logf("tower out vs oracle: min row cosine %.5f, max abs %.4f over %d tokens", c, maxAbs, o.NImageTokens)
	if c < 0.99 {
		t.Fatalf("tower output min row cosine %.5f", c)
	}
}
