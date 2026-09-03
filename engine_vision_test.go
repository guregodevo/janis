package llm

import (
	"encoding/json"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// The engine answers a question about a frame with its own tower: the
// answer opens with the mlx-vlm oracle's words for the same frame.
func TestEngineSees(t *testing.T) {
	home, _ := os.UserHomeDir()
	m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/*/config.json"))
	if len(m) == 0 {
		t.Skip("Qwen3.5-9B not downloaded")
	}
	f, err := os.Open("qwen35/testdata/vision_frame.png")
	if err != nil {
		t.Skip(err)
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile("qwen35/testdata/vision_oracle.json")
	if err != nil {
		t.Skip(err)
	}
	var o struct {
		Question string `json:"question"`
		Text     string `json:"text"`
	}
	json.Unmarshal(raw, &o)

	e, err := Open(filepath.Dir(m[0]))
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	if !e.CanSee() {
		t.Fatal("a Qwen3.5 engine must be able to see")
	}
	t0 := time.Now()
	got, err := e.See([]image.Image{img}, o.Question, 40)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("saw in %s: %q", time.Since(t0).Round(time.Millisecond), got)
	head := o.Text
	if len(head) > 30 {
		head = head[:30]
	}
	if !strings.HasPrefix(got, head) {
		t.Errorf("answer %q does not open like the oracle %q", got, o.Text)
	}
	// Several images in one prompt (the thumbnail judge): the earlier
	// images' features must survive the later encodes.
	t0 = time.Now()
	got, err = e.See([]image.Image{img, img, img}, "Which of these frames is the sharpest? Answer with its number and why.", 60)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("three frames in %s: %q", time.Since(t0).Round(time.Millisecond), got)
	if got == "" {
		t.Error("no answer for three frames")
	}
}
