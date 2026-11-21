//go:build darwin && arm64 && cgo

package inkling2

import (
	"path/filepath"
	"testing"

	"github.com/guregodevo/janis/mlxc"
	"github.com/guregodevo/janis/safetensors"
)

// AN IMAGE BECOMES THE ROWS THE REFERENCE GIVES, AND A PROMPT THAT HOLDS
// IT THE REFERENCE'S LOGITS.
//
// tiny-vision is the f32 model with a vision tower of 2 layers; the image
// is 8 by 12 pixels, 2 by 3 tokens. The prompt is read whole, then in
// chunks that cut the run of image tokens.
func TestAnImageGivesTheReferenceRowsAndLogits(t *testing.T) {
	dir := filepath.Join("testdata", "tiny-vision")
	cfg, err := LoadConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	vcfg, ok, err := LoadVisionConfig(dir)
	if err != nil || !ok {
		t.Fatalf("vision config: ok %v, %v", ok, err)
	}
	st, err := safetensors.OpenModel(dir)
	if err != nil {
		t.Fatal(err)
	}
	b := mlxc.New()
	t.Cleanup(b.Close)
	m, err := LoadModel(b, st, cfg)
	if err != nil {
		t.Fatal(err)
	}
	v, err := m.LoadVision(st, vcfg)
	if err != nil {
		t.Fatal(err)
	}
	want, err := safetensors.Open(filepath.Join(dir, "expected.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	rows, err := v.Encode(wantFloats(t, want, "pixels"), 8, 12)
	if err != nil {
		t.Fatal(err)
	}
	wantRows := wantFloats(t, want, "image_rows")
	got := b.Floats(rows)
	if len(got) != len(wantRows) {
		t.Fatalf("%d values for the image, the reference has %d", len(got), len(wantRows))
	}
	if d, s := maxDiff(got, wantRows); d > 2e-4*s {
		t.Errorf("image rows off by %.3g (largest value %.3g)", d, s)
	}

	ids := wantTokens(t, want)
	var at []int
	for i, id := range ids {
		if id == vcfg.ImageToken {
			at = append(at, i)
		}
	}
	if len(at) != 6 {
		t.Fatalf("%d image tokens in the prompt, want 6", len(at))
	}
	m.SetImages(rows, at)
	full := wantFloats(t, want, "logits")
	_, scale := maxDiff(full, full)
	vocab := cfg.UnpaddedVocab
	for _, chunks := range [][]int{{len(ids)}, {4, 3, 4}} {
		caches, pos := m.newCaches(), 0
		for _, n := range chunks {
			chunk := ids[pos : pos+n]
			logits := b.Floats(m.logits(m.hidden(b.FromInt32(chunk, n), n, pos, caches, nil)))
			if d, _ := maxDiff(logits, full[pos*vocab:(pos+n)*vocab]); d > 2e-4*scale {
				t.Errorf("chunks %v, tokens %d..%d: logits off by %.3g (largest value %.3g)", chunks, pos, pos+n-1, d, scale)
			}
			pos += n
		}
	}
	if _, err := v.Encode(make([]float32, 3*8*10), 8, 10); err == nil {
		t.Error("an image whose width is no multiple of the block was encoded")
	}
}
