//go:build ggml

package inkling2

import (
	"fmt"
	"math"
	"testing"

	"github.com/guregodevo/janis/engine"
	"github.com/guregodevo/janis/ggml"
	"github.com/guregodevo/janis/qwen"
)

// THE GGML BACKEND FOLLOWS THE REFERENCE.
//
// The f32 tiny model on ggml (the CPU device here, CUDA on a GPU host).
// ggml rounds a quantized matmul's input to 8 bits, which the reference
// does not: the first four visits are within 5% of the reference's, and
// with random weights an expert choice flips after them, as it does for
// the bf16 model on MLX. The run goes on to the logits and a generation,
// which must be numbers.
func TestGGMLMatchesTheReference(t *testing.T) {
	b := ggml.New()
	t.Cleanup(b.Close)
	m, want := loadTiny(t, b, "f32", false)
	b.PinAll()
	ids := wantTokens(t, want)
	check := func(name string, got engine.Tensor) {
		t.Helper()
		g, w := b.Floats(got), wantFloats(t, want, name)
		if len(g) != len(w) {
			t.Fatalf("%s: %d values, the reference has %d", name, len(g), len(w))
		}
		if d, s := maxDiff(g, w); d > 5e-2*s || math.IsNaN(d) {
			t.Errorf("%s: off by %.3g (largest value %.3g)", name, d, s)
		}
	}
	var visits []engine.Tensor
	h := m.hidden(b.FromInt32(ids, len(ids)), len(ids), 0, m.newCaches(), func(i int, h engine.Tensor) {
		if i < 0 {
			check("embed", h)
			return
		}
		visits = append(visits, h)
	})
	if len(visits) != m.Cfg.Visits() {
		t.Fatalf("%d visits, want %d", len(visits), m.Cfg.Visits())
	}
	for i, v := range visits[:4] {
		check(fmt.Sprintf("visit.%02d", i), v)
	}
	for _, v := range b.Floats(m.logits(h)) {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("a logit is %v", v)
		}
	}
	if got := m.NewSession().Generate(b, ids[:8], 8, qwen.SampleParams{}); len(got) != 8 {
		t.Errorf("generated %v, want 8 tokens", got)
	}
}

// ONE TOKEN AT A TIME GIVES THE LOGITS OF ONE PASS ON GGML.
//
// One token goes through the backend's own attention (RelAttend): the
// caches read in place and padded, the bias gathered by distance, the
// window a mask. A chunk goes through the model's operations. Twelve
// tokens in two chunks, then one at a time past the sliding window of 8.
func TestGGMLStepsGiveTheLogitsOfOnePass(t *testing.T) {
	t.Setenv("JANIS_GGML_REL_F32", "1")
	b := ggml.New()
	t.Cleanup(b.Close)
	m, want := loadTiny(t, b, "f32", false)
	b.PinAll()
	ids := wantTokens(t, want)
	vocab := m.Cfg.UnpaddedVocab
	full := b.Floats(m.logits(m.hidden(b.FromInt32(ids, len(ids)), len(ids), 0, m.newCaches(), nil)))
	_, scale := maxDiff(full, full)

	caches := m.newCaches()
	for pos := 0; pos < len(ids); {
		n := 1
		if pos < 12 {
			n = 6
		}
		got := b.Floats(m.logits(m.hidden(b.FromInt32(ids[pos:pos+n], n), n, pos, caches, nil)))
		if d, _ := maxDiff(got, full[pos*vocab:(pos+n)*vocab]); d > 2e-3*scale || math.IsNaN(d) {
			t.Errorf("tokens %d..%d: off by %.3g from one pass (largest value %.3g)", pos, pos+n-1, d, scale)
		}
		pos += n
	}
	if got := caches[0].(*VisitCache).K.Shape()[2]; got != len(ids) {
		t.Errorf("a sliding layer keeps %d keys on ggml, want all %d", got, len(ids))
	}
}

// A RECORDED STEP GENERATES WHAT A STEP BUILT EVERY TOKEN GENERATES.
//
// Greedy from the first 8 tokens: with the decode step recorded once and
// run again (JANIS_GGML_REPLAY=cpu) and with it built every token, then a
// second reply on the same session from a prompt that keeps the first 6
// tokens, which cuts the caches back before decoding again.
func TestGGMLARecordedStepGeneratesTheSameTokens(t *testing.T) {
	t.Setenv("JANIS_GGML_REL_F32", "1")
	run := func(replay string) [][]int32 {
		t.Setenv("JANIS_GGML_REPLAY", replay)
		b := ggml.New()
		defer b.Close()
		m, want := loadTiny(t, b, "f32", false)
		b.PinAll()
		ids := wantTokens(t, want)
		s := m.NewSession()
		first := s.Generate(b, ids[:8], 12, qwen.SampleParams{})
		second := s.Generate(b, append(append([]int32(nil), ids[:6]...), ids[10:14]...), 12, qwen.SampleParams{})
		return [][]int32{first, second}
	}
	built, recorded := run("0"), run("cpu")
	if fmt.Sprint(built) != fmt.Sprint(recorded) {
		t.Errorf("recorded steps generated %v, steps built every token %v", recorded, built)
	}
	if len(recorded[0]) != 12 || len(recorded[1]) != 12 {
		t.Errorf("generated %d and %d tokens, want 12 and 12", len(recorded[0]), len(recorded[1]))
	}
}
