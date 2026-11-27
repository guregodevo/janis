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
