//go:build ggml

package inkling2

import (
	"fmt"
	"math"
	"math/rand"
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

// ONE TOKEN AT A TIME FOLLOWS ONE PASS ON GGML.
//
// One token goes through the backend's own operations (the attention over
// caches read in place, the convolutions, the mixes, the gates); a chunk
// goes through the model's. Twelve tokens in two chunks, then one at a
// time past the sliding window of 8. The two round differently (a
// quantized matmul rounds its input by block), and the tiny model's random
// weights grow that from token to token: the first three single tokens are
// held to 3% of the largest logit. TestGGMLATokensOperationsAreTheModels
// holds each operation to the model's own.
func TestGGMLStepsFollowOnePass(t *testing.T) {
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
		n, tol := 1, 3e-2
		if pos < 12 {
			n, tol = 6, 2e-3
		}
		got := b.Floats(m.logits(m.hidden(b.FromInt32(ids[pos:pos+n], n), n, pos, caches, nil)))
		d, _ := maxDiff(got, full[pos*vocab:(pos+n)*vocab])
		if math.IsNaN(d) || (pos < 15 && d > tol*scale) {
			t.Errorf("tokens %d..%d: off by %.3g from one pass (largest value %.3g)", pos, pos+n-1, d, scale)
		}
		pos += n
	}
	if got := caches[0].(*VisitCache).K.Shape()[2]; got != len(ids) {
		t.Errorf("a sliding layer keeps %d keys on ggml, want all %d", got, len(ids))
	}
}

// A PROMPT READ IN CHUNKS GIVES THE LOGITS OF ONE PASS ON GGML.
//
// The first chunk goes through the model's own attention, the ones after
// through the backend's (the caches read in place, the bias gathered for
// the whole chunk in one operation). Chunks of 6, 6, 6 and 2: the third
// starts past the sliding window of 8, so a sliding layer reads its caches
// from the middle.
func TestGGMLChunksGiveTheLogitsOfOnePass(t *testing.T) {
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
		n := min(6, len(ids)-pos)
		got := b.Floats(m.logits(m.hidden(b.FromInt32(ids[pos:pos+n], n), n, pos, caches, nil)))
		if d, _ := maxDiff(got, full[pos*vocab:(pos+n)*vocab]); d > 2e-3*scale || math.IsNaN(d) {
			t.Errorf("tokens %d..%d: off by %.3g from one pass (largest value %.3g)", pos, pos+n-1, d, scale)
		}
		pos += n
	}
}

// A TOKEN'S OPERATIONS ON GGML ARE THE MODEL'S OWN.
//
// Each part of a layer that has no state, on one token: through the
// backend's fused operation and through the model's operations (the
// backend's set aside), on every layer of the tiny model. They agree to
// 0.2% of the largest value.
func TestGGMLATokensOperationsAreTheModels(t *testing.T) {
	b := ggml.New()
	t.Cleanup(b.Close)
	m, _ := loadTiny(t, b, "f32", false)
	b.PinAll()
	c := m.c
	r := rand.New(rand.NewSource(3))
	rnd := func(shape ...int) engine.Tensor {
		n := 1
		for _, d := range shape {
			n *= d
		}
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(r.NormFloat64())
		}
		return b.FromFloats(v, shape...)
	}
	both := func(name string, f func() engine.Tensor) {
		t.Helper()
		fused := b.Floats(f())
		rel := c.rel
		c.rel = nil
		own := b.Floats(f())
		c.rel = rel
		if d, s := maxDiff(fused, own); d > 2e-3*s || math.IsNaN(d) || len(fused) != len(own) {
			t.Errorf("%s: off by %.3g (largest value %.3g)", name, d, s)
		}
	}
	hidden := m.Cfg.Hidden
	for i, l := range m.Layers {
		x, msg := rnd(1, 1, hidden), rnd(1, 1, hidden)
		both(fmt.Sprintf("layer %d MLP", i), func() engine.Tensor { return l.MLP.Forward(c, x) })
		blocks := []engine.Tensor{rnd(1, 1, hidden), rnd(1, 1, hidden)}
		both(fmt.Sprintf("layer %d depth mix", i), func() engine.Tensor {
			c.stacks = nil
			return c.depthAttention(blocks, x, l.MLPQueryT)
		})
		if l.MLPGate != nil {
			both(fmt.Sprintf("layer %d message gate", i), func() engine.Tensor { return l.MLPGate.Forward(c, x, msg) })
			// The query, key and value gates as one batch against each alone,
			// on three inputs and on one shared by the three.
			gates := []*MessageGate{l.QGate, l.KGate, l.VGate}
			three, msgs := rnd(1, 1, 3, hidden), rnd(1, 1, 3, hidden)
			for _, in := range []engine.Tensor{three, x} {
				both(fmt.Sprintf("layer %d gates in a batch", i), func() engine.Tensor {
					if c.rel != nil {
						var d, f, a []engine.Tensor
						for _, g := range gates {
							d, f, a = append(d, g.DownT), append(f, g.ForgetT), append(a, g.AdmitT)
						}
						return c.rel.GateMixRows(in, msgs, d, f, a, defaultEps)
					}
					var out engine.Tensor
					for j, g := range gates {
						h := in
						if len(in.Shape()) == 4 {
							h = c.pick(in, j)
						}
						row := c.ExpandDims(g.Forward(c, h, c.pick(msgs, j)), 2)
						if out == nil {
							out = row
						} else {
							out = c.Concat(out, row, 2)
						}
					}
					return out
				})
			}
		}
		both(fmt.Sprintf("layer %d clip", i), func() engine.Tensor {
			if c.rel != nil {
				return c.ScalarMul(c.l2unit(x, nil, 3), 3)
			}
			return c.l2clip(x, nil, 3)
		})
		// A convolution over a tail of three earlier inputs: the cache is
		// filled by the model's own path, then one token each way.
		prime := rnd(1, 3, hidden)
		conv := func() engine.Tensor {
			vc := &VisitCache{rel: c.rel}
			vc.begin(3)
			c.shortConv(prime, l.MLPConv, vc, 3)
			vc.begin(1)
			return c.shortConv(x, l.MLPConv, vc, 3)
		}
		both(fmt.Sprintf("layer %d convolution", i), conv)
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
