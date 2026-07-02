package qwen

import (
	"math/rand"
	"testing"
)

// onlyGrammar is a test Grammar that is always Active and allows exactly one token.
type onlyGrammar struct{ allow int32 }

func (g onlyGrammar) Active() bool         { return true }
func (g onlyGrammar) Allows(id int32) bool { return id == g.allow }
func (g onlyGrammar) Advance(id int32)     {}

func TestSampleTokenGrammarGreedy(t *testing.T) {
	logits := []float32{5, 1, 9, 2} // argmax is index 2
	rng := rand.New(rand.NewSource(1))

	// No grammar: greedy picks the argmax (unchanged behavior).
	if got := sampleToken(logits, SampleParams{Temp: 0}, rng); got != 2 {
		t.Fatalf("nil grammar greedy: want argmax 2, got %d", got)
	}

	// Grammar allows only token 0: greedy must pick 0 despite 2 being the argmax.
	got := sampleToken(logits, SampleParams{Temp: 0, Grammar: onlyGrammar{allow: 0}}, rng)
	if got != 0 {
		t.Fatalf("constrained greedy: want the only allowed token 0, got %d", got)
	}
}

func TestSampleTokenGrammarSampled(t *testing.T) {
	logits := []float32{5, 1, 9, 2}
	// With temperature sampling and a grammar allowing only token 1, every draw
	// must return 1 regardless of the RNG.
	for seed := int64(0); seed < 20; seed++ {
		rng := rand.New(rand.NewSource(seed))
		got := sampleToken(logits, SampleParams{Temp: 0.8, TopP: 1, Grammar: onlyGrammar{allow: 1}}, rng)
		if got != 1 {
			t.Fatalf("seed %d: constrained sampling must return the only allowed token 1, got %d", seed, got)
		}
	}
}

func TestSampleTokenNilGrammarUnchanged(t *testing.T) {
	// A nil grammar must not perturb the sampled distribution: the same seed yields
	// the same token with and without an explicit (but inactive) grammar absent.
	logits := []float32{2, 2, 2, 2}
	for seed := int64(0); seed < 10; seed++ {
		a := sampleToken(logits, SampleParams{Temp: 1, TopP: 1}, rand.New(rand.NewSource(seed)))
		b := sampleToken(logits, SampleParams{Temp: 1, TopP: 1}, rand.New(rand.NewSource(seed)))
		if a != b {
			t.Fatalf("seed %d: nil-grammar sampling not deterministic: %d vs %d", seed, a, b)
		}
	}
}
