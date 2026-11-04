package qwen

import (
	"math"
	"math/rand"
	"sort"
	"testing"
)

// evenGrammar allows the even token ids.
type evenGrammar struct{}

func (evenGrammar) Active() bool         { return true }
func (evenGrammar) Allows(id int32) bool { return id%2 == 0 }
func (evenGrammar) Advance(int32)        {}
func (evenGrammar) InString() bool       { return false }

// referenceSample is the sampler as it was, with the whole vocabulary sorted,
// ties broken by token id.
func referenceSample(logits []float32, p SampleParams, r float64) int32 {
	maxL := logits[0]
	for _, x := range logits {
		maxL = float32(math.Max(float64(maxL), float64(x)))
	}
	probs := make([]float64, len(logits))
	var sum float64
	for i, x := range logits {
		probs[i] = math.Exp((float64(x) - float64(maxL)) / float64(p.Temp))
		sum += probs[i]
	}
	for i := range probs {
		probs[i] /= sum
	}
	var idx []int
	for i := range probs {
		if p.Grammar == nil || p.Grammar.Allows(int32(i)) {
			idx = append(idx, i)
		}
	}
	sort.SliceStable(idx, func(a, b int) bool { return probs[idx[a]] > probs[idx[b]] })
	var cum float64
	cut := len(idx)
	for rank, id := range idx {
		cum += probs[id]
		if cum >= float64(p.TopP) {
			cut = rank + 1
			break
		}
	}
	x := r * cum
	var acc float64
	for rank := 0; rank < cut; rank++ {
		acc += probs[idx[rank]]
		if x <= acc {
			return int32(idx[rank])
		}
	}
	return int32(idx[cut-1])
}

// THE TOP-P SAMPLER PICKS WHAT A FULL SORT WOULD.
//
// It ranks only the best candidates instead of sorting the vocabulary (18 ms
// a token at 248k). Logits on a coarse grid make ties, as bf16 logits do; a
// flat set needs more candidates than the first round keeps.
func TestTopPPicksWhatAFullSortWould(t *testing.T) {
	src := rand.New(rand.NewSource(7))
	for trial := 0; trial < 400; trial++ {
		n := 50 + src.Intn(3000)
		spread := []float64{0.05, 1, 8}[trial%3] // flat, middling, peaked
		logits := make([]float32, n)
		for i := range logits {
			logits[i] = float32(math.Round(src.NormFloat64()*spread*8) / 8)
		}
		p := SampleParams{Temp: []float32{0.3, 0.6, 1}[trial%3], TopP: []float32{0.5, 0.95, 1}[trial%2+trial%3/2]}
		if trial%4 == 0 {
			p.Grammar = evenGrammar{}
		}
		seed := src.Int63()
		got := sampleToken(append([]float32(nil), logits...), p, rand.New(rand.NewSource(seed)), nil)
		want := referenceSample(logits, p, rand.New(rand.NewSource(seed)).Float64())
		if got != want {
			t.Fatalf("trial %d (n=%d, spread %.2f, temp %.1f, top-p %.2f, grammar %v): got %d, a full sort gives %d",
				trial, n, spread, p.Temp, p.TopP, p.Grammar != nil, got, want)
		}
	}
}
