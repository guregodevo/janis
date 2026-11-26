package qwen

import (
	"math"
	"math/rand"
	"sort"
	"testing"
)

// refTopP is the draw as it was: a softmax over every token, the tokens in
// order of probability, the top-p share of them, one random number.
func refTopP(logits []float32, temp, topP float32, rng *rand.Rand) int32 {
	maxL := logits[0]
	for _, x := range logits {
		maxL = max(maxL, x)
	}
	probs := make([]float64, len(logits))
	var sum float64
	for i, x := range logits {
		probs[i] = math.Exp((float64(x) - float64(maxL)) / float64(temp))
		sum += probs[i]
	}
	idx := make([]int, len(logits))
	for i := range idx {
		probs[i] /= sum
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool { return probs[idx[a]] > probs[idx[b]] })
	tp := float64(topP)
	if tp <= 0 || tp > 1 {
		tp = 1
	}
	var cum float64
	cut := len(idx)
	for rank, id := range idx {
		cum += probs[id]
		if cum >= tp {
			cut = rank + 1
			break
		}
	}
	r := rng.Float64() * cum
	var acc float64
	for rank := 0; rank < cut; rank++ {
		acc += probs[idx[rank]]
		if r <= acc {
			return int32(idx[rank])
		}
	}
	return int32(idx[cut-1])
}

// The draw without a top-k picks the token the full softmax and sort pick,
// for the same random number: peaked and flat logits, a small vocabulary
// and one summed on several cores, with and without a top-p.
func TestTopPDrawsWhatTheFullSoftmaxDraws(t *testing.T) {
	src := rand.New(rand.NewSource(5))
	for _, n := range []int{50, 3000, 70000} {
		for _, spread := range []float64{0.5, 3, 8} {
			logits := make([]float32, n)
			for i := range logits {
				logits[i] = float32(src.NormFloat64() * spread)
			}
			for _, temp := range []float32{0.6, 1, 1.5} {
				for _, topP := range []float32{0.5, 0.95, 1, 0} {
					for seed := int64(1); seed <= 3; seed++ {
						want := refTopP(logits, temp, topP, rand.New(rand.NewSource(seed)))
						got := sampleToken(append([]float32(nil), logits...), SampleParams{Temp: temp, TopP: topP}, rand.New(rand.NewSource(seed)), nil)
						if got != want {
							t.Fatalf("n=%d spread=%g temp=%g top-p=%g seed=%d: drew %d, the full softmax draws %d", n, spread, temp, topP, seed, got, want)
						}
					}
				}
			}
		}
	}
}
