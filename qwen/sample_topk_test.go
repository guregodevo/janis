package qwen

import (
	"math/rand"
	"sort"
	"testing"
)

// Top-k keeps the K most likely tokens, and top-p is then a share of those:
// transformers' order (temperature, top-k, top-p).
func TestTopKKeepsTheKMostLikely(t *testing.T) {
	// Probabilities 0.4, 0.3, 0.2, 0.1 at temperature 1.
	logits := func() []float32 { return []float32{-0.9163, -1.2040, -1.6094, -2.3026} }
	draw := func(p SampleParams) map[int32]int {
		rng := rand.New(rand.NewSource(1))
		seen := map[int32]int{}
		for i := 0; i < 4000; i++ {
			seen[sampleToken(logits(), p, rng, nil)]++
		}
		return seen
	}
	if seen := draw(SampleParams{Temp: 1, TopP: 1}); len(seen) != 4 {
		t.Fatalf("no top-k: %d tokens drawn, want all 4", len(seen))
	}
	seen := draw(SampleParams{Temp: 1, TopP: 1, TopK: 2})
	if len(seen) != 2 || seen[0] == 0 || seen[1] == 0 {
		t.Fatalf("top-k 2: drew %v, want only tokens 0 and 1", seen)
	}
	if share := float64(seen[0]) / 4000; share < 0.53 || share > 0.61 {
		t.Fatalf("top-k 2: token 0 drawn %.3f of the time, want 0.4/0.7 = 0.571", share)
	}
	// Top-p 0.5 of the two kept (0.571, 0.429) is the first alone; of all
	// four (0.4, 0.3, …) it would be the first two.
	if seen := draw(SampleParams{Temp: 1, TopP: 0.5, TopK: 2}); len(seen) != 1 || seen[0] != 4000 {
		t.Fatalf("top-k 2, top-p 0.5: drew %v, want token 0 only", seen)
	}
	if seen := draw(SampleParams{Temp: 1, TopP: 0.5}); len(seen) != 2 {
		t.Fatalf("top-p 0.5: drew %v, want tokens 0 and 1", seen)
	}
}

// The K kept are the K a full sort keeps, ties to the lower id, at any K.
func TestTopProbsIsAFullSortsFirstK(t *testing.T) {
	src := rand.New(rand.NewSource(3))
	logits := make([]float32, 5000)
	for i := range logits {
		logits[i] = float32(src.Intn(400)) / 8 // many ties
	}
	order := make([]int, len(logits))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return logits[order[a]] > logits[order[b]] })
	for _, k := range []int{1, 2, 20, 64, 4999, 5000, 6000} {
		got := topProbs(logits, k, nil)
		want := order[:min(k, len(order))]
		if len(got) != len(want) {
			t.Fatalf("k=%d: %d kept, want %d", k, len(got), len(want))
		}
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("k=%d: rank %d is token %d, want %d", k, i, got[i], want[i])
			}
		}
	}
	even := func(id int) bool { return id%2 == 0 }
	for i, id := range topProbs(logits, 20, even) {
		if id%2 != 0 {
			t.Fatalf("rank %d is token %d, which the filter refuses", i, id)
		}
	}
}

// A greedy pick is the first of the largest logits.
func TestGreedyTakesTheFirstLargest(t *testing.T) {
	if got := sampleToken([]float32{1, 7, 3, 7, 2}, SampleParams{}, nil, nil); got != 1 {
		t.Fatalf("greedy picked %d, want 1", got)
	}
}
