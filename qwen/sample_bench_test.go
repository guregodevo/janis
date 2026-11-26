package qwen

import (
	"math/rand"
	"testing"
)

// BenchmarkSampleToken: one sampled token over a 248k vocabulary at the
// server's defaults (temperature 0.6, top-p 0.95, repeat penalty 1.1).
func BenchmarkSampleToken(b *testing.B) {
	src := rand.New(rand.NewSource(1))
	logits := make([]float32, 248320)
	for i := range logits {
		logits[i] = float32(src.NormFloat64() * 3)
	}
	logits[1234] = 30
	recent := make([]int32, 64)
	for i := range recent {
		recent[i] = int32(src.Intn(len(logits)))
	}
	p := SampleParams{Temp: 0.6, TopP: 0.95, RepeatPenalty: 1.1, RepeatLastN: 64}
	rng := rand.New(rand.NewSource(2))
	buf := make([]float32, len(logits))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(buf, logits)
		sampleToken(buf, p, rng, recent)
	}
}

// BenchmarkSampleTokenTopK: the same with Qwen3.8's own sampling
// (temperature 1, top-k 20, top-p 0.95).
func BenchmarkSampleTokenTopK(b *testing.B) {
	src := rand.New(rand.NewSource(1))
	logits := make([]float32, 248320)
	for i := range logits {
		logits[i] = float32(src.NormFloat64() * 3)
	}
	logits[1234] = 30
	p := SampleParams{Temp: 1, TopP: 0.95, TopK: 20}
	rng := rand.New(rand.NewSource(2))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sampleToken(logits, p, rng, nil)
	}
}
