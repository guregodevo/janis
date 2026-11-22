//go:build darwin && arm64 && cgo

package qwen35

import (
	"testing"
	"time"

	"github.com/guregodevo/janis/qwen"
)

// Not a check: tokens per second of a batched decode step at a few batch sizes.
func TestBatchedDecodeSpeed(t *testing.T) {
	b, m := smallModel(t)
	prompt := []int32{785, 6722, 315, 9625, 374, 40, 1079, 264, 2632, 315, 279, 1879, 13}
	for _, B := range []int{1, 4, 8, 16} {
		caches := make([][]qwen.LayerCache, B)
		offsets := make([]int, B)
		ids := make([]int32, B)
		for j := range caches {
			s := m.NewSession()
			s.Prefill(b, prompt, nil)
			caches[j], offsets[j], ids[j] = s.Caches, len(prompt), 264
		}
		const steps = 20
		start := time.Now()
		for n := 0; n < steps; n++ {
			b.Floats(m.ForwardBatch(b, ids, offsets, caches))
			for j := range offsets {
				offsets[j]++
			}
		}
		el := time.Since(start).Seconds()
		t.Logf("B=%2d: %.1f ms a step, %.0f tok/s", B, 1000*el/steps, float64(B*steps)/el)
	}
}
