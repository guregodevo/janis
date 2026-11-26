//go:build ggml

package ggml

import "testing"

// The prefill window follows the free memory and the prompt: the largest of
// 256 to 2,048 tokens whose forward fits in half of what is free.
func TestPrefillChunkFollowsFreeMemory(t *testing.T) {
	const gb = int64(1) << 30
	for _, tc := range []struct {
		name   string
		free   int64
		tokens int
		want   int
	}{
		{"an A100 with the 4-bit 27B loaded, 12k prompt", 58 * gb, 12000, 2048},
		{"a 32 GB card with the 27B loaded, 12k prompt", 11 * gb, 12000, 1024},
		{"a 24 GB card with the 27B loaded, 12k prompt", 3 * gb, 12000, 256},
		{"the same card, 1.5k prompt", 3 * gb, 1500, 1024},
		{"a 32 GB card, 100k prompt", 11 * gb, 100000, 256},
		{"nothing free", 0, 1000, 256},
		{"a device that names no memory", -1, 1000, 256},
	} {
		if got := prefillChunkFor(tc.free, tc.tokens); got != tc.want {
			t.Errorf("%s: %d tokens a forward, want %d", tc.name, got, tc.want)
		}
	}
}
