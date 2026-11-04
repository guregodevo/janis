//go:build darwin && arm64 && cgo

package qwen

import (
	"testing"

	"github.com/guregodevo/janis/engine"
	"github.com/guregodevo/janis/mlxc"
)

// THE IN-PLACE KV CACHE HOLDS WHAT CONCATENATION WOULD.
//
// On MLX a step's keys and values are written into a buffer with spare room
// (mlx-lm's KVCache) instead of concatenated: a concatenation copied every
// layer's whole cache every token. Appends that cross the buffer's size, a
// cut back for prefix reuse, and a cache restored as a bare view (a saved
// session) must all read back as the concatenation of what went in.
func TestTheInPlaceKVCacheHoldsWhatConcatenationWould(t *testing.T) {
	b := mlxc.New()
	defer b.Close()
	const H, D = 2, 4
	step := func(seed, L int) engine.Tensor {
		data := make([]float32, H*L*D)
		for i := range data {
			data[i] = float32(seed*1000 + i)
		}
		return b.Cast(b.FromFloats(data, 1, H, L, D), engine.BF16)
	}
	check := func(what string, c *KVCache, want engine.Tensor) {
		t.Helper()
		if c.Len() != want.Shape()[2] {
			t.Fatalf("%s: %d tokens cached, want %d", what, c.Len(), want.Shape()[2])
		}
		got, exp := b.Floats(b.Cast(c.K, engine.F32)), b.Floats(b.Cast(want, engine.F32))
		for i := range exp {
			if got[i] != exp[i] {
				t.Fatalf("%s: element %d is %v, want %v", what, i, got[i], exp[i])
			}
		}
	}

	c := &KVCache{}
	var want engine.Tensor
	for i, L := range []int{100, 1, 1, 300, 1} { // 402 tokens: past one 256-token step
		x := step(i, L)
		c.Update(b, x, x)
		if want == nil {
			want = x
		} else {
			want = b.Concat(want, x, 2)
		}
		check("append", c, want)
	}

	c.Truncate(b, 150)
	want = b.Slice(want, 2, 0, 150)
	x := step(9, 7)
	c.Update(b, x, x)
	want = b.Concat(want, x, 2)
	check("append after a cut", c, want)

	restored := &KVCache{K: c.K, V: c.V}
	x = step(10, 3)
	restored.Update(b, x, x)
	check("append to a restored view", restored, b.Concat(want, x, 2))
}
