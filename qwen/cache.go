package qwen

import "github.com/guregodevo/janis/engine"

// LayerCache is one decoder layer's cross-step state. KV attention layers
// cache keys/values [B, n_kv_heads, T, head_dim]; linear-attention layers
// (qwen3_5 Gated DeltaNet) keep a recurrent state instead. The session's
// generate loop only needs length, truncation, and the live tensors to
// pin/sweep — everything else stays behind the implementing type.
type LayerCache interface {
	// Len is the number of cached time steps.
	Len() int
	// CanTruncate reports whether the cache can be cut back to exactly n
	// steps. KV caches always can; a recurrent state only at its ends (a
	// full reset, or a no-op) — mid-history states were never stored.
	CanTruncate(n int) bool
	// Truncate cuts the cache back to n steps (n == 0 resets it).
	Truncate(b engine.Backend, n int)
	// Mark keeps the state at the current length as a point Truncate can
	// return to. A KV cache can return anywhere and keeps nothing; a
	// recurrent state keeps a copy of itself. Truncate drops the marks past
	// the length it cuts to; Forget drops the one at n.
	Mark()
	Forget(n int)
	// Tensors returns the live state tensors, the marks' included (for
	// Pin/Unpin/Eval).
	Tensors() []engine.Tensor
}

// KVCache holds the accumulated key/value tensors for one attention layer,
// shaped [B, n_kv_heads, T, head_dim] and growing along axis 2 (time).
type KVCache struct {
	K, V engine.Tensor

	// kBuf, vBuf hold K and V with spare room on a backend that appends in
	// place (engine.KVAppender); K and V are views of their first Len()
	// tokens. Nil elsewhere, and after a reset.
	kBuf, vBuf engine.Tensor
}

// Len is the number of cached time steps.
func (c *KVCache) Len() int {
	if c.K == nil {
		return 0
	}
	return c.K.Shape()[2]
}

// CanTruncate: a KV cache can be cut to any prefix.
func (c *KVCache) CanTruncate(int) bool { return true }

// Mark keeps nothing: Truncate reaches any prefix already.
func (c *KVCache) Mark() {}

func (c *KVCache) Forget(int) {}

// Truncate keeps only the first n time steps (for prefix reuse across requests).
func (c *KVCache) Truncate(b engine.Backend, n int) {
	if c.K == nil {
		return
	}
	if n <= 0 {
		c.K, c.V, c.kBuf, c.vBuf = nil, nil, nil, nil
		return
	}
	if n >= c.K.Shape()[2] {
		return
	}
	c.K = b.Slice(c.K, 2, 0, n)
	c.V = b.Slice(c.V, 2, 0, n)
}

// Tensors returns the live K/V tensors (none while empty).
func (c *KVCache) Tensors() []engine.Tensor {
	if c.K == nil {
		return nil
	}
	if c.kBuf != nil {
		return []engine.Tensor{c.K, c.V, c.kBuf, c.vBuf}
	}
	return []engine.Tensor{c.K, c.V}
}

// Update appends the new step's k/v and returns the full cached k/v.
// Update appends this step's keys and values. A backend whose activations
// are f32 (ggml) hands them over as f32; the cache keeps them as f16, half
// the bytes over a long prompt (a 6k-token coder prompt on the 30B MoE is
// 1.2 GB of f32 cache, more than an L4 has left). MLX hands over the
// model's own dtype already and is left alone.
func (c *KVCache) Update(b engine.Backend, kNew, vNew engine.Tensor) (engine.Tensor, engine.Tensor) {
	if rr, ok := b.(engine.RawReader); ok && rr.DTypeOf(kNew) == engine.F32 {
		kNew, vNew = b.Cast(kNew, engine.F16), b.Cast(vNew, engine.F16)
	}
	if ap, ok := b.(engine.KVAppender); ok {
		n := c.Len()
		c.kBuf, c.K = ap.AppendKV(c.kBuf, c.K, n, kNew)
		c.vBuf, c.V = ap.AppendKV(c.vBuf, c.V, n, vNew)
		return c.K, c.V
	}
	if c.K == nil {
		c.K, c.V = kNew, vNew
	} else {
		c.K = b.Concat(c.K, kNew, 2)
		if vc, ok := b.(engine.ValueCache); ok {
			c.V = vc.AppendValues(c.V, vNew)
		} else {
			c.V = b.Concat(c.V, vNew, 2)
		}
	}
	return c.K, c.V
}
