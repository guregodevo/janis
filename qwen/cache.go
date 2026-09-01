package qwen

import "memdoor/llm/engine"

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
	// Tensors returns the live state tensors (for Pin/Unpin/Eval).
	Tensors() []engine.Tensor
}

// KVCache holds the accumulated key/value tensors for one attention layer,
// shaped [B, n_kv_heads, T, head_dim] and growing along axis 2 (time).
type KVCache struct {
	K, V engine.Tensor
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

// Truncate keeps only the first n time steps (for prefix reuse across requests).
func (c *KVCache) Truncate(b engine.Backend, n int) {
	if c.K == nil {
		return
	}
	if n <= 0 {
		c.K, c.V = nil, nil
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
	return []engine.Tensor{c.K, c.V}
}

// Update appends the new step's k/v and returns the full cached k/v.
func (c *KVCache) Update(b engine.Backend, kNew, vNew engine.Tensor) (engine.Tensor, engine.Tensor) {
	if c.K == nil {
		c.K, c.V = kNew, vNew
	} else {
		c.K = b.Concat(c.K, kNew, 2)
		c.V = b.Concat(c.V, vNew, 2)
	}
	return c.K, c.V
}
