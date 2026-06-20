package qwen

import "memdoor/llm/engine"

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
