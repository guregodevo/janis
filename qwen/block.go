package qwen

import (
	"fmt"

	"memdoor/llm/engine"
	"memdoor/llm/safetensors"
)

// Block is a Qwen2 decoder block: pre-attention norm + attention + residual,
// then pre-MLP norm + MLP + residual.
type Block struct {
	InputNorm *RMSNorm
	Attn      *Attention
	PostNorm  *RMSNorm
	MLP       *MLP
}

// LoadBlock loads decoder block i (prefix "model.layers.i").
func LoadBlock(b engine.Backend, st *safetensors.File, i int, cfg Config) (*Block, error) {
	prefix := fmt.Sprintf("model.layers.%d", i)
	inNorm, err := LoadRMSNorm(b, st, prefix+".input_layernorm", cfg.RMSEps)
	if err != nil {
		return nil, err
	}
	attn, err := LoadAttention(b, st, prefix+".self_attn", cfg)
	if err != nil {
		return nil, err
	}
	postNorm, err := LoadRMSNorm(b, st, prefix+".post_attention_layernorm", cfg.RMSEps)
	if err != nil {
		return nil, err
	}
	mlp, err := LoadMLP(b, st, prefix+".mlp", cfg.GroupSize, cfg.Bits)
	if err != nil {
		return nil, err
	}
	return &Block{InputNorm: inNorm, Attn: attn, PostNorm: postNorm, MLP: mlp}, nil
}

// Forward runs the block on x [B, L, hidden] -> [B, L, hidden] (no cache).
func (bl *Block) Forward(b engine.Backend, x engine.Tensor) engine.Tensor {
	return bl.ForwardCached(b, x, nil, 0)
}

// ForwardCached runs the block with an optional KV cache and position offset.
func (bl *Block) ForwardCached(b engine.Backend, x engine.Tensor, cache *KVCache, offset int) engine.Tensor {
	h := b.Add(x, bl.Attn.ForwardCached(b, bl.InputNorm.Forward(b, x), cache, offset))
	return b.Add(h, bl.MLP.Forward(b, bl.PostNorm.Forward(b, h)))
}
