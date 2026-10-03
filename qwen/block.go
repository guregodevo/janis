package qwen

import (
	"fmt"

	"github.com/guregodevo/janis/engine"
	"github.com/guregodevo/janis/safetensors"
)

// feedForward is the per-layer FFN: a dense MLP or a sparse MoEBlock.
type feedForward interface {
	Forward(b engine.Backend, x engine.Tensor) engine.Tensor
}

// Block is a Qwen2 decoder block: pre-attention norm + attention + residual,
// then pre-FFN norm + FFN (dense MLP or sparse MoE) + residual.
type Block struct {
	InputNorm *RMSNorm
	Attn      *Attention
	PostNorm  *RMSNorm
	FFN       feedForward
}

// LoadBlock loads decoder block i (prefix "model.layers.i"). For MoE models a
// non-nil store supplies the layer's experts; layers selected by
// decoder_sparse_step use a MoEBlock, the rest a dense MLP.
func LoadBlock(b engine.Backend, st *safetensors.File, i int, cfg Config, store *ExpertStore) (*Block, error) {
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
	var ffn feedForward
	if cfg.NumExperts > 0 && cfg.SparseStep > 0 && (i+1)%cfg.SparseStep == 0 {
		if ffn, err = LoadMoEBlock(b, st, i, cfg, store); err != nil {
			return nil, err
		}
	} else if ffn, err = LoadMLP(b, st, prefix+".mlp", cfg.GroupSize, cfg.Bits); err != nil {
		return nil, err
	}
	return &Block{InputNorm: inNorm, Attn: attn, PostNorm: postNorm, FFN: ffn}, nil
}

// Forward runs the block on x [B, L, hidden] -> [B, L, hidden] (no cache).
func (bl *Block) Forward(b engine.Backend, x engine.Tensor) engine.Tensor {
	return bl.ForwardCached(b, x, nil, 0)
}

// ForwardCached runs the block with an optional KV cache and position offset.
func (bl *Block) ForwardCached(b engine.Backend, x engine.Tensor, cache *KVCache, offset int) engine.Tensor {
	h := b.Add(x, bl.Attn.ForwardCached(b, bl.InputNorm.Forward(b, x), cache, offset))
	return b.Add(h, bl.FFN.Forward(b, bl.PostNorm.Forward(b, h)))
}
