package cpu

import (
	"math"

	"memdoor/llm/engine"
)

// RoPE applies rotary position embedding to the last `dims` features of x,
// positions running offset..offset+L along the second-to-last axis. Matches the
// MLX convention: non-traditional (NeoX) rotates feature i against feature
// i+dims/2; traditional (GPT-J) rotates the interleaved pair (2i, 2i+1).
func (b *Backend) RoPE(x engine.Tensor, dims int, traditional bool, base, scale float32, offset int) engine.Tensor {
	inv := make([]float32, dims/2)
	for i := range inv {
		inv[i] = float32(math.Pow(float64(base), -float64(2*i)/float64(dims)))
	}
	return b.applyRoPE(x, dims, traditional, scale, offset, inv)
}

// RoPEFreqs is RoPE with explicit per-dim inverse frequencies (base ignored) —
// llama3 rope scaling supplies these directly.
func (b *Backend) RoPEFreqs(x engine.Tensor, dims int, traditional bool, scale float32, offset int, freqs engine.Tensor) engine.Tensor {
	inv := append([]float32(nil), as(freqs).data...)
	return b.applyRoPE(x, dims, traditional, scale, offset, inv)
}

func (b *Backend) applyRoPE(x engine.Tensor, dims int, traditional bool, scale float32, offset int, inv []float32) engine.Tensor {
	tx := as(x)
	D := tx.shape[len(tx.shape)-1]
	L := tx.shape[len(tx.shape)-2]
	out := newTensor(engine.F32, tx.shape...)
	copy(out.data, tx.data) // features beyond `dims` pass through unchanged
	half := dims / 2
	rows := len(tx.data) / D // total (outer*L) rows of length D, row-major
	for r := 0; r < rows; r++ {
		pos := float32(offset + r%L) // L is the second-to-last axis
		o := r * D
		for i := 0; i < half; i++ {
			theta := pos * scale * inv[i]
			c := float32(math.Cos(float64(theta)))
			s := float32(math.Sin(float64(theta)))
			var i0, i1 int
			if traditional {
				i0, i1 = o+2*i, o+2*i+1
			} else {
				i0, i1 = o+i, o+i+half
			}
			a, b2 := tx.data[i0], tx.data[i1]
			out.data[i0] = a*c - b2*s
			out.data[i1] = a*s + b2*c
		}
	}
	return out
}

// SDPA is scaled dot-product attention over [B, heads, L, dim]. GQA: when q has
// more heads than k/v, query head h shares kv head h/(Hq/Hk). causal applies the
// lower-triangular mask, offset by (Lk-Lq) so a 1-query decode step attends to
// the whole cache.
func (b *Backend) SDPA(q, k, v engine.Tensor, scale float32, causal bool) engine.Tensor {
	tq, tk, tv := as(q), as(k), as(v)
	B, Hq, Lq, Dh := tq.shape[0], tq.shape[1], tq.shape[2], tq.shape[3]
	Hk, Lk := tk.shape[1], tk.shape[2]
	group := Hq / Hk
	out := newTensor(engine.F32, B, Hq, Lq, Dh)
	qs, ks, vs, os := strides(tq.shape), strides(tk.shape), strides(tv.shape), strides(out.shape)
	delta := Lk - Lq // causal offset for KV-cache decode

	scores := make([]float32, Lk)
	for bi := 0; bi < B; bi++ {
		for h := 0; h < Hq; h++ {
			hk := h / group
			for i := 0; i < Lq; i++ {
				limit := Lk
				if causal {
					limit = delta + i + 1
				}
				qoff := bi*qs[0] + h*qs[1] + i*qs[2]
				mx := float32(math.Inf(-1))
				for j := 0; j < limit; j++ {
					koff := bi*ks[0] + hk*ks[1] + j*ks[2]
					var s float32
					for d := 0; d < Dh; d++ {
						s += tq.data[qoff+d] * tk.data[koff+d]
					}
					s *= scale
					scores[j] = s
					if s > mx {
						mx = s
					}
				}
				var sum float32
				for j := 0; j < limit; j++ {
					e := float32(math.Exp(float64(scores[j] - mx)))
					scores[j] = e
					sum += e
				}
				ooff := bi*os[0] + h*os[1] + i*os[2]
				for d := 0; d < Dh; d++ {
					var acc float32
					for j := 0; j < limit; j++ {
						voff := bi*vs[0] + hk*vs[1] + j*vs[2]
						acc += scores[j] / sum * tv.data[voff+d]
					}
					out.data[ooff+d] = acc
				}
			}
		}
	}
	return out
}
