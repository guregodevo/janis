//go:build ggml

package ggml

import (
	"fmt"
	"testing"

	"github.com/guregodevo/janis/cpu"
	"github.com/guregodevo/janis/engine"
)

// The q_proj of Qwen3-0.6B: [2048, 1024] affine 4-bit against 13 tokens in
// a 3-D batch, the first quantized matmul of a real forward.
func TestQuantMatmulAtModelShape(t *testing.T) {
	g := New()
	defer g.Close()
	c := cpu.New()
	const out, in, gs, bits = 2048, 1024, 64, 4
	P, G := in*bits/32, in/gs
	raw := rndPackedU32(out*P, 1)
	sc, bi := rnd(out*G, 2), rnd(out*G, 3)
	x := rnd(1*13*in, 4)
	gW, cW := g.FromRaw(engine.U32, raw, out, P), c.FromRaw(engine.U32, raw, out, P)
	gS, gB := g.Cast(g.FromFloats(sc, out, G), engine.F16), g.Cast(g.FromFloats(bi, out, G), engine.F16)
	cS, cB := c.FromFloats(sc, out, G), c.FromFloats(bi, out, G)
	relOK(t, "QuantMatmul(M=13, 3-D) on Q4_1", 1.5e-2,
		g.Floats(g.QuantMatmul(g.FromFloats(x, 1, 13, in), gW, gS, gB, true, gs, bits)),
		c.Floats(c.QuantMatmul(c.FromFloats(x, 1, 13, in), cW, cS, cB, true, gs, bits)))
	x1 := rnd(in, 5)
	relOK(t, "QuantMatmul(M=1, 3-D) on Q4_1", 1.5e-2,
		g.Floats(g.QuantMatmul(g.FromFloats(x1, 1, 1, in), gW, gS, gB, true, gs, bits)),
		c.Floats(c.QuantMatmul(c.FromFloats(x1, 1, 1, in), cW, cS, cB, true, gs, bits)))
}

// The MoE gather: a stack of E experts, K picked per token by global id,
// the gate/up shape (one input vector per token shared by the K slots) and
// the down shape (one vector per slot). Against the CPU backend's gather.
func TestGatherQuantMatmulMatchesCPU(t *testing.T) {
	g := New()
	defer g.Close()
	c := cpu.New()
	const E, out, in, gs, bits, K, seq = 4, 8, 128, 64, 4, 2, 3
	P, G := in*bits/32, in/gs
	raw := rndPackedU32(E*out*P, 31)
	sc, bi := rnd(E*out*G, 32), rnd(E*out*G, 33)
	ids := []int32{0, 3, 1, 2, 3, 0} // [1, seq, K]
	gW, cW := g.FromRaw(engine.U32, raw, E, out, P), c.FromRaw(engine.U32, raw, E, out, P)
	gS, gB := g.FromFloats(sc, E, out, G), g.FromFloats(bi, E, out, G)
	cS, cB := c.FromFloats(sc, E, out, G), c.FromFloats(bi, E, out, G)
	gI, cI := g.FromInt32(ids, 1, seq, K), c.FromInt32(ids, 1, seq, K)
	x := rnd(seq*in, 34)
	relOK(t, "gather, one vector per token (gate/up)", 1.5e-2,
		g.Floats(g.GatherQuantMatmul(g.FromFloats(x, 1, seq, 1, 1, in), gW, gS, gB, gI, true, gs, bits)),
		c.Floats(c.GatherQuantMatmul(c.FromFloats(x, 1, seq, 1, 1, in), cW, cS, cB, cI, true, gs, bits)))
	xk := rnd(seq*K*in, 35)
	relOK(t, "gather, one vector per slot (down)", 1.5e-2,
		g.Floats(g.GatherQuantMatmul(g.FromFloats(xk, 1, seq, K, 1, in), gW, gS, gB, gI, true, gs, bits)),
		c.Floats(c.GatherQuantMatmul(c.FromFloats(xk, 1, seq, K, 1, in), cW, cS, cB, cI, true, gs, bits)))
	if got := g.GatherQuantMatmul(g.FromFloats(x, 1, seq, 1, 1, in), gW, gS, gB, gI, true, gs, bits).Shape(); fmt.Sprint(got) != fmt.Sprint([]int{1, seq, K, 1, out}) {
		t.Fatalf("shape [lead…, K, 1, out]: %v", got)
	}
}
