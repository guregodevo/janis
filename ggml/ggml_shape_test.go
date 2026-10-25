//go:build ggml

package ggml

import (
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
