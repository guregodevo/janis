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

// An 8-bit affine weight (the MoE routers) has no ggml block type: it goes
// to the device as f16 and must still match the oracle.
func TestEightBitWeightsMatchCPU(t *testing.T) {
	g := New()
	defer g.Close()
	c := cpu.New()
	const out, in, gs, bits = 16, 128, 64, 8
	P, G := in*bits/32, in/gs
	raw := rndPackedU32(out*P, 41)
	sc, bi := rnd(out*G, 42), rnd(out*G, 43)
	x := rnd(3*in, 44)
	gW, cW := g.FromRaw(engine.U32, raw, out, P), c.FromRaw(engine.U32, raw, out, P)
	gS, gB := g.FromFloats(sc, out, G), g.FromFloats(bi, out, G)
	cS, cB := c.FromFloats(sc, out, G), c.FromFloats(bi, out, G)
	relOK(t, "QuantMatmul 8-bit (f16 on the device)", 5e-3,
		g.Floats(g.QuantMatmul(g.FromFloats(x, 1, 3, in), gW, gS, gB, true, gs, bits)),
		c.Floats(c.QuantMatmul(c.FromFloats(x, 1, 3, in), cW, cS, cB, true, gs, bits)))
}

// The MoE block as moeApply runs it, at the 30B's widths: gate and up
// gathers on one token, SiLU, the product, the down gather, the weighted
// mean. Against the CPU backend, so a device's kernels are tested too.
func TestMoEBlockMatchesCPU(t *testing.T) {
	g := New()
	defer g.Close()
	c := cpu.New()
	t.Log(g.Version())
	const E, hidden, inter, gs, bits, K = 16, 2048, 768, 64, 4, 8
	mk := func(rows, in int, seed int64) (engine.Tensor, engine.Tensor, engine.Tensor, engine.Tensor, engine.Tensor, engine.Tensor) {
		P, G := in*bits/32, in/gs
		raw := rndPackedU32(E*rows*P, seed)
		sc, bi := rnd(E*rows*G, seed+1), rnd(E*rows*G, seed+2)
		return g.FromRaw(engine.U32, raw, E, rows, P), g.FromFloats(sc, E, rows, G), g.FromFloats(bi, E, rows, G),
			c.FromRaw(engine.U32, raw, E, rows, P), c.FromFloats(sc, E, rows, G), c.FromFloats(bi, E, rows, G)
	}
	gGW, gGS, gGB, cGW, cGS, cGB := mk(inter, hidden, 51) // gate: hidden -> inter
	gUW, gUS, gUB, cUW, cUS, cUB := mk(inter, hidden, 61) // up
	gDW, gDS, gDB, cDW, cDS, cDB := mk(hidden, inter, 71) // down: inter -> hidden
	ids := []int32{3, 0, 15, 7, 9, 1, 12, 5}
	scores := []float32{0.3, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05}
	x := rnd(hidden, 81)
	run := func(b engine.Backend, w, s, bb, uw, us, ub, dw, ds, db engine.Tensor) []float32 {
		xe := b.FromFloats(x, 1, 1, 1, 1, hidden)
		ri := b.FromInt32(ids, 1, 1, K)
		gt := b.GatherQuantMatmul(xe, w, s, bb, ri, true, gs, bits)
		ut := b.GatherQuantMatmul(xe, uw, us, ub, ri, true, gs, bits)
		hh := b.Mul(b.SiLU(gt), ut)
		d := b.GatherQuantMatmul(hh, dw, ds, db, ri, true, gs, bits)
		d2 := b.Reshape(d, 1, 1, K, hidden)
		sc := b.FromFloats(scores, 1, 1, K, 1)
		return b.Floats(b.ScalarMul(b.Mean(b.Mul(d2, sc), 2), float32(K)))
	}
	got := run(g, gGW, gGS, gGB, gUW, gUS, gUB, gDW, gDS, gDB)
	want := run(c, cGW, cGS, cGB, cUW, cUS, cUB, cDW, cDS, cDB)
	bad := 0
	for _, v := range got {
		if v != v || v > 1e30 || v < -1e30 {
			bad++
		}
	}
	if bad > 0 {
		t.Fatalf("%d of %d NaN/Inf in the MoE block output on %s", bad, len(got), g.Version())
	}
	relOK(t, "MoE block (gate, up, SiLU, product, down, weighted mean)", 3e-2, got, want)
}

// An input with a channel past 2,000 (Qwen3's attention outputs have them)
// overflows ggml's q8_1 block sum on a Q4_1 weight; the weight that meets
// such an input is kept as Q8_0 and must still match the oracle.
func TestAnOutlierInputKeepsTheWeightExact(t *testing.T) {
	g := New()
	defer g.Close()
	c := cpu.New()
	const out, in, gs, bits = 32, 512, 64, 4
	P, G := in*bits/32, in/gs
	raw := rndPackedU32(out*P, 91)
	sc, bi := rnd(out*G, 92), rnd(out*G, 93)
	x := rnd(in, 94)
	x[37] = 5000 // one outlier channel
	gW, cW := g.FromRaw(engine.U32, raw, out, P), c.FromRaw(engine.U32, raw, out, P)
	gS, gB := g.FromFloats(sc, out, G), g.FromFloats(bi, out, G)
	cS, cB := c.FromFloats(sc, out, G), c.FromFloats(bi, out, G)
	got := g.Floats(g.QuantMatmul(g.FromFloats(x, 1, 1, in), gW, gS, gB, true, gs, bits))
	for _, v := range got {
		if v != v {
			t.Fatal("NaN in the output: the q8_1 overflow")
		}
	}
	relOK(t, "QuantMatmul with an outlier input (Q8_0 weight)", 1e-2, got,
		c.Floats(c.QuantMatmul(c.FromFloats(x, 1, 1, in), cW, cS, cB, true, gs, bits)))
	if gW.(*tensor).q4.t._type != 8 { // GGML_TYPE_Q8_0
		t.Fatalf("the weight that met the outlier is Q8_0, got type %d", int(gW.(*tensor).q4.t._type))
	}
}

// A pinned result leaves its batch (its own buffer, the same values) and
// is still right in a later graph after the batch's other results are
// swept.
func TestPinnedResultsLeaveTheirBatch(t *testing.T) {
	g := New()
	defer g.Close()
	a := g.FromFloats(rnd(64, 101), 8, 8)
	sq := g.Mul(a, a) // a result in a batch, with others
	tmp := g.SiLU(a)  // the same batch
	g.Eval(sq, tmp)
	want := g.Floats(sq)
	g.Pin(sq)
	if x := sq.(*tensor); x.grp != nil || x.buf == nil {
		t.Fatalf("pinned: own buffer, no batch: grp=%v buf=%v", x.grp, x.buf)
	}
	g.Sweep() // frees tmp and the batch
	got := g.Floats(g.Add(sq, g.FromFloats(make([]float32, 64), 8, 8)))
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("value %d changed after the batch was swept: %v vs %v", i, got[i], want[i])
		}
	}
}
