//go:build ggml

package ggml

import (
	"fmt"
	"math"
	"runtime"
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

// The KV cache is f16: attention over f16 keys and values, concatenated as
// the cache grows, matches the oracle's f32 within f16's precision.
func TestAttentionOverAnF16Cache(t *testing.T) {
	g := New()
	defer g.Close()
	c := cpu.New()
	q, k1, v1, k2, v2 := rnd(1*4*1*8, 111), rnd(1*2*5*8, 112), rnd(1*2*5*8, 113), rnd(1*2*1*8, 114), rnd(1*2*1*8, 115)
	scale := float32(1.0 / math.Sqrt(8))
	gk := g.Concat(g.Cast(g.FromFloats(k1, 1, 2, 5, 8), engine.F16), g.Cast(g.FromFloats(k2, 1, 2, 1, 8), engine.F16), 2)
	gv := g.Concat(g.Cast(g.FromFloats(v1, 1, 2, 5, 8), engine.F16), g.Cast(g.FromFloats(v2, 1, 2, 1, 8), engine.F16), 2)
	if gk.(*tensor).dt != engine.F16 || fmt.Sprint(gk.Shape()) != "[1 2 6 8]" {
		t.Fatalf("the concatenated cache stays f16 and grows: %v %v", gk.(*tensor).dt, gk.Shape())
	}
	ck := c.Concat(c.FromFloats(k1, 1, 2, 5, 8), c.FromFloats(k2, 1, 2, 1, 8), 2)
	cv := c.Concat(c.FromFloats(v1, 1, 2, 5, 8), c.FromFloats(v2, 1, 2, 1, 8), 2)
	diffOK(t, "SDPA over an f16 cache (decode, GQA)", 5e-3,
		g.Floats(g.SDPA(g.FromFloats(q, 1, 4, 1, 8), gk, gv, scale, true)),
		c.Floats(c.SDPA(c.FromFloats(q, 1, 4, 1, 8), ck, cv, scale, true)))
	// Truncate slices the cache; it stays f16.
	if s := g.Slice(gk, 2, 0, 3); s.(*tensor).dt != engine.F16 || fmt.Sprint(s.Shape()) != "[1 2 3 8]" {
		t.Fatalf("a sliced cache stays f16: %v %v", s.(*tensor).dt, s.Shape())
	}
}

// The cache grows a token at a time past its spare room, is truncated and
// grown again, and reads back exactly what the oracle's concatenations hold.
func TestTheCacheGrowsInPlace(t *testing.T) {
	g := New()
	defer g.Close()
	c := cpu.New()
	var gk, ck engine.Tensor
	for i := 0; i < cacheGrow+40; i++ {
		step := rnd(1*2*1*8, int64(1000+i))
		gs, cs := g.Cast(g.FromFloats(step, 1, 2, 1, 8), engine.F16), c.FromFloats(step, 1, 2, 1, 8)
		if gk == nil {
			gk, ck = gs, cs
			continue
		}
		gk, ck = g.Concat(gk, gs, 2), c.Concat(ck, cs, 2)
		if i%97 == 0 {
			g.Pin(gk)
			g.Eval(gk)
			g.Sweep()
			g.Unpin(gk)
		}
	}
	if fmt.Sprint(gk.Shape()) != fmt.Sprint(ck.Shape()) {
		t.Fatalf("shapes: %v vs %v", gk.Shape(), ck.Shape())
	}
	diffOK(t, "cache after growth past its room", 2e-3, g.Floats(gk), c.Floats(ck))
	// Truncate, then append: a copy, a new buffer, the same values.
	gk, ck = g.Slice(gk, 2, 0, 100), c.Slice(ck, 2, 0, 100)
	more := rnd(1*2*3*8, 5000)
	gk, ck = g.Concat(gk, g.Cast(g.FromFloats(more, 1, 2, 3, 8), engine.F16), 2), c.Concat(ck, c.FromFloats(more, 1, 2, 3, 8), 2)
	diffOK(t, "cache after truncate and append", 2e-3, g.Floats(gk), c.Floats(ck))
	if gk.(*tensor).grp == nil || gk.(*tensor).grp.base == nil || gk.(*tensor).cacheLen != 103 {
		t.Fatalf("a cache view with its buffer: %+v", gk.(*tensor))
	}
}

// TestTemporariesShareTheBatch: a chain of ops whose intermediates the
// model drops gets a batch the size of what is held, not of every result.
// 200 results of 1 MiB each, only the last one held: the batch is a few
// MiB, where it was 200 before the flush asked the collector (2026-10-26,
// the 27B hybrid's prefill on an L4).
func TestTemporariesShareTheBatch(t *testing.T) {
	g := New()
	defer g.Close()
	x := g.FromFloats(rnd(512*512, 40), 512, 512) // 1 MiB
	y := x
	for i := 0; i < 200; i++ {
		y = g.SiLU(y) // each result a temporary once the next is recorded
	}
	got := g.Floats(y)
	if len(got) != 512*512 {
		t.Fatalf("%d values", len(got))
	}
	t.Logf("batch: %d results, %d held, %.1f MiB on the device", g.lastBatch.results, g.lastBatch.held, float64(g.lastBatch.bytes)/(1<<20))
	if g.lastBatch.held > 8 {
		t.Errorf("%d results held of %d: the dropped temporaries were kept", g.lastBatch.held, g.lastBatch.results)
	}
	if g.lastBatch.bytes > 16<<20 {
		t.Errorf("the batch took %.1f MiB for 1 MiB of held results", float64(g.lastBatch.bytes)/(1<<20))
	}
}

// TestADroppedChainKeepsItsLeaf: the model drops every link of a chain but
// the last, the collector runs before the flush (reuseAbove 0), and the
// leaf at the bottom must still be there when the graph reads it: a node
// inherits a pending input's dependencies. Without that the 27B's load
// freed a weight under the graph and died in ggml_build_forward_expand.
func TestADroppedChainKeepsItsLeaf(t *testing.T) {
	g := New()
	defer g.Close()
	c := cpu.New()
	saved := reuseAbove
	reuseAbove = 0
	defer func() { reuseAbove = saved }()
	data := rnd(64*64, 41)
	chain := func(b engine.Backend) engine.Tensor {
		y := b.SiLU(b.FromFloats(data, 64, 64)) // the leaf is held by nothing but the chain
		for i := 0; i < 5; i++ {
			y = b.SiLU(y)
		}
		return y
	}
	y := chain(g)
	runtime.GC()
	runtime.GC()
	diffOK(t, "dropped chain", 2e-3, g.Floats(y), c.Floats(chain(c)))
	t.Logf("batch: %d results, %d held", g.lastBatch.results, g.lastBatch.held)
}
