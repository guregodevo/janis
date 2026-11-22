//go:build ggml

package ggml

import (
	"fmt"
	"math"
	"testing"

	"github.com/guregodevo/janis/cpu"
	"github.com/guregodevo/janis/engine"
)

// TestRecordedStepAttention appends a token to a key and value cache and
// reads them with attention in a recorded step, then runs the step again
// for the next token, against the CPU backend over the whole sequence.
func TestRecordedStepAttention(t *testing.T) {
	replayOnCPU, replayTest = true, true
	defer func() { replayOnCPU, replayTest = false, false }()
	for _, dims := range [][5]int{{4, 2, 8, 30, 64}, {16, 8, 128, 30, 330}, {16, 8, 128, 31, 330}, {16, 8, 128, 40, 330}, {4, 2, 8, 31, 64}} {
		for _, flash := range []bool{false, true} {
			recordedAttention(t, flash, dims[0], dims[1], dims[2], dims[3], dims[4])
		}
	}
}

func recordedAttention(t *testing.T, flash bool, H, Hk, D, P, reserve int) {
	{
		g := New()
		if flash && !g.cpu {
			g.Close()
			return // a GPU backend reads attention without flash (flashOnCPU)
		}
		g.flash = flash
		c := cpu.New()
		total := P + 3
		k, v := rnd(Hk*total*D, 31), rnd(Hk*total*D, 32)
		part := func(x []float32, from, n int) []float32 {
			out := make([]float32, 0, Hk*n*D)
			for h := 0; h < Hk; h++ {
				out = append(out, x[(h*total+from)*D:(h*total+from+n)*D]...)
			}
			return out
		}
		prefix := func(x []float32, n int) []float32 { return part(x, 0, n) }
		g.ReserveCache(reserve)
		gk := g.Cast(g.FromFloats(part(k, 0, P), 1, Hk, P, D), engine.F16)
		gv := g.Cast(g.FromFloats(part(v, 0, P), 1, Hk, P, D), engine.F16)
		gk = g.Concat(gk, g.Cast(g.FromFloats(part(k, P, 1), 1, Hk, 1, D), engine.F16), 2)
		gv = g.AppendValues(gv, g.Cast(g.FromFloats(part(v, P, 1), 1, Hk, 1, D), engine.F16))
		g.Floats(gk)
		scale := float32(1.0 / math.Sqrt(float64(D)))
		name := fmt.Sprintf("%s, %d heads of %d", map[bool]string{false: "transposed V", true: "flash"}[flash], H, D)
		want := func(n int, q []float32) []float32 {
			return c.Floats(c.SDPA(c.FromFloats(q, 1, H, 1, D),
				c.FromFloats(prefix(k, n), 1, Hk, n, D), c.FromFloats(prefix(v, n), 1, Hk, n, D), scale, false))
		}

		at := P + 1
		if g.BeginStep(at, nil) == nil {
			t.Fatal("no recording")
		}
		kl, vl := as(g.FromFloats(part(k, at, 1), 1, Hk, 1, D)), as(g.FromFloats(part(v, at, 1), 1, Hk, 1, D))
		ql := as(g.FromFloats(rnd(H*D, 33), 1, H, 1, D))
		k2 := g.Concat(gk, g.Cast(kl, engine.F16), 2)
		v2 := g.AppendValues(gv, g.Cast(vl, engine.F16))
		out := g.Reshape(g.SDPA(ql, k2, v2, scale, false), 1, 1, H*D)
		st, got := g.EndStep(g.FromInt32([]int32{0}, 1), out, nil)
		if st == nil {
			t.Fatalf("%s: the step was not recorded", name)
		}
		diffOK(t, name+": recorded step", 5e-3, got, want(at+1, rnd(H*D, 33)))

		at++
		setFloat32(kl, part(k, at, 1))
		setFloat32(vl, part(v, at, 1))
		q := rnd(H*D, 34)
		setFloat32(ql, q)
		diffOK(t, name+": step run again", 5e-3, st.Run(0, at), want(at+1, q))
		st.Close()
		g.Close()
	}
}

// TestRecordedStepCarryAndRoPE runs a recorded step that reads a carried
// state, replaces it (state' = state*g + x) and rotates a vector at the
// step's position, three times, against the same computed afresh.
func TestRecordedStepCarryAndRoPE(t *testing.T) {
	replayOnCPU, replayTest = true, true
	defer func() { replayOnCPU, replayTest = false, false }()
	g := New()
	c := cpu.New()
	const N, H, D = 16, 2, 8
	state := rnd(N, 41)
	st0 := g.FromFloats(state, 1, N)
	g.Floats(st0)
	moved := g.BeginStep(5, []engine.Tensor{st0})
	if moved == nil {
		t.Fatal("no recording")
	}
	xl := as(g.FromFloats(rnd(N, 42), 1, N))
	ql := as(g.FromFloats(rnd(H*D, 43), 1, H, 1, D))
	next := g.Add(g.ScalarMul(moved[0], 0.5), xl)
	rot := g.Reshape(g.RoPE(ql, D, false, 10000, 1, 5), 1, H*D)
	out := g.Concat(next, rot, 1)
	step, got := g.EndStep(g.FromInt32([]int32{0}, 1), out, []engine.Tensor{next})
	if step == nil {
		t.Fatal("the step was not recorded")
	}
	cur := state
	for run := 0; run < 3; run++ {
		at := 5 + run
		x, q := rnd(N, int64(50+run)), rnd(H*D, int64(60+run))
		if run > 0 {
			setFloat32(xl, x)
			setFloat32(ql, q)
			got = step.Run(0, at)
		} else {
			x, q = rnd(N, 42), rnd(H*D, 43)
		}
		cs := c.Add(c.ScalarMul(c.FromFloats(cur, 1, N), 0.5), c.FromFloats(x, 1, N))
		cr := c.Reshape(c.RoPE(c.FromFloats(q, 1, H, 1, D), D, false, 10000, 1, at), 1, H*D)
		want := c.Floats(c.Concat(cs, cr, 1))
		diffOK(t, "run "+string(rune('0'+run)), 1e-4, got, want)
		cur = c.Floats(cs)
		diffOK(t, "carried state after run "+string(rune('0'+run)), 1e-4, g.Floats(moved[0]), cur)
	}
	step.Close()
	g.Close()
}
