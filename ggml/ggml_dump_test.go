//go:build ggml

package ggml

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/guregodevo/janis/cpu"
	"github.com/guregodevo/janis/engine"
)

// TestDumpedQuantMatmul replays a QuantMatmul the check dumped under
// /tmp/janis-nan (a real weight and the real input that gave NaN) on this
// backend (CPU, or the device JANIS_GGML_DEVICE names) and on the pure-Go
// oracle, and says which rows are bad where. Skipped without a dump.
func TestDumpedQuantMatmul(t *testing.T) {
	dir := "/tmp/janis-nan"
	mb, err := os.ReadFile(dir + "/meta.json")
	if err != nil {
		t.Skip("no dump")
	}
	var meta struct {
		X, W, Scales []int
		GroupSize    int `json:"group_size"`
		Bits         int `json:"bits"`
	}
	if err := json.Unmarshal(mb, &meta); err != nil {
		t.Fatal(err)
	}
	readF := func(name string, n int) []float32 {
		b, err := os.ReadFile(dir + "/" + name)
		if err != nil {
			t.Fatal(err)
		}
		out := make([]float32, n)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
		}
		return out
	}
	n := func(s []int) int {
		p := 1
		for _, d := range s {
			p *= d
		}
		return p
	}
	x := readF("x.f32", n(meta.X))
	wb, _ := os.ReadFile(dir + "/w.u32")
	sc := readF("scales.f32", n(meta.Scales))
	bi := readF("biases.f32", n(meta.Scales))

	g := New()
	defer g.Close()
	c := cpu.New()
	t.Log(g.Version(), "x", meta.X, "w", meta.W, "scales", meta.Scales)
	got := g.Floats(g.QuantMatmul(g.FromFloats(x, meta.X...), g.FromRaw(engine.U32, wb, meta.W...), g.FromFloats(sc, meta.Scales...), g.FromFloats(bi, meta.Scales...), true, meta.GroupSize, meta.Bits))
	want := c.Floats(c.QuantMatmul(c.FromFloats(x, meta.X...), c.FromRaw(engine.U32, wb, meta.W...), c.FromFloats(sc, meta.Scales...), c.FromFloats(bi, meta.Scales...), true, meta.GroupSize, meta.Bits))
	out := meta.W[0]
	var badG, badC []int
	worst := 0.0
	for i := range got {
		if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
			badG = append(badG, i%out)
		}
		if math.IsNaN(float64(want[i])) || math.IsInf(float64(want[i]), 0) {
			badC = append(badC, i%out)
		}
		if d := math.Abs(float64(got[i] - want[i])); d > worst && !math.IsNaN(d) {
			worst = d
		}
	}
	t.Logf("bad rows on %s: %v; bad rows on the pure-Go oracle: %v; worst finite |diff| %.4g", g.Version(), badG, badC, worst)
	if len(badG) > 0 {
		// The rows' parameters, for the record.
		G := meta.Scales[len(meta.Scales)-1]
		for _, r := range badG[:min(3, len(badG))] {
			mn, mx := math.Inf(1), math.Inf(-1)
			bmn, bmx := math.Inf(1), math.Inf(-1)
			for gi := 0; gi < G; gi++ {
				v, b := float64(sc[r*G+gi]), float64(bi[r*G+gi])
				mn, mx, bmn, bmx = math.Min(mn, v), math.Max(mx, v), math.Min(bmn, b), math.Max(bmx, b)
			}
			t.Logf("row %d: scales [%.3g, %.3g], biases [%.3g, %.3g]", r, mn, mx, bmn, bmx)
		}
		t.Fail()
	}
}
