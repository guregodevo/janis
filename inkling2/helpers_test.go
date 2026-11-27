package inkling2

import (
	"encoding/binary"
	"math"
	"path/filepath"
	"testing"

	"github.com/guregodevo/janis/engine"
	"github.com/guregodevo/janis/safetensors"
)

// loadTiny loads one of the models testdata/gen_fixture.py made with the
// reference implementation on b, and what the reference computed on it.
func loadTiny(t *testing.T, b engine.Backend, name string, stream bool) (*Model, *safetensors.File) {
	t.Helper()
	dir := filepath.Join("testdata", "tiny-"+name)
	cfg, err := LoadConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	cfg.StreamExperts = stream
	st, err := safetensors.OpenModel(dir)
	if err != nil {
		t.Fatal(err)
	}
	m, err := LoadModel(b, st, cfg)
	if err != nil {
		t.Fatal(err)
	}
	want, err := safetensors.Open(filepath.Join(dir, "expected.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	return m, want
}

func wantFloats(t *testing.T, st *safetensors.File, name string) []float32 {
	t.Helper()
	_, _, raw, err := st.Get(name)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}

func wantTokens(t *testing.T, st *safetensors.File) []int32 {
	t.Helper()
	_, _, raw, err := st.Get("tokens")
	if err != nil {
		t.Fatal(err)
	}
	out := make([]int32, len(raw)/4)
	for i := range out {
		out[i] = int32(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}

// maxDiff is the largest absolute difference, and the largest value of want.
func maxDiff(got, want []float32) (diff, scale float64) {
	for i := range want {
		diff = math.Max(diff, math.Abs(float64(got[i]-want[i])))
		scale = math.Max(scale, math.Abs(float64(want[i])))
	}
	return diff, scale
}
