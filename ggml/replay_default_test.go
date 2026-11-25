//go:build ggml

package ggml

import "testing"

// A CUDA BACKEND REPLAYS DECODE STEPS BY DEFAULT.
//
// Recorded steps and padded plain reads are on for CUDA unless
// JANIS_GGML_REPLAY=0; Metal and the CPU backend keep the plain steps
// unless asked. JANIS_GGML_PADDECODE overrides the reads either way.
func TestACUDABackendReplaysByDefault(t *testing.T) {
	cuda, metal, cpu := &Backend{name: "CUDA0"}, &Backend{name: "MTL0"}, &Backend{name: "CPU", cpu: true}
	for _, c := range []struct {
		replay, pad string
		b           *Backend
		on, padded  bool
	}{
		{"", "", cuda, true, true},
		{"0", "", cuda, false, false},
		{"0", "1", cuda, false, true},
		{"", "0", cuda, true, false},
		{"", "", metal, false, false},
		{"1", "", metal, true, true},
		{"", "", cpu, false, false},
		{"1", "", cpu, false, false},
		{"cpu", "", cpu, true, true},
	} {
		t.Setenv("JANIS_GGML_REPLAY", c.replay)
		t.Setenv("JANIS_GGML_PADDECODE", c.pad)
		if got := c.b.replayOn(); got != c.on {
			t.Errorf("%s, REPLAY=%q: records %v, want %v", c.b.name, c.replay, got, c.on)
		}
		if got := c.b.padDecode(); got != c.padded {
			t.Errorf("%s, REPLAY=%q PADDECODE=%q: padded reads %v, want %v", c.b.name, c.replay, c.pad, got, c.padded)
		}
	}
}
