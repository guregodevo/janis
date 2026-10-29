//go:build !(darwin && arm64 && cgo)

package llm

import (
	"os"

	"github.com/guregodevo/janis/cpu"
	"github.com/guregodevo/janis/engine"
)

// newBackend returns ggml in a build that links it (the Linux release, faster
// on the same cores), else the portable pure-Go CPU backend (Linux, Intel Mac);
// JANIS_BACKEND=cpu forces the pure-Go one.
func newBackend() engine.Backend {
	if ggmlBackend != nil && os.Getenv("JANIS_BACKEND") != "cpu" {
		return ggmlBackend()
	}
	return cpu.New()
}

// Accelerated reports whether the active backend is GPU-accelerated. Always
// false here — the CPU backend runs on the cores. Callers use this to size the
// model: large models are fine on a GPU but decode too slowly on CPU.
func Accelerated() bool { return false }

// EngineInfo describes the auto-detected inference path for this host (for
// `llm status`): the pure-Go CPU backend plus whichever SIMD kernel the CPU
// supports (detected at runtime, no env var).
func EngineInfo() string { return "CPU backend, SIMD: " + cpu.SIMDInfo() }
