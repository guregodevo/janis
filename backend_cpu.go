//go:build !(darwin && arm64 && cgo)

package llm

import (
	"memdoor/llm/cpu"
	"memdoor/llm/engine"
)

// newBackend returns the portable pure-Go CPU backend (Linux, Intel Mac).
func newBackend() engine.Backend { return cpu.New() }

// Accelerated reports whether the active backend is GPU-accelerated. Always
// false here — the CPU backend runs on the cores. Callers use this to size the
// model: large models are fine on a GPU but decode too slowly on CPU.
func Accelerated() bool { return false }

// EngineInfo describes the auto-detected inference path for this host (for
// `llm status`): the pure-Go CPU backend plus whichever SIMD kernel the CPU
// supports (detected at runtime, no env var).
func EngineInfo() string { return "CPU backend, SIMD: " + cpu.SIMDInfo() }
