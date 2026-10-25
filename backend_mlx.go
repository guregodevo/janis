//go:build darwin && arm64 && cgo

package llm

import (
	"os"

	"github.com/guregodevo/janis/cpu"
	"github.com/guregodevo/janis/engine"
	"github.com/guregodevo/janis/mlxc"
)

// newBackend returns the Apple-Silicon MLX backend by default. Setting
// JANIS_BACKEND=cpu forces the portable pure-Go backend instead — used to
// validate the Linux/Intel path on a Mac, and available as a fallback. Any
// other value (or unset) keeps MLX. mlxc is imported only here, so non-darwin
// builds never pull in its cgo.
func newBackend() engine.Backend {
	switch os.Getenv("JANIS_BACKEND") {
	case "cpu":
		return cpu.New()
	case "ggml":
		if ggmlBackend != nil {
			return ggmlBackend()
		}
	}
	return mlxc.New()
}

// Accelerated reports whether the active backend is GPU-accelerated (MLX on
// Apple Silicon) — true unless JANIS_BACKEND=cpu forces the pure-Go backend.
// Callers use this to size the model: large models are fine on a GPU but decode
// too slowly on CPU.
func Accelerated() bool { return os.Getenv("JANIS_BACKEND") != "cpu" }

// EngineInfo describes the auto-detected inference path for this host (for
// `llm status`).
func EngineInfo() string {
	switch os.Getenv("JANIS_BACKEND") {
	case "cpu":
		return "CPU backend (forced), SIMD: " + cpu.SIMDInfo()
	case "ggml":
		if ggmlBackend != nil {
			return "ggml backend (forced): " + ggmlBackend().Version()
		}
	}
	return "MLX backend (Metal GPU)"
}
