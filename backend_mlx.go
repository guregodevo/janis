//go:build darwin && arm64

package llm

import (
	"os"

	"memdoor/llm/cpu"
	"memdoor/llm/engine"
	"memdoor/llm/mlxc"
)

// newBackend returns the Apple-Silicon MLX backend by default. Setting
// MEMDOOR_BACKEND=cpu forces the portable pure-Go backend instead — used to
// validate the Linux/Intel path on a Mac, and available as a fallback. Any
// other value (or unset) keeps MLX. mlxc is imported only here, so non-darwin
// builds never pull in its cgo.
func newBackend() engine.Backend {
	if os.Getenv("MEMDOOR_BACKEND") == "cpu" {
		return cpu.New()
	}
	return mlxc.New()
}
