//go:build darwin && arm64

package llm

import (
	"memdoor/llm/engine"
	"memdoor/llm/mlxc"
)

// newBackend returns the Apple-Silicon MLX backend (the only file that imports
// mlxc, so non-darwin builds never pull in its cgo).
func newBackend() engine.Backend { return mlxc.New() }
