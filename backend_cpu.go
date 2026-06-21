//go:build !(darwin && arm64)

package llm

import (
	"memdoor/llm/cpu"
	"memdoor/llm/engine"
)

// newBackend returns the portable pure-Go CPU backend (Linux, Intel Mac).
func newBackend() engine.Backend { return cpu.New() }
