//go:build !(darwin && arm64 && cgo)

package whisper

import (
	"memdoor/llm/cpu"
	"memdoor/llm/engine"
)

func testBackend() engine.Backend { return cpu.New() }
