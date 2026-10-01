//go:build darwin && arm64 && cgo

package whisper

import (
	"memdoor/llm/engine"
	"memdoor/llm/mlxc"
)

func testBackend() engine.Backend { return mlxc.New() }
