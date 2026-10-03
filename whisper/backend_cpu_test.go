//go:build !(darwin && arm64 && cgo)

package whisper

import (
	"github.com/guregodevo/janis/cpu"
	"github.com/guregodevo/janis/engine"
)

func testBackend() engine.Backend { return cpu.New() }
