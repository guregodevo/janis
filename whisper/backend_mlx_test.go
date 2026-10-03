//go:build darwin && arm64 && cgo

package whisper

import (
	"github.com/guregodevo/janis/engine"
	"github.com/guregodevo/janis/mlxc"
)

func testBackend() engine.Backend { return mlxc.New() }
