//go:build ggml

package llm

import (
	"github.com/guregodevo/janis/engine"
	"github.com/guregodevo/janis/ggml"
)

func init() { ggmlBackend = func() engine.Backend { return ggml.New() } }
