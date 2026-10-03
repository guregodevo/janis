package cpu

import "github.com/guregodevo/janis/engine"

// Compile-time proof the CPU backend satisfies the full engine.Backend.
var _ engine.Backend = (*Backend)(nil)
