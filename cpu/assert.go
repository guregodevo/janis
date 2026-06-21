package cpu

import "memdoor/llm/engine"

// Compile-time proof the CPU backend satisfies the full engine.Backend.
var _ engine.Backend = (*Backend)(nil)
