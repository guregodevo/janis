package llm

import "github.com/guregodevo/janis/engine"

// ggmlBackend is set by backend_ggml.go in a build with the ggml tag:
// JANIS_BACKEND=ggml then runs the engine on ggml (CPU, or the device
// JANIS_GGML_DEVICE names). Nil in every other build.
var ggmlBackend func() engine.Backend
