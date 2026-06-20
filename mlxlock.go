package llm

import "sync"

// mlxComputeMu serializes ALL MLX GPU compute across every consumer in this
// process — the chat Engine and the Embedder. MLX/Metal evaluation is not safe
// under concurrent calls from multiple goroutines: with chat and embeddings now
// both running in-process (they were separate llama-server subprocesses under
// llamafit), an unsynchronized embed firing while a chat generation is in flight
// segfaults inside mlx_eval. A single global lock is the simplest correct fix —
// on a local single-user system the lost parallelism is irrelevant, and the GPU
// was the serial bottleneck anyway.
//
// Hold it around the full synchronous compute (generation / embedding) so the
// blocking Floats/Generate call finishes — and the GPU is idle — before another
// consumer proceeds.
var mlxComputeMu sync.Mutex
