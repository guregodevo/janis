package llm

import "memdoor/llm/engine"

// pinAll / sweep drive the OPTIONAL engine.Sweeper: the MLX backend needs manual
// array lifecycle (Pin/Sweep), while the pure-Go CPU backend is GC-managed and
// doesn't implement it. These helpers let the engine/embedder/reranker share one
// code path across both — a no-op when the backend has no Sweeper.
func pinAll(b engine.Backend) {
	if s, ok := b.(engine.Sweeper); ok {
		s.PinAll()
	}
}

func sweep(b engine.Backend) {
	if s, ok := b.(engine.Sweeper); ok {
		s.Sweep()
	}
}
