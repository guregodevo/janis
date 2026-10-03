package whisper

import (
	"testing"

	"github.com/guregodevo/janis/engine"
)

// newTestBackend returns the backend the whisper receipt tests run against:
// the MLX GPU backend on Apple Silicon, the pure-Go CPU backend elsewhere
// (CI, Linux, Intel). The tests import mlxc nowhere directly — on Linux
// that package is an empty stub, and `undefined: mlxc.New` was red CI for
// two weeks before this helper.
func newTestBackend(t *testing.T) engine.Backend {
	t.Helper()
	return testBackend()
}
