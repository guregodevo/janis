//go:build !arm64

package cpu

// hasQuant4SIMD is false off arm64 — QuantMatmul uses the portable blocked-Go
// decode path there (incl. the friend's linux/amd64).
const hasQuant4SIMD = false

// qChannel4Accel has no SIMD kernel off arm64 (the friend's linux/amd64 and
// everything else use the portable blocked-Go path in QuantMatmul). Returning
// ok=false keeps that path the default with zero risk. An amd64 SSE/AVX kernel
// can land here later behind its own build tag.
func qChannel4Accel(w *uint32, x, scales, biases *float32, groups, wordsPerGroup int) (float32, bool) {
	return 0, false
}
