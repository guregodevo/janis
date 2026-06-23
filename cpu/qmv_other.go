//go:build !arm64 && !amd64

package cpu

// No init here, so quant4SIMD stays false off arm64 — QuantMatmul auto-selects
// the portable blocked-Go decode path (incl. the friend's linux/amd64). An
// amd64 AVX2 kernel lands as qmv_amd64.go: an init() gated on cpu.X86.HasAVX2
// flips quant4SIMD on only when the CPU actually supports it (best-fit, no env).

// qChannel4Accel has no SIMD kernel off arm64 yet. Returning ok=false keeps the
// blocked-Go path the default with zero risk.
func qChannel4Accel(w *uint32, x, scales, biases *float32, groups, wordsPerGroup int) (float32, bool) {
	return 0, false
}
