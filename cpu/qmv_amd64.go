//go:build amd64

package cpu

import "golang.org/x/sys/cpu"

// Detect AVX2+FMA at runtime and use the AVX2 decode kernel only when the CPU
// actually has them (Haswell 2013+). Older x86 falls back to the portable
// blocked-Go path — no env var, no SIGILL on unsupported hardware.
func init() {
	if cpu.X86.HasAVX2 && cpu.X86.HasFMA {
		quant4SIMD = true
		simdName = "AVX2"
	}
}

// qChannel4Accel computes one 4-bit quantized output channel via the AVX2 kernel
// when available. Guarded by quant4SIMD so it never executes AVX2 on a CPU
// lacking it (the direct unit-test caller relies on this — returns ok=false and
// skips when there's no kernel).
func qChannel4Accel(w *uint32, x, scales, biases *float32, groups, wordsPerGroup int) (float32, bool) {
	if !quant4SIMD {
		return 0, false
	}
	return qChannel4AVX2(w, x, scales, biases, groups, wordsPerGroup), true
}

//go:noescape
func qChannel4AVX2(w *uint32, x, scales, biases *float32, groups, wordsPerGroup int) float32
