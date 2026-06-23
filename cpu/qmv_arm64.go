//go:build arm64

package cpu

import "golang.org/x/sys/cpu"

// Detect NEON (Advanced SIMD) at runtime. It's baseline on ARMv8 so this is
// effectively always true, but going through the same detect-and-fall-back path
// as amd64 keeps selection uniform and env-free, and guards the unlikely
// no-ASIMD host instead of running an unsupported instruction.
func init() {
	if cpu.ARM64.HasASIMD {
		quant4SIMD = true
		simdName = "NEON"
	}
}

// qChannel4Accel computes one quantized output channel (4-bit) using the NEON
// kernel: acc = Σ_g scales[g]·(Σ x_i·q_i) + biases[g]·(Σ x_i), over `groups`
// groups of `wordsPerGroup` packed u32 words (8 nibbles each). Returns ok=true
// on arm64; QuantMatmul uses the portable blocked-Go path when ok is false.
//
// w points at the channel's first packed word, x at the input vector start,
// scales/biases at the channel's first group. The kernel walks them in lockstep
// and never escapes (the slices outlive the call).
func qChannel4Accel(w *uint32, x, scales, biases *float32, groups, wordsPerGroup int) (float32, bool) {
	if !quant4SIMD {
		return 0, false
	}
	return qChannel4NEON(w, x, scales, biases, groups, wordsPerGroup), true
}

//go:noescape
func qChannel4NEON(w *uint32, x, scales, biases *float32, groups, wordsPerGroup int) float32
