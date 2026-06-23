package cpu

// quant4SIMD is set at init by arch-specific runtime CPU-feature detection
// (qmv_<arch>.go): true when this CPU has a 4-bit-decode SIMD kernel it actually
// supports — NEON on arm64 (ASIMD), AVX2 on amd64 (when present). No env var,
// no manual config: the engine detects the host and uses the best path it can,
// falling back to the portable blocked-Go loop otherwise. simdName labels it for
// `llm status` / diagnostics.
var (
	quant4SIMD bool
	simdName   = "none (portable Go)"
)

// SIMDInfo reports the decode kernel selected for this host, e.g.
// "NEON" or "none (portable Go)". Surfaced so an operator can see what a given
// box actually picked without guessing.
func SIMDInfo() string { return simdName }
