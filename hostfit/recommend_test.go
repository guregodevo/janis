package hostfit

import "testing"

// The 16 GB accelerated tier adopted Qwen3.5-9B over Qwen3-8B (2026-09-01) —
// this pins the tier map so a refactor can't silently regress the default.
func TestRecommendTiers(t *testing.T) {
	cases := []struct {
		ram  int
		acc  bool
		want string
	}{
		{8, true, "Qwen3-4B-Instruct-2507-4bit"},
		{16, true, "Qwen3.5-9B-MLX-4bit"},
		{24, true, "Qwen3.5-9B-MLX-4bit"},
		{32, true, "Qwen3-14B-4bit"},
		{64, true, "Qwen3-32B-4bit"},
		{64, false, "Qwen3-1.7B-4bit"}, // CPU is compute-bound regardless of RAM
	}
	for _, c := range cases {
		if got := (Host{RAMGiB: c.ram, Accelerated: c.acc}).Recommend(); got != c.want {
			t.Errorf("Recommend(ram=%d, acc=%v) = %q, want %q", c.ram, c.acc, got, c.want)
		}
	}
}

// The parameter regex must read "9" out of the adopted model's name — the
// leading "3.5" family version is a decoy ("3.5" is not followed by a B).
func TestParamsBReadsQwen35Name(t *testing.T) {
	if got := ParamsB("Qwen3.5-9B-MLX-4bit"); got != 9 {
		t.Errorf("ParamsB(Qwen3.5-9B-MLX-4bit) = %v, want 9", got)
	}
}
