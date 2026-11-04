package hostfit

import "testing"

// The 16 GB tier adopted Qwen3.5-9B over Qwen3-8B (2026-09-01) —
// this pins the tier map so a refactor can't silently regress the default.
func TestRecommendTiers(t *testing.T) {
	cases := []struct {
		ram  int
		want string
	}{
		{8, "Qwen3-4B-Instruct-2507-4bit"},
		{16, "Qwen3.5-9B-MLX-4bit"},
		{24, "Qwen3.5-9B-MLX-4bit"},
		{32, "Qwen3.8-27B-4bit"}, // 15 GiB of weights, 2 GiB of KV at 32K: the 27B fits 32 GB
		{64, "Qwen3.8-27B-4bit"},
	}
	for _, c := range cases {
		if got := (Host{RAMGiB: c.ram}).Recommend(); got != c.want {
			t.Errorf("Recommend(ram=%d) = %q, want %q", c.ram, got, c.want)
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

// MemTotal rounds up to the installed RAM: a 16 GB VM and a 32 GB box report
// less than that, and must still land in their tier.
func TestMemTotalGiBRoundsUp(t *testing.T) {
	for kb, want := range map[int64]int{16307812: 16, 32820000: 32, 16 * 1024 * 1024: 16} {
		if got := memTotalGiB(kb); got != want {
			t.Errorf("memTotalGiB(%d) = %d, want %d", kb, got, want)
		}
	}
}
