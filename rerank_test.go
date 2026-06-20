package llm

import (
	"os"
	"path/filepath"
	"testing"
)

// TestRerankerSeparation validates the cross-encoder does what the bi-encoder
// couldn't: score a query against a RELEVANT page far above a TANGENTIAL one,
// so corpus gaps (no relevant page) are detectable. Gated MLX_RERANK=1.
//
//	MLX_RERANK=1 go test ./llm/ -run TestRerankerSeparation -v
func TestRerankerSeparation(t *testing.T) {
	if os.Getenv("MLX_RERANK") != "1" {
		t.Skip("set MLX_RERANK=1 to run")
	}
	dir := rerankerDir(t)
	r, err := OpenReranker(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	cy := filepath.Join(os.Getenv("HOME"), ".memdoor/workspaces/hackernews/wikis/cyberlaw")
	read := func(slug string) string {
		b, err := os.ReadFile(filepath.Join(cy, slug))
		if err != nil {
			t.Skipf("page %s missing: %v", slug, err)
		}
		return string(b)
	}
	breach := read("art-33notification-of-a-personal-data-breach-to-the-supervis.md")
	cloud := read("united-states-v-microsoft-corp-the-cloud-act-2018.md")
	auto := read("art-22automated-individual-decision-making-including-profili.md")

	type c struct {
		name, query, page string
		rel               bool // expected relevant
	}
	cases := []c{
		{"breach→breach (REL)", "data breach notify supervisory authority 72 hours", breach, true},
		{"breach→cloud (tang)", "data breach notify supervisory authority 72 hours", cloud, false},
		{"loan→art22 (REL)", "automated loan decision right to explanation profiling", auto, true},
		{"loan→breach (tang)", "automated loan decision right to explanation profiling", breach, false},
		{"litigation→cloud (GAP)", "litigation hold personal phone employee data preservation", cloud, false},
		{"litigation→breach (GAP)", "litigation hold personal phone employee data preservation", breach, false},
	}
	var relMin, tangMax float32 = 1e9, -1e9
	for _, tc := range cases {
		s := r.Score(tc.query, tc.page)
		t.Logf("%-26s score=%+.3f", tc.name, s)
		if tc.rel && s < relMin {
			relMin = s
		}
		if !tc.rel && s > tangMax {
			tangMax = s
		}
	}
	t.Logf("min relevant=%+.3f  max tangential=%+.3f  (separation=%+.3f)", relMin, tangMax, relMin-tangMax)
	if relMin <= tangMax {
		t.Fatalf("cross-encoder did NOT separate: relevant floor %.3f <= tangential ceiling %.3f", relMin, tangMax)
	}
}

func rerankerDir(t *testing.T) string {
	t.Helper()
	if d := os.Getenv("MLX_RERANKER_DIR"); d != "" {
		return d
	}
	m, _ := filepath.Glob(filepath.Join(os.Getenv("HOME"), ".cache/huggingface/hub/models--BAAI--bge-reranker-base/snapshots/*/model.safetensors"))
	if len(m) == 0 {
		t.Skip("bge-reranker-base not downloaded")
	}
	return filepath.Dir(m[0])
}
