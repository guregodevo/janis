package llm

import (
	"os"
	"path/filepath"
	"sort"
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

// TestRerankerSchoolPrecision is the head-to-head precision probe: rank a fixed
// candidate set (relevant school/student-data laws mixed with tangential privacy
// pages) against a school-data query and report whether the relevant pages float
// to the top. Run once per reranker via MLX_RERANKER_DIR to compare base vs
// v2-m3 on the known Q7/FERPA mis-ranking — without tuning to it.
//
//	MLX_RERANK=1 MLX_RERANKER_DIR=<dir> go test ./llm/ -run TestRerankerSchoolPrecision -v
func TestRerankerSchoolPrecision(t *testing.T) {
	if os.Getenv("MLX_RERANK") != "1" {
		t.Skip("set MLX_RERANK=1 to run")
	}
	dir := rerankerDir(t)
	r, err := OpenReranker(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	t.Logf("reranker dir: %s", dir)

	cy := filepath.Join(os.Getenv("HOME"), ".memdoor/workspaces/hackernews/wikis/cyberlaw")
	read := func(slug string) string {
		b, err := os.ReadFile(filepath.Join(cy, slug+".md"))
		if err != nil {
			t.Skipf("page %s missing: %v", slug, err)
		}
		return string(b)
	}

	query := "school student education records data privacy law"
	// rel=true: these directly govern student/school data. rel=false: tangential
	// privacy laws that the bi-encoder (and base reranker) wrongly floated up.
	type cand struct {
		slug string
		rel  bool
	}
	cands := []cand{
		{"ferpa-20-usc-1232g", true},
		{"ferpa-regulations-34-cfr-99", true},
		{"sopipa-cal-bus-prof-code-ss-22584", true},
		{"coppa-15-usc-6501", true},
		{"art-8conditions-applicable-to-child-s-consent-in-relation-to", true},
		{"california-consumer-privacy-act-ccpa-cpra", false},
		{"right-to-privacy", false},
		{"illinois-biometric-information-privacy-act-bipa-740-ilcs-14", false},
		{"electronic-communications-privacy-act-ecpa-1986", false},
		{"australia-privacy-act-1988", false},
	}

	type scored struct {
		slug  string
		rel   bool
		score float32
	}
	var rows []scored
	for _, c := range cands {
		rows = append(rows, scored{c.slug, c.rel, r.Score(query, read(c.slug))})
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].score > rows[j].score })

	// Precision@K: how many of the top-K (K = number of relevant pages) are
	// actually relevant. A perfect reranker scores 1.0.
	k := 0
	for _, c := range cands {
		if c.rel {
			k++
		}
	}
	hits := 0
	t.Logf("query: %q", query)
	for i, row := range rows {
		mark := "    "
		if row.rel {
			mark = "REL "
		}
		top := ""
		if i < k {
			top = "  <-- top-K"
			if row.rel {
				hits++
			}
		}
		t.Logf("  %2d. %s%+.3f  %s%s", i+1, mark, row.score, row.slug, top)
	}
	t.Logf("precision@%d = %d/%d = %.2f", k, hits, k, float64(hits)/float64(k))
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
