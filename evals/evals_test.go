package evals

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

func TestLastBoxed(t *testing.T) {
	for in, want := range map[string]string{
		`so \boxed{42}.`:                    "42",
		`\boxed{1} then \boxed{ -7 }`:       "-7",
		`\boxed{\frac{1}{2}}`:               `\frac{1}{2}`,
		`no box`:                            "",
		`\boxed{12`:                         "",
		`\boxed{\$1,234} is the profit`:     `\$1,234`,
		`<think>\boxed{3}</think>\boxed{4}`: "4",
	} {
		if got := LastBoxed(in); got != want {
			t.Errorf("LastBoxed(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestSame(t *testing.T) {
	for _, c := range []struct {
		got, want string
		same      bool
	}{
		{`\$1,234`, "1234", true},
		{"−5", "-5", true},
		{"2.50", "2.5", true},
		{`\frac{1}{2}`, `\frac{1}{2}`, true},
		{"41", "42", false},
		{"", "0", false},
	} {
		if Same(c.got, c.want) != c.same {
			t.Errorf("Same(%q, %q) = %v", c.got, c.want, !c.same)
		}
	}
}

func TestReadProblems(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "p.jsonl")
	os.WriteFile(path, []byte(`{"question":"1+1?","answer":2}
`+"\n"+`{"question":"name?","answer":"x"}`), 0o644)
	ps, sum, err := ReadProblems(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(ps) != 2 || ps[0].Answer != "2" || ps[1].Answer != "x" || len(sum) != 64 {
		t.Fatalf("got %+v %q", ps, sum)
	}
	os.WriteFile(path, []byte(`{"question":"no answer"}`), 0o644)
	if _, _, err := ReadProblems(path); err == nil || !strings.Contains(err.Error(), ":1:") {
		t.Fatalf("a problem with no answer: %v", err)
	}
}

// fakeServer answers problem "q<n>" with \boxed{n} when n is even and
// runs out of tokens on odd ones; it records each request's seed.
func fakeServer(t *testing.T, edits []string) (*httptest.Server, *sync.Map) {
	seeds := &sync.Map{}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/weights", func(w http.ResponseWriter, r *http.Request) {
		if edits == nil {
			http.NotFound(w, r)
			return
		}
		json.NewEncoder(w).Encode(map[string]any{"edits": edits})
	})
	mux.HandleFunc("POST /v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Messages []struct{ Content string }
			Seed     int
			Effort   string `json:"reasoning_effort"`
		}
		json.NewDecoder(r.Body).Decode(&req)
		q := req.Messages[0].Content
		seeds.Store(q, req.Seed)
		var n int
		fmt.Sscanf(q, "q%d", &n)
		content, finish := fmt.Sprintf(`thinking... \boxed{%d}`, n), "stop"
		if n%2 == 1 {
			content, finish = "thinking and thinking", "length"
		}
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"content": content}, "finish_reason": finish}},
			"usage":   map[string]any{"completion_tokens": 100 + n},
		})
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv, seeds
}

func TestRunScoresEachProblemOnItsSeed(t *testing.T) {
	srv, seeds := fakeServer(t, []string{"a.dt_bias"})
	var ps []Problem
	for i := 0; i < 6; i++ {
		ps = append(ps, Problem{Question: fmt.Sprintf("q%d", i), Answer: fmt.Sprint(i)})
	}
	s, err := Run(context.Background(), Config{URL: srv.URL, Seed: 5000, MaxTokens: 64, Parallel: 3}, ps, "set")
	if err != nil {
		t.Fatal(err)
	}
	if s.N != 6 || s.Correct != 3 || s.AtCap != 3 || s.MeanTokens != 102.5 || s.Problems != "set" {
		t.Fatalf("score %+v", s)
	}
	if len(s.Edits) != 1 || s.Edits[0] != "a.dt_bias" {
		t.Fatalf("edits %v", s.Edits)
	}
	for i, r := range s.Results {
		if r.Problem != i || r.Seed != 5000+i {
			t.Fatalf("result %d: %+v", i, r)
		}
		if got, _ := seeds.Load(fmt.Sprintf("q%d", i)); got != 5000+i {
			t.Fatalf("problem %d sent seed %v", i, got)
		}
	}
	if s.Summary() != "correct 3/6  at cap 3/6  mean tokens 102" {
		t.Fatalf("summary %q", s.Summary())
	}
}

func TestRunWithoutEditSaysNoEdits(t *testing.T) {
	srv, _ := fakeServer(t, nil)
	s, err := Run(context.Background(), Config{URL: srv.URL}, []Problem{{Question: "q0", Answer: "0"}}, "set")
	if err != nil {
		t.Fatal(err)
	}
	if s.Edits == nil || len(s.Edits) != 0 {
		t.Fatalf("edits %v", s.Edits)
	}
}

func TestRunStopsOnAFailedRequest(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			http.NotFound(w, r)
			return
		}
		http.Error(w, "model gone", http.StatusInternalServerError)
	}))
	defer srv.Close()
	_, err := Run(context.Background(), Config{URL: srv.URL, Parallel: 2}, []Problem{{"q0", "0"}, {"q1", "1"}, {"q2", "2"}}, "set")
	if err == nil || !strings.Contains(err.Error(), "model gone") {
		t.Fatalf("err %v", err)
	}
}

func score(correct ...bool) Score {
	s := Score{Problems: "set", Seed: 1, MaxTokens: 10}
	for i, c := range correct {
		s.Results = append(s.Results, Result{Problem: i, Seed: 1 + i, Correct: c})
	}
	return s
}

func TestCompare(t *testing.T) {
	base := score(false, false, true, true, false)
	edited := score(true, true, true, false, false)
	p, err := Compare(base, edited)
	if err != nil {
		t.Fatal(err)
	}
	if p.N != 5 || p.Base != 2 || p.Edited != 3 || p.Gained != 2 || p.Lost != 1 {
		t.Fatalf("%+v", p)
	}
	// 3 changed, 2 or more gains by luck: (3 + 1) / 8.
	if math.Abs(p.P-0.5) > 1e-12 {
		t.Fatalf("p %v", p.P)
	}
	if p.Pass(0.05) {
		t.Fatal("2 gained 1 lost passed")
	}
}

func TestSignTest(t *testing.T) {
	for _, c := range []struct {
		g, k int
		p    float64
	}{
		{0, 0, 1},
		{5, 5, 1.0 / 32},
		{0, 5, 1},
		{9, 10, 11.0 / 1024},
	} {
		if got := signTest(c.g, c.k); math.Abs(got-c.p) > 1e-12 {
			t.Errorf("signTest(%d, %d) = %v, want %v", c.g, c.k, got, c.p)
		}
	}
	p := Paired{Gained: 9, Lost: 1, P: signTest(9, 10)}
	if !p.Pass(0.05) {
		t.Fatal("9 gained 1 lost at p 0.011 failed")
	}
}

func TestCompareRefusesUnpairedScores(t *testing.T) {
	base := score(true, false)
	for name, change := range map[string]func(*Score){
		"problems": func(s *Score) { s.Problems = "other" },
		"seeds":    func(s *Score) { s.Seed = 2 },
		"caps":     func(s *Score) { s.MaxTokens = 20 },
		"models":   func(s *Score) { s.Model = "m" },
		"results":  func(s *Score) { s.Results = s.Results[:1] },
	} {
		e := score(true, false)
		change(&e)
		if _, err := Compare(base, e); err == nil || !strings.Contains(err.Error(), name) {
			t.Errorf("%s: %v", name, err)
		}
	}
}
