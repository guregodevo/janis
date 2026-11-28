package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/guregodevo/janis/evals"
)

// weightsServer stands in for janis serve --edit: two weights, edits kept
// in memory, the path of the last save or load recorded.
type weightsServer struct {
	mu     sync.Mutex
	values map[string][]float32
	path   string
}

func newWeightsServer(t *testing.T) (*weightsServer, string) {
	ws := &weightsServer{values: map[string][]float32{
		"layers.0.linear_attn.dt_bias": {1, 2},
		"layers.0.mlp.down_proj":       {5, 6},
	}}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/weights", func(w http.ResponseWriter, r *http.Request) {
		var data []map[string]string
		for _, n := range []string{"layers.0.linear_attn.dt_bias", "layers.0.mlp.down_proj"} {
			data = append(data, map[string]string{"name": n})
		}
		json.NewEncoder(w).Encode(map[string]any{"data": data, "edits": []string{}})
	})
	mux.HandleFunc("GET /v1/weights/{name}", func(w http.ResponseWriter, r *http.Request) {
		ws.mu.Lock()
		defer ws.mu.Unlock()
		json.NewEncoder(w).Encode(map[string]any{"values": ws.values[r.PathValue("name")]})
	})
	mux.HandleFunc("POST /v1/weights/{name}", func(w http.ResponseWriter, r *http.Request) {
		var req struct{ Values []float32 }
		json.NewDecoder(r.Body).Decode(&req)
		ws.mu.Lock()
		ws.values[r.PathValue("name")] = req.Values
		ws.mu.Unlock()
		json.NewEncoder(w).Encode(map[string]any{"edits": []string{r.PathValue("name")}})
	})
	for _, op := range []string{"save", "load"} {
		mux.HandleFunc("POST /v1/weights/"+op, func(w http.ResponseWriter, r *http.Request) {
			var req struct{ Path string }
			json.NewDecoder(r.Body).Decode(&req)
			ws.mu.Lock()
			ws.path = req.Path
			ws.mu.Unlock()
			json.NewEncoder(w).Encode(map[string]any{"edits": []string{"a"}})
		})
	}
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return ws, srv.URL
}

func TestWeightsShiftEditsOnlyTheMatches(t *testing.T) {
	ws, u := newWeightsServer(t)
	var out bytes.Buffer
	if err := weightsCmd([]string{"shift", "--url", u, "--match", `dt_bias$`, "--by", "0.5"}, &out); err != nil {
		t.Fatal(err)
	}
	if out.String() != "shifted 1 weights by +0.5\n" {
		t.Fatalf("printed %q", out.String())
	}
	if v := ws.values["layers.0.linear_attn.dt_bias"]; v[0] != 1.5 || v[1] != 2.5 {
		t.Fatalf("dt_bias %v", v)
	}
	if v := ws.values["layers.0.mlp.down_proj"]; v[0] != 5 {
		t.Fatalf("down_proj changed: %v", v)
	}
	if err := weightsCmd([]string{"shift", "--url", u, "--match", `nothing`, "--by", "1"}, &out); err == nil {
		t.Fatal("a match of no weight was not refused")
	}
}

func TestWeightsSaveSendsAnAbsolutePath(t *testing.T) {
	ws, u := newWeightsServer(t)
	var out bytes.Buffer
	if err := weightsCmd([]string{"save", "--url", u, "edits.safetensors"}, &out); err != nil {
		t.Fatal(err)
	}
	if !filepath.IsAbs(ws.path) || filepath.Base(ws.path) != "edits.safetensors" {
		t.Fatalf("sent %q", ws.path)
	}
}

func writeScore(t *testing.T, dir, name string, correct ...bool) string {
	s := evals.Score{Problems: "set", Seed: 1, MaxTokens: 10, N: len(correct)}
	for i, c := range correct {
		s.Results = append(s.Results, evals.Result{Problem: i, Seed: 1 + i, Correct: c})
	}
	p := filepath.Join(dir, name)
	if err := writeJSON(p, s); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestCompareEndsWithTheVerdict(t *testing.T) {
	dir := t.TempDir()
	none := make([]bool, 10)
	all := []bool{true, true, true, true, true, true, true, true, true, true}
	base := writeScore(t, dir, "base.json", none...)
	better := writeScore(t, dir, "better.json", all...)
	for _, c := range []struct {
		a, b, last string
	}{{base, better, "PASS"}, {better, base, "FAIL"}, {base, base, "FAIL"}} {
		var out bytes.Buffer
		if err := evalCmd([]string{"compare", c.a, c.b}, &out); err != nil {
			t.Fatal(err)
		}
		lines := strings.Split(strings.TrimSpace(out.String()), "\n")
		if lines[len(lines)-1] != c.last {
			t.Fatalf("%s vs %s:\n%s", c.a, c.b, out.String())
		}
	}
	other := filepath.Join(dir, "other.json")
	writeJSON(other, evals.Score{Problems: "different"})
	if err := evalCmd([]string{"compare", base, other}, &bytes.Buffer{}); err == nil {
		t.Fatal("scores of different problems were compared")
	}
}

func TestEvalWritesTheScore(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			http.NotFound(w, r)
			return
		}
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"content": `\boxed{4}`}, "finish_reason": "stop"}},
			"usage":   map[string]any{"completion_tokens": 7},
		})
	}))
	defer srv.Close()
	dir := t.TempDir()
	problems := filepath.Join(dir, "p.jsonl")
	os.WriteFile(problems, []byte(`{"question":"2+2?","answer":4}`+"\n"+`{"question":"2+3?","answer":5}`+"\n"), 0o644)
	dest := filepath.Join(dir, "score.json")
	var out bytes.Buffer
	if err := evalCmd([]string{"--url", srv.URL, "--problems", problems, "--out", dest}, &out); err != nil {
		t.Fatal(err)
	}
	if out.String() != "correct 1/2  at cap 0/2  mean tokens 7\n" {
		t.Fatalf("printed %q", out.String())
	}
	var s evals.Score
	if err := readJSON(dest, &s); err != nil || s.N != 2 || s.Seed != 1000 || s.MaxTokens != 16384 {
		t.Fatalf("score %+v %v", s, err)
	}
	if err := evalCmd([]string{"--problems", problems}, &out); err == nil {
		t.Fatal("no --out was not refused")
	}
}
