package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestPercentile(t *testing.T) {
	xs := []float64{5, 1, 4, 2, 3}
	if got := percentile(xs, 50); got != 3 {
		t.Errorf("p50 = %v, want 3", got)
	}
	if got := percentile(xs, 100); got != 5 {
		t.Errorf("p100 must be the max, got %v", got)
	}
	if got := percentile([]float64{7}, 95); got != 7 {
		t.Errorf("a single sample is its own percentile, got %v", got)
	}
	if got := percentile(nil, 95); got != 0 {
		t.Errorf("empty is 0, got %v", got)
	}
	// p95 of 20 samples is the 19th by nearest-rank — not an interpolation
	// that quietly reports a value no request actually saw.
	twenty := make([]float64, 20)
	for i := range twenty {
		twenty[i] = float64(i + 1)
	}
	if got := percentile(twenty, 95); got != 19 {
		t.Errorf("p95 of 1..20 = %v, want 19", got)
	}
}

func TestParseSweep(t *testing.T) {
	got, err := parseSweep("8, 1,4")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 || got[0] != 1 || got[2] != 8 {
		t.Errorf("want sorted [1 4 8], got %v", got)
	}
	for _, bad := range []string{"", "0", "-2", "abc", "2,x"} {
		if _, err := parseSweep(bad); err == nil {
			t.Errorf("parseSweep(%q) should fail", bad)
		}
	}
}

// The shared prefix must be byte-identical across sessions and runs, or the
// server's prefix cache — the effect under test — never hits.
func TestLoremIsDeterministic(t *testing.T) {
	a, b := lorem(500), lorem(500)
	if a != b {
		t.Error("the shared prefix must be identical every call, or prefix caching cannot be measured")
	}
	if lorem(500) == lorem(400) {
		t.Error("different lengths should differ")
	}
	if lorem(0) != "" {
		t.Error("zero tokens is empty")
	}
}

// streamOnce must report the time to the FIRST CONTENT delta — not the time to
// the response header, and not to a role-only opening chunk, which every vLLM
// stream sends before any text.
func TestStreamOnceMeasuresFirstContentDelta(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fl, _ := w.(http.Flusher)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		// role-only chunk first: must NOT count as the first token
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{}}]}\n\n")
		fl.Flush()
		time.Sleep(60 * time.Millisecond)
		for i := 0; i < 3; i++ {
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n")
			fl.Flush()
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
		fl.Flush()
	}))
	defer srv.Close()

	c := &client{base: srv.URL, http: srv.Client()}
	ttft, n, err := c.streamOnce("m", "prompt", 8)
	if err != nil {
		t.Fatal(err)
	}
	if n != 3 {
		t.Errorf("counted %d content deltas, want 3", n)
	}
	if ttft < 50*time.Millisecond {
		t.Errorf("TTFT %v — the role-only chunk was counted as first content", ttft)
	}
}

func TestStreamOnceReportsHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "model not found", http.StatusNotFound)
	}))
	defer srv.Close()
	c := &client{base: srv.URL, http: srv.Client()}
	if _, _, err := c.streamOnce("m", "p", 8); err == nil {
		t.Error("a 404 must surface as an error, not a zero measurement")
	}
}

// A level's errors are counted, not silently dropped — an endpoint that refuses
// at high concurrency is the finding, so it must not read as "0 errors".
func TestRunLevelCountsErrors(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "overloaded", http.StatusTooManyRequests)
	}))
	defer srv.Close()
	c := &client{base: srv.URL, http: srv.Client()}

	got := runLevel(c, "m", "shared", 2, 10, 8, 2)
	if got.Errors != 4 {
		t.Errorf("2 sessions x 2 rounds of failures = 4 errors, got %d", got.Errors)
	}
	if got.P95TTFT != 0 {
		t.Errorf("no successful request means no TTFT, got %v", got.P95TTFT)
	}
}

// The JSON line is the machine-readable receipt; it must carry the fields a
// later comparison needs.
func TestResultJSONShape(t *testing.T) {
	b, err := json.Marshal(Result{Concurrency: 8, P95TTFT: 1.5, TokensPerSec: 42})
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"concurrency", "p95_ttft_s", "tokens_per_sec_total", "errors"} {
		if !contains(string(b), want) {
			t.Errorf("JSON missing %q: %s", want, b)
		}
	}
}

func contains(s, sub string) bool {
	return len(s) >= len(sub) && (func() bool {
		for i := 0; i+len(sub) <= len(s); i++ {
			if s[i:i+len(sub)] == sub {
				return true
			}
		}
		return false
	})()
}
