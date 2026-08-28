// Command tenancybench measures how many developers one rented GPU actually
// serves — the number the pricing offer is staked on and nobody has measured.
//
// Why it exists: with idle pause arithmetically excluded at every price we sell
// and the batch/CI market retracted, multi-tenancy is the only lever that makes
// a GPU-hour cheaper than tokens for a workload that idles ~98% of the time
// (docs/internal/CAPACITY_MODEL.md). `pool.DefaultPerMember = 40` is an M/M/1
// guess its own comment calls "the single most commercially important open
// number". This measures it.
//
// It drives an OpenAI-compatible endpoint (vLLM on a rented GPU, or any local
// server) with N concurrent synthetic coding sessions that SHARE a long prefix —
// the way a real team does, all on one repository — and sweeps N upward,
// reporting where p95 time-to-first-token crosses the 2s bar.
//
// Usage:
//
//	tenancybench -endpoint https://host:port/v1 [flags]
//	  -endpoint  OpenAI-compatible base URL (required)
//	  -token     bearer token for that endpoint
//	  -model     model name to request (default: whatever /v1/models reports first)
//	  -sweep     concurrency levels to try (default 1,2,4,8,16,32)
//	  -shared    tokens of shared team prefix per request (default 2000)
//	  -private   tokens of per-session private context (default 400)
//	  -gen       tokens to generate per request (default 64)
//	  -rounds    requests per session per level (default 3)
//	  -ttft-bar  the SLO in seconds (default 2.0)
//	  -json      emit one JSON line per level instead of a table
//
// It changes nothing and rents nothing: point it at an endpoint you already
// have. Prefix caching is the effect under test, so every session sends the
// SAME shared prefix — a server with prefix caching on will show flat TTFT far
// past the point an uncached one collapses.
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

func main() {
	var (
		endpoint = flag.String("endpoint", "", "OpenAI-compatible base URL (required)")
		token    = flag.String("token", "", "bearer token")
		model    = flag.String("model", "", "model name (default: first from /v1/models)")
		sweepArg = flag.String("sweep", "1,2,4,8,16,32", "concurrency levels")
		shared   = flag.Int("shared", 2000, "tokens of shared team prefix")
		private  = flag.Int("private", 400, "tokens of private per-session context")
		gen      = flag.Int("gen", 64, "tokens to generate per request")
		rounds   = flag.Int("rounds", 3, "requests per session per level")
		ttftBar  = flag.Float64("ttft-bar", 2.0, "p95 TTFT SLO in seconds")
		asJSON   = flag.Bool("json", false, "emit JSON lines")
	)
	flag.Parse()

	if *endpoint == "" {
		fmt.Fprintln(os.Stderr, "tenancybench: -endpoint is required")
		flag.Usage()
		os.Exit(2)
	}
	levels, err := parseSweep(*sweepArg)
	if err != nil {
		fmt.Fprintln(os.Stderr, "tenancybench:", err)
		os.Exit(2)
	}

	c := &client{base: strings.TrimRight(*endpoint, "/"), token: *token,
		http: &http.Client{Timeout: 10 * time.Minute}}

	name := *model
	if name == "" {
		if name, err = c.firstModel(); err != nil {
			fmt.Fprintln(os.Stderr, "tenancybench: could not read /v1/models:", err)
			os.Exit(1)
		}
	}

	prefix := lorem(*shared) // identical for every session — the shared team context

	if !*asJSON {
		fmt.Printf("model %s · shared prefix %d tok · private %d tok · gen %d tok · SLO p95 TTFT %.1fs\n\n",
			name, *shared, *private, *gen, *ttftBar)
		fmt.Printf("%-6s %9s %9s %9s %11s %8s\n", "conc", "p50 TTFT", "p95 TTFT", "max TTFT", "tok/s total", "errors")
		fmt.Printf("%-6s %9s %9s %9s %11s %8s\n", "----", "--------", "--------", "--------", "-----------", "------")
	}

	var lastGood int
	for _, n := range levels {
		r := runLevel(c, name, prefix, n, *private, *gen, *rounds)
		if *asJSON {
			b, _ := json.Marshal(r)
			fmt.Println(string(b))
		} else {
			fmt.Printf("%-6d %8.2fs %8.2fs %8.2fs %11.1f %8d\n",
				r.Concurrency, r.P50TTFT, r.P95TTFT, r.MaxTTFT, r.TokensPerSec, r.Errors)
		}
		if r.Errors == 0 && r.P95TTFT <= *ttftBar {
			lastGood = n
		}
	}

	if !*asJSON {
		fmt.Println()
		if lastGood == 0 {
			fmt.Printf("No level met the %.1fs p95 bar — the endpoint is over budget at every concurrency tried.\n", *ttftBar)
			return
		}
		fmt.Printf("MEASURED: %d concurrent sessions hold the %.1fs p95 TTFT bar.\n", lastGood, *ttftBar)
		fmt.Printf("At a 2%% duty cycle that is ~%d developers per rental (50%% safety).\n", lastGood*50/2)
		fmt.Println("Compare pool.DefaultPerMember, and see docs/internal/CAPACITY_MODEL.md.")
	}
}

// Result is one concurrency level's measurement.
type Result struct {
	Concurrency  int     `json:"concurrency"`
	P50TTFT      float64 `json:"p50_ttft_s"`
	P95TTFT      float64 `json:"p95_ttft_s"`
	MaxTTFT      float64 `json:"max_ttft_s"`
	TokensPerSec float64 `json:"tokens_per_sec_total"`
	Errors       int     `json:"errors"`
}

// runLevel drives n sessions concurrently, each sending the SAME shared prefix
// followed by its own private context — the shape of a team on one codebase.
func runLevel(c *client, model, prefix string, n, private, gen, rounds int) Result {
	var (
		mu     sync.Mutex
		ttfts  []float64
		tokens int
		errs   int
	)
	start := time.Now()

	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(session int) {
			defer wg.Done()
			for r := 0; r < rounds; r++ {
				body := prefix + "\n\n" + lorem(private) +
					fmt.Sprintf("\n\nSession %d, request %d. Answer in one short sentence.", session, r)
				ttft, got, err := c.streamOnce(model, body, gen)
				mu.Lock()
				if err != nil {
					errs++
				} else {
					ttfts = append(ttfts, ttft.Seconds())
					tokens += got
				}
				mu.Unlock()
			}
		}(i)
	}
	wg.Wait()

	elapsed := time.Since(start).Seconds()
	res := Result{Concurrency: n, Errors: errs}
	if elapsed > 0 {
		res.TokensPerSec = float64(tokens) / elapsed
	}
	res.P50TTFT, res.P95TTFT, res.MaxTTFT = percentile(ttfts, 50), percentile(ttfts, 95), percentile(ttfts, 100)
	return res
}

// percentile returns the p-th percentile of xs (p in 0..100). Empty is 0.
// Nearest-rank, so p100 is the max and a single sample is its own percentile.
func percentile(xs []float64, p float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	if p <= 0 {
		return s[0]
	}
	rank := int((p/100)*float64(len(s)) + 0.999999) // ceil
	if rank < 1 {
		rank = 1
	}
	if rank > len(s) {
		rank = len(s)
	}
	return s[rank-1]
}

// parseSweep reads "1,2,4,8" into levels, rejecting anything that would make a
// meaningless run.
func parseSweep(s string) ([]int, error) {
	var out []int
	for _, f := range strings.Split(s, ",") {
		f = strings.TrimSpace(f)
		if f == "" {
			continue
		}
		n, err := strconv.Atoi(f)
		if err != nil || n < 1 {
			return nil, fmt.Errorf("bad concurrency %q: want positive integers like 1,2,4,8", f)
		}
		out = append(out, n)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("empty sweep")
	}
	sort.Ints(out)
	return out, nil
}

// lorem returns roughly n tokens of filler. Deterministic per length so every
// session sends a byte-identical shared prefix — which is what lets the server's
// prefix cache do the thing under test.
func lorem(tokens int) string {
	if tokens <= 0 {
		return ""
	}
	words := []string{"package", "func", "return", "error", "context", "value", "struct",
		"interface", "server", "client", "request", "handler", "buffer", "result"}
	rng := rand.New(rand.NewSource(int64(tokens))) // seeded by length: same text every run
	var b strings.Builder
	for i := 0; i < tokens; i++ {
		b.WriteString(words[rng.Intn(len(words))])
		b.WriteByte(' ')
	}
	return b.String()
}

// ---- OpenAI-compatible client (streaming, for real TTFT) --------------------

type client struct {
	base  string
	token string
	http  *http.Client
}

func (c *client) url(path string) string {
	u := c.base
	if !strings.HasSuffix(u, "/v1") {
		u += "/v1"
	}
	return u + path
}

func (c *client) do(req *http.Request) (*http.Response, error) {
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}
	req.Header.Set("Content-Type", "application/json")
	return c.http.Do(req)
}

func (c *client) firstModel() (string, error) {
	req, err := http.NewRequest(http.MethodGet, c.url("/models"), nil)
	if err != nil {
		return "", err
	}
	resp, err := c.do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	var out struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", err
	}
	if len(out.Data) == 0 {
		return "", fmt.Errorf("no models served")
	}
	return out.Data[0].ID, nil
}

// streamOnce sends one streaming chat completion and returns the time to the
// FIRST content delta — the number the SLO is written against — plus how many
// deltas arrived.
func (c *client) streamOnce(model, content string, maxTokens int) (time.Duration, int, error) {
	payload := map[string]any{
		"model":      model,
		"stream":     true,
		"max_tokens": maxTokens,
		"messages":   []map[string]string{{"role": "user", "content": content}},
	}
	b, err := json.Marshal(payload)
	if err != nil {
		return 0, 0, err
	}
	req, err := http.NewRequest(http.MethodPost, c.url("/chat/completions"), bytes.NewReader(b))
	if err != nil {
		return 0, 0, err
	}
	sent := time.Now()
	resp, err := c.do(req)
	if err != nil {
		return 0, 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return 0, 0, fmt.Errorf("status %d: %s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}

	var ttft time.Duration
	count := 0
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			break
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if json.Unmarshal([]byte(data), &chunk) != nil || len(chunk.Choices) == 0 {
			continue
		}
		if chunk.Choices[0].Delta.Content == "" {
			continue
		}
		if ttft == 0 {
			ttft = time.Since(sent)
		}
		count++
	}
	if err := sc.Err(); err != nil {
		return 0, 0, err
	}
	if ttft == 0 {
		return 0, 0, fmt.Errorf("no content delta received")
	}
	return ttft, count, nil
}
