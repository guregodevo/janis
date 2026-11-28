// Package evals measures a model a janis server serves on a fixed set of
// problems with fixed seeds, and compares two such measurements problem by
// problem. It is the harness of a weight edit (serve --edit): the checkpoint
// and the edited model answer the same prompts with the same seeds, so a
// problem that changes its answer changed it because of the edit.
package evals

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
)

// Problem is one line of a problems file: a question whose answer is put in
// \boxed{}, and that answer.
type Problem struct {
	Question string `json:"question"`
	Answer   string `json:"answer"`
}

// ReadProblems reads a JSONL file of problems ({"question", "answer"}; the
// answer a string or a number) and the SHA-256 of its bytes, which names the
// set in a Score.
func ReadProblems(path string) ([]Problem, string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, "", err
	}
	sum := sha256.Sum256(b)
	var out []Problem
	sc := bufio.NewScanner(bytes.NewReader(b))
	sc.Buffer(make([]byte, 1<<20), 16<<20)
	for n := 1; sc.Scan(); n++ {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var raw struct {
			Question string          `json:"question"`
			Answer   json.RawMessage `json:"answer"`
		}
		if err := json.Unmarshal(line, &raw); err != nil {
			return nil, "", fmt.Errorf("%s:%d: %w", path, n, err)
		}
		ans := strings.TrimSpace(string(raw.Answer))
		if s, err := strconv.Unquote(ans); err == nil {
			ans = s
		}
		if raw.Question == "" || ans == "" {
			return nil, "", fmt.Errorf("%s:%d: a problem needs a question and an answer", path, n)
		}
		out = append(out, Problem{Question: raw.Question, Answer: ans})
	}
	if err := sc.Err(); err != nil {
		return nil, "", err
	}
	if len(out) == 0 {
		return nil, "", fmt.Errorf("%s: no problem in it", path)
	}
	return out, hex.EncodeToString(sum[:]), nil
}

// Config is how the problems are asked. Problem i is sent with seed Seed+i.
type Config struct {
	URL             string // the server, http://127.0.0.1:8090
	Model           string // empty: the model the server has loaded
	Seed            int
	MaxTokens       int
	Parallel        int // requests in flight (JANIS_PARALLEL on the server)
	Temperature     float64
	TopP            float64
	TopK            int
	ReasoningEffort string // "high" turns thinking on; empty leaves it off
	// RepetitionPenalty, when above 0, is sent as repetition_penalty (janis:
	// a token seen in the last 64 generated has its logit divided by it).
	// 0 leaves the server's own.
	RepetitionPenalty float64
}

// Result is one problem's answer.
type Result struct {
	Problem int    `json:"problem"`
	Seed    int    `json:"seed"`
	Correct bool   `json:"correct"`
	AtCap   bool   `json:"at_cap"` // the reply ran out of tokens
	Tokens  int    `json:"tokens"`
	Want    string `json:"want"`
	Got     string `json:"got"`  // the last \boxed{} of the reply, empty when none
	Tail    string `json:"tail"` // the reply's last characters
}

// Score is one measurement: what was asked, of which model with which
// edits, and each problem's result.
type Score struct {
	Model     string `json:"model"`
	Problems  string `json:"problems"` // SHA-256 of the problems file
	Seed      int    `json:"seed"`
	MaxTokens int    `json:"max_tokens"`
	// RepetitionPenalty is the penalty asked for, 0 for the server's own.
	// Two scores that differ in it still pair: a penalty is an arm to beat.
	RepetitionPenalty float64  `json:"repetition_penalty,omitempty"`
	Edits             []string `json:"edits"` // the weights the server had replaced
	N                 int      `json:"n"`
	Correct           int      `json:"correct"`
	AtCap             int      `json:"at_cap"`
	MeanTokens        float64  `json:"mean_tokens"`
	Results           []Result `json:"results"`
}

// Summary is the one line a run prints.
func (s Score) Summary() string {
	return fmt.Sprintf("correct %d/%d  at cap %d/%d  mean tokens %.0f", s.Correct, s.N, s.AtCap, s.N, s.MeanTokens)
}

const tailChars = 1500

// Run asks every problem and grades each reply. A request that fails stops
// the run: a score with a hole in it would not pair with another.
func Run(ctx context.Context, c Config, problems []Problem, set string) (Score, error) {
	if c.Parallel < 1 {
		c.Parallel = 1
	}
	edits, err := serverEdits(ctx, c.URL)
	if err != nil {
		return Score{}, err
	}
	results := make([]Result, len(problems))
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	// The first failure is the one reported: the requests it cancels fail
	// after it, with the cancellation for a reason.
	var failMu sync.Mutex
	var failed error
	next := make(chan int)
	var wg sync.WaitGroup
	for w := 0; w < c.Parallel; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range next {
				r, err := ask(ctx, c, i, problems[i])
				if err != nil {
					failMu.Lock()
					if failed == nil {
						failed = fmt.Errorf("problem %d: %w", i, err)
					}
					failMu.Unlock()
					cancel()
					continue
				}
				results[i] = r
			}
		}()
	}
feed:
	for i := range problems {
		select {
		case next <- i:
		case <-ctx.Done():
			break feed
		}
	}
	close(next)
	wg.Wait()
	if failed != nil {
		return Score{}, failed
	}
	if err := ctx.Err(); err != nil {
		return Score{}, err
	}
	s := Score{Model: c.Model, Problems: set, Seed: c.Seed, MaxTokens: c.MaxTokens, RepetitionPenalty: c.RepetitionPenalty,
		Edits: edits, N: len(results), Results: results}
	tokens := 0
	for _, r := range results {
		if r.Correct {
			s.Correct++
		}
		if r.AtCap {
			s.AtCap++
		}
		tokens += r.Tokens
	}
	s.MeanTokens = float64(tokens) / float64(len(results))
	return s, nil
}

func ask(ctx context.Context, c Config, i int, p Problem) (Result, error) {
	body := map[string]any{
		"messages":    []map[string]string{{"role": "user", "content": p.Question}},
		"temperature": c.Temperature,
		"top_p":       c.TopP,
		"top_k":       c.TopK,
		"max_tokens":  c.MaxTokens,
		"seed":        c.Seed + i,
	}
	if c.Model != "" {
		body["model"] = c.Model
	}
	if c.ReasoningEffort != "" {
		body["reasoning_effort"] = c.ReasoningEffort
	}
	if c.RepetitionPenalty > 0 {
		body["repetition_penalty"] = c.RepetitionPenalty
	}
	var reply struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	if err := Call(ctx, http.MethodPost, c.URL+"/v1/chat/completions", body, &reply); err != nil {
		return Result{}, err
	}
	if len(reply.Choices) == 0 {
		return Result{}, fmt.Errorf("the reply has no choice")
	}
	text := reply.Choices[0].Message.Content
	got := LastBoxed(text)
	r := Result{
		Problem: i,
		Seed:    c.Seed + i,
		AtCap:   reply.Choices[0].FinishReason == "length",
		Tokens:  reply.Usage.CompletionTokens,
		Want:    p.Answer,
		Got:     got,
		Correct: got != "" && Same(got, p.Answer),
		Tail:    tail(text, tailChars),
	}
	return r, nil
}

// serverEdits is the list of weights the server has replaced; a server
// without --edit has none to say.
func serverEdits(ctx context.Context, url string) ([]string, error) {
	var w struct {
		Edits []string `json:"edits"`
	}
	err := Call(ctx, http.MethodGet, url+"/v1/weights", nil, &w)
	var he *HTTPError
	if errors.As(err, &he) && (he.Code == http.StatusNotFound || he.Code == http.StatusNotImplemented) {
		return []string{}, nil
	}
	if err != nil {
		return nil, err
	}
	if w.Edits == nil {
		w.Edits = []string{}
	}
	return w.Edits, nil
}

// HTTPError is an answer the server gave with a status other than 200.
type HTTPError struct {
	Code int
	Body string
}

func (e *HTTPError) Error() string { return fmt.Sprintf("%d: %s", e.Code, strings.TrimSpace(e.Body)) }

// Call sends a JSON body (none when body is nil) and decodes a JSON answer
// into out (skipped when out is nil).
func Call(ctx context.Context, method, url string, body, out any) error {
	var rd io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return err
		}
		rd = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, url, rd)
	if err != nil {
		return err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusOK {
		return &HTTPError{Code: resp.StatusCode, Body: string(b)}
	}
	if out == nil {
		return nil
	}
	return json.Unmarshal(b, out)
}

func tail(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[len(r)-n:])
}

// LastBoxed is what the last \boxed{...} of a reply holds, braces inside it
// kept (\boxed{\frac{1}{2}}); empty when there is none or it is not closed.
func LastBoxed(s string) string {
	const open = `\boxed{`
	i := strings.LastIndex(s, open)
	if i < 0 {
		return ""
	}
	start := i + len(open)
	depth := 1
	for j := start; j < len(s); j++ {
		switch s[j] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return strings.TrimSpace(s[start:j])
			}
		}
	}
	return ""
}

var unformat = strings.NewReplacer(" ", "", `\,`, "", `\!`, "", `\$`, "", "$", "", ",", "", "−", "-")

// Same says whether an answer matches the expected one: equal once spaces,
// dollar signs and thousands separators are dropped, or equal as numbers.
func Same(got, want string) bool {
	g, w := unformat.Replace(got), unformat.Replace(want)
	if g == w {
		return true
	}
	gf, err1 := strconv.ParseFloat(g, 64)
	wf, err2 := strconv.ParseFloat(w, 64)
	return err1 == nil && err2 == nil && gf == wf
}

// Paired is two scores compared problem by problem.
type Paired struct {
	N, Base, Edited int     // problems, and how many each got right
	Gained, Lost    int     // right only with the edit, right only without
	P               float64 // one-sided exact sign test: the chance of gaining this much by luck
}

// Compare pairs two scores of the same problems, seeds and cap. It refuses
// two that were not asked the same way: their difference would not be the
// edit's.
func Compare(base, edited Score) (Paired, error) {
	switch {
	case base.Problems != edited.Problems:
		return Paired{}, fmt.Errorf("the two scores are of different problems files (%.12s, %.12s)", base.Problems, edited.Problems)
	case base.Seed != edited.Seed:
		return Paired{}, fmt.Errorf("the two scores have different seeds (%d, %d)", base.Seed, edited.Seed)
	case base.MaxTokens != edited.MaxTokens:
		return Paired{}, fmt.Errorf("the two scores have different token caps (%d, %d)", base.MaxTokens, edited.MaxTokens)
	case base.Model != edited.Model:
		return Paired{}, fmt.Errorf("the two scores are of different models (%q, %q)", base.Model, edited.Model)
	case len(base.Results) != len(edited.Results):
		return Paired{}, fmt.Errorf("the two scores have %d and %d results", len(base.Results), len(edited.Results))
	}
	p := Paired{N: len(base.Results)}
	for i, b := range base.Results {
		e := edited.Results[i]
		if b.Problem != e.Problem || b.Seed != e.Seed {
			return Paired{}, fmt.Errorf("result %d is problem %d seed %d in one and problem %d seed %d in the other", i, b.Problem, b.Seed, e.Problem, e.Seed)
		}
		if b.Correct {
			p.Base++
		}
		if e.Correct {
			p.Edited++
		}
		switch {
		case e.Correct && !b.Correct:
			p.Gained++
		case b.Correct && !e.Correct:
			p.Lost++
		}
	}
	p.P = signTest(p.Gained, p.Gained+p.Lost)
	return p, nil
}

// signTest is P(X >= g) for X ~ Binomial(k, 1/2): under no effect each
// problem that changed is a gain or a loss with equal chance. It is the
// exact form of the paired sign-flip permutation test on 0/1 scores.
func signTest(g, k int) float64 {
	if k == 0 {
		return 1
	}
	p := 0.0
	for j := g; j <= k; j++ {
		lc, _ := math.Lgamma(float64(k + 1))
		l1, _ := math.Lgamma(float64(j + 1))
		l2, _ := math.Lgamma(float64(k - j + 1))
		p += math.Exp(lc - l1 - l2 - float64(k)*math.Ln2)
	}
	return math.Min(p, 1)
}

// Pass is the rule: the edit gained more problems than it lost, at p < alpha.
func (p Paired) Pass(alpha float64) bool {
	return p.Gained > p.Lost && p.P < alpha
}
