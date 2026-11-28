package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"

	"github.com/guregodevo/janis/evals"
)

const defaultURL = "http://127.0.0.1:8090"

// evalCmd is janis eval: the problems of a file asked of a running server
// with fixed seeds, the score written to --out. janis eval compare pairs two
// scores.
func evalCmd(args []string, out io.Writer) error {
	if len(args) > 0 && args[0] == "compare" {
		return compareCmd(args[1:], out)
	}
	fs := flag.NewFlagSet("eval", flag.ContinueOnError)
	u := fs.String("url", defaultURL, "the janis server")
	model := fs.String("model", "", "the model to ask (none: the one the server has loaded)")
	problems := fs.String("problems", "", "a JSONL file of {\"question\", \"answer\"}; the answer goes in \\boxed{}")
	dest := fs.String("out", "", "the score file to write")
	seed := fs.Int("seed", 1000, "problem i is asked with seed+i")
	maxTokens := fs.Int("max-tokens", 16384, "the token cap of one reply, thinking included")
	parallel := fs.Int("parallel", 1, "requests in flight (match JANIS_PARALLEL)")
	temp := fs.Float64("temperature", 0.6, "")
	topP := fs.Float64("top-p", 0.95, "")
	topK := fs.Int("top-k", 20, "")
	effort := fs.String("effort", "high", "reasoning_effort: high thinks, none does not")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *problems == "" || *dest == "" {
		return errors.New("janis eval: --problems and --out are needed")
	}
	ps, set, err := evals.ReadProblems(*problems)
	if err != nil {
		return err
	}
	c := evals.Config{URL: *u, Model: *model, Seed: *seed, MaxTokens: *maxTokens, Parallel: *parallel,
		Temperature: *temp, TopP: *topP, TopK: *topK, ReasoningEffort: *effort}
	s, err := evals.Run(context.Background(), c, ps, set)
	if err != nil {
		return err
	}
	if err := writeJSON(*dest, s); err != nil {
		return err
	}
	fmt.Fprintln(out, s.Summary())
	return nil
}

func compareCmd(args []string, out io.Writer) error {
	fs := flag.NewFlagSet("eval compare", flag.ContinueOnError)
	alpha := fs.Float64("alpha", 0.05, "PASS needs p below it")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 2 {
		return errors.New("janis eval compare: BASE.json EDITED.json")
	}
	var base, edited evals.Score
	if err := readJSON(fs.Arg(0), &base); err != nil {
		return err
	}
	if err := readJSON(fs.Arg(1), &edited); err != nil {
		return err
	}
	p, err := evals.Compare(base, edited)
	if err != nil {
		return err
	}
	verdict := "FAIL"
	if p.Pass(*alpha) {
		verdict = "PASS"
	}
	fmt.Fprintf(out, "| | correct | at cap |\n|---|---|---|\n| base | %d/%d | %d |\n| edited | %d/%d | %d |\n\n",
		p.Base, p.N, base.AtCap, p.Edited, p.N, edited.AtCap)
	fmt.Fprintf(out, "gained %d  lost %d  p %.4f  (one-sided exact sign test, alpha %g)\n%s\n", p.Gained, p.Lost, p.P, *alpha, verdict)
	return nil
}

// weightsCmd is janis weights: the weight-edit calls of a server started
// with --edit.
func weightsCmd(args []string, out io.Writer) error {
	if len(args) == 0 {
		return errors.New("janis weights shift|revert|save|load")
	}
	fs := flag.NewFlagSet("weights "+args[0], flag.ContinueOnError)
	u := fs.String("url", defaultURL, "the janis server, started with --edit")
	match := fs.String("match", "", "shift: a regular expression the weights' names match")
	by := fs.Float64("by", 0, "shift: the value added to every element")
	if err := fs.Parse(args[1:]); err != nil {
		return err
	}
	ctx := context.Background()
	var answer struct {
		Edits []string `json:"edits"`
	}
	switch args[0] {
	case "shift":
		n, err := shift(ctx, *u, *match, *by)
		if err != nil {
			return err
		}
		fmt.Fprintf(out, "shifted %d weights by %+g\n", n, *by)
		return nil
	case "revert":
		if err := evals.Call(ctx, http.MethodPost, *u+"/v1/weights/revert", map[string]any{}, &answer); err != nil {
			return err
		}
		fmt.Fprintln(out, "reverted: the checkpoint's weights")
		return nil
	case "save", "load":
		if fs.NArg() != 1 {
			return fmt.Errorf("janis weights %s PATH", args[0])
		}
		// The server reads and writes the path on this host, from its own
		// directory: it gets the path from here.
		p, err := filepath.Abs(fs.Arg(0))
		if err != nil {
			return err
		}
		if err := evals.Call(ctx, http.MethodPost, *u+"/v1/weights/"+args[0], map[string]string{"path": p}, &answer); err != nil {
			return err
		}
		fmt.Fprintf(out, "%s %s: %d weights edited\n", args[0], p, len(answer.Edits))
		return nil
	}
	return fmt.Errorf("janis weights: %q is not shift, revert, save or load", args[0])
}

// shift adds by to every element of each weight whose name matches.
func shift(ctx context.Context, u, match string, by float64) (int, error) {
	if match == "" {
		return 0, errors.New("janis weights shift: --match is needed")
	}
	re, err := regexp.Compile(match)
	if err != nil {
		return 0, err
	}
	var list struct {
		Data []struct {
			Name string `json:"name"`
		} `json:"data"`
	}
	if err := evals.Call(ctx, http.MethodGet, u+"/v1/weights", nil, &list); err != nil {
		return 0, err
	}
	n := 0
	for _, w := range list.Data {
		if !re.MatchString(w.Name) {
			continue
		}
		at := u + "/v1/weights/" + url.PathEscape(w.Name)
		var cur struct {
			Values []float32 `json:"values"`
		}
		if err := evals.Call(ctx, http.MethodGet, at, nil, &cur); err != nil {
			return n, fmt.Errorf("%s: %w", w.Name, err)
		}
		for i := range cur.Values {
			cur.Values[i] += float32(by)
		}
		if err := evals.Call(ctx, http.MethodPost, at, map[string]any{"values": cur.Values}, nil); err != nil {
			return n, fmt.Errorf("%s: %w", w.Name, err)
		}
		n++
	}
	if n == 0 {
		return 0, fmt.Errorf("janis weights shift: no weight's name matches %s", match)
	}
	return n, nil
}

func writeJSON(path string, v any) error {
	b, err := json.MarshalIndent(v, "", " ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0o644)
}

func readJSON(path string, v any) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(b, v); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	return nil
}
