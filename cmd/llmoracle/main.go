//go:build darwin && arm64 && cgo

// Command llmoracle drives the engine over RAW token ids — no chat template,
// no stop tokens — and prints the greedy continuation. Its output is diffed
// against a reference implementation (mlx-lm in Python) running the same ids,
// which is how a new architecture port is proven token-exact.
//
// Usage: llmoracle <model-name-substring> <comma-separated ids> <n-greedy>
package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/guregodevo/janis/qwen"
	"github.com/guregodevo/janis/qwen35"
	"github.com/guregodevo/janis/safetensors"

	"github.com/guregodevo/janis/mlxc"
)

func main() {
	if len(os.Args) < 4 {
		fmt.Println("usage: llmoracle <model> <ids> <n>")
		os.Exit(2)
	}
	model := os.Args[1]
	var ids []int32
	for _, s := range strings.Split(os.Args[2], ",") {
		n, err := strconv.Atoi(strings.TrimSpace(s))
		if err != nil {
			panic(err)
		}
		ids = append(ids, int32(n))
	}
	nGen, _ := strconv.Atoi(os.Args[3])

	home, _ := os.UserHomeDir()
	dirs, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--mlx-community--"+model+"/snapshots/*"))
	if len(dirs) == 0 {
		fmt.Printf("model %q not cached\n", model)
		os.Exit(1)
	}
	cfg, err := qwen.LoadConfig(dirs[0])
	if err != nil {
		panic(err)
	}
	if cfg.ModelType != "qwen3_5" {
		fmt.Println("llmoracle currently drives qwen3_5 only")
		os.Exit(1)
	}
	st, err := safetensors.OpenModel(dirs[0])
	if err != nil {
		panic(err)
	}
	b := mlxc.New()
	defer b.Close()
	m, err := qwen35.LoadModel(b, st, cfg)
	if err != nil {
		panic(err)
	}
	b.PinAll() // weights must survive the session's per-step sweeps

	// Top-5 last-position logits from a stateless pass — the numerical
	// fidelity check against the reference (greedy drift alone can't
	// distinguish tiny dtype differences from a real bug).
	logits := b.Floats(m.Logits(b, ids))
	type tl struct {
		id int
		v  float32
	}
	top := make([]tl, 0, 6)
	for i, v := range logits {
		top = append(top, tl{i, v})
		for j := len(top) - 1; j > 0 && top[j].v > top[j-1].v; j-- {
			top[j], top[j-1] = top[j-1], top[j]
		}
		if len(top) > 5 {
			top = top[:5]
		}
	}
	fmt.Print("top5:")
	for _, t := range top {
		fmt.Printf(" (%d, %.4f)", t.id, t.v)
	}
	fmt.Println()

	sess := m.NewSession()
	out := sess.Generate(b, ids, nGen, qwen.SampleParams{Temp: 0, Stop: map[int32]bool{}})
	strs := make([]string, len(out))
	for i, t := range out {
		strs[i] = strconv.Itoa(int(t))
	}
	fmt.Println("greedy:", strings.Join(strs, ","))
}
