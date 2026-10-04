// Command llmsmoke proves memdoor can embed the engine in-process: load a model
// from the HF cache and run one chat, no subprocess/HTTP.
//
// Usage: llmsmoke [model-name-substring] [prompt]
// Defaults to Qwen2.5-3B-Instruct-4bit. Set JANIS_BACKEND=cpu (on a Mac) to
// run the pure-Go CPU backend instead of MLX — running it both ways on the same
// model at temp=0 is how we verify the CPU forward matches MLX end-to-end.
package main

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime/pprof"
	"strconv"
	"time"

	"github.com/guregodevo/janis"
)

func main() {
	model := "Qwen2.5-3B-Instruct-4bit"
	prompt := "What is the capital of France? Answer in one short sentence."
	if len(os.Args) > 1 {
		model = os.Args[1]
	}
	if len(os.Args) > 2 {
		prompt = os.Args[2]
	}

	home, _ := os.UserHomeDir()
	dirs, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--mlx-community--"+model+"/snapshots/*"))
	if len(dirs) == 0 {
		fmt.Printf("model %q not cached\n", model)
		os.Exit(1)
	}

	backend := os.Getenv("JANIS_BACKEND")
	if backend == "" {
		backend = "mlx (default)"
	}
	fmt.Printf("backend=%s model=%s\n", backend, model)

	eng, err := llm.Open(dirs[0])
	if err != nil {
		panic(err)
	}
	defer eng.Close()
	fmt.Println("loaded:", eng.ModelType())

	// JANIS_CPUPROFILE=path writes a CPU profile around the generation.
	if p := os.Getenv("JANIS_CPUPROFILE"); p != "" {
		f, ferr := os.Create(p)
		if ferr != nil {
			panic(ferr)
		}
		_ = pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	maxTok := 32
	if v := os.Getenv("JANIS_MAXTOK"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			maxTok = n
		}
	}
	start := time.Now()
	reply := eng.Chat([]llm.Message{
		{Role: "user", Content: prompt},
	}, llm.Options{Temp: 0, MaxTokens: maxTok})
	dur := time.Since(start)
	ntok := eng.NumTokens(reply)
	fmt.Printf("REPLY: %s\n", reply)
	fmt.Printf("[%d tok in %.1fs → %.2f tok/s]\n", ntok, dur.Seconds(), float64(ntok)/dur.Seconds())
}
