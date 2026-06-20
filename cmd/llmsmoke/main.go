// Command llmsmoke proves memdoor can embed the engine in-process: load a model
// from the HF cache and run one chat, no subprocess/HTTP.
package main

import (
	"fmt"
	"os"
	"path/filepath"

	"memdoor/llm"
)

func main() {
	home, _ := os.UserHomeDir()
	dirs, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--mlx-community--Qwen2.5-3B-Instruct-4bit/snapshots/*"))
	if len(dirs) == 0 {
		fmt.Println("model not cached")
		os.Exit(1)
	}

	eng, err := llm.Open(dirs[0])
	if err != nil {
		panic(err)
	}
	defer eng.Close()
	fmt.Println("loaded:", eng.ModelType())

	reply := eng.Chat([]llm.Message{
		{Role: "user", Content: "What is the capital of France? Answer in one short sentence."},
	}, llm.Options{Temp: 0, MaxTokens: 32})
	fmt.Println("REPLY:", reply)
}
