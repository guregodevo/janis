package hostfit

import "testing"

// The model cache holds embedders next to chat models. Offering one as a brain
// breaks every turn — an embedder cannot answer, so the failure is silent and
// total.
func TestEmbeddersAreNotOfferedAsBrains(t *testing.T) {
	for _, name := range []string{"bge-m3-mlx-fp16", "bge-small-en", "gte-large", "e5-mistral-7b", "nomic-embed-text", "bge-reranker-v2"} {
		if !IsEmbeddingModel(name) {
			t.Errorf("%s should be recognised as an embedder", name)
		}
	}
	for _, name := range []string{"Qwen3-8B-4bit", "Qwen2.5-Coder-14B-Instruct-4bit", "Llama-3.1-8B", "Mistral-Small-Instruct"} {
		if IsEmbeddingModel(name) {
			t.Errorf("%s is a chat model and must stay selectable", name)
		}
	}
}
