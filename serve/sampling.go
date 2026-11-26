package serve

import (
	"encoding/json"
	"os"
	"path/filepath"
)

// sampling is how a reply is sampled when its request does not say.
type sampling struct {
	temp, topP float32
	topK       int
	// repeatPenalty is 1 for none; 0 leaves the engine's own
	// (JANIS_REPEAT_PENALTY, 1.1 by default).
	repeatPenalty float32
}

// janisSampling is the sampling of a model that ships no generation config.
var janisSampling = sampling{temp: defaultTemp, topP: defaultTopP}

// modelSampling is the sampling the model in dir was released with: its
// generation_config.json, else the generation_config inside config.json (an
// MLX conversion keeps a copy there). Qwen3.8-27B names temperature 1.0,
// top_p 0.95, top_k 20 and no repetition penalty. A config that names no
// penalty means none: 1.1 over 64 tokens is janis's guard for a model that
// says nothing. A model with no config, or one that says do_sample false,
// keeps janis's own. JANIS_MODEL_SAMPLING=0 keeps janis's own for every model.
//
// Measured on Qwen3.8-27B-8bit, one LiveCodeBench problem (abc388_g), one
// call (2026-11-26, A100): 1 of 4 solved at temperature 0.6, top_p 0.95,
// penalty 1.1; the original weights in transformers solved 3 of 4 at the
// model's own values, and greedy solved it in neither.
func modelSampling(dir string) sampling {
	if os.Getenv("JANIS_MODEL_SAMPLING") == "0" {
		return janisSampling
	}
	type genConfig struct {
		DoSample          *bool    `json:"do_sample"`
		Temperature       *float64 `json:"temperature"`
		TopP              *float64 `json:"top_p"`
		TopK              *int     `json:"top_k"`
		RepetitionPenalty *float64 `json:"repetition_penalty"`
	}
	var g *genConfig
	if b, err := os.ReadFile(filepath.Join(dir, "generation_config.json")); err == nil {
		var c genConfig
		if json.Unmarshal(b, &c) == nil {
			g = &c
		}
	}
	if g == nil {
		if b, err := os.ReadFile(filepath.Join(dir, "config.json")); err == nil {
			var c struct {
				GenerationConfig *genConfig `json:"generation_config"`
			}
			if json.Unmarshal(b, &c) == nil {
				g = c.GenerationConfig
			}
		}
	}
	if g == nil || (g.DoSample != nil && !*g.DoSample) || (g.Temperature == nil && g.TopP == nil && g.TopK == nil) {
		return janisSampling
	}
	s := sampling{temp: 1, topP: 1, repeatPenalty: 1} // transformers' values for what a config leaves out
	if g.Temperature != nil && *g.Temperature > 0 {
		s.temp = float32(*g.Temperature)
	}
	if g.TopP != nil && *g.TopP > 0 && *g.TopP <= 1 {
		s.topP = float32(*g.TopP)
	}
	if g.TopK != nil && *g.TopK > 0 {
		s.topK = *g.TopK
	}
	if g.RepetitionPenalty != nil && *g.RepetitionPenalty > 0 {
		s.repeatPenalty = float32(*g.RepetitionPenalty)
	}
	return s
}
