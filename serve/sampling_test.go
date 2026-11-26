package serve

import (
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeModel(t *testing.T, files map[string]string) string {
	t.Helper()
	dir := t.TempDir()
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func TestAModelIsSampledAsItsGenerationConfigSays(t *testing.T) {
	qwen := `{"do_sample": true, "temperature": 1.0, "top_k": 20, "top_p": 0.95}`
	cases := []struct {
		name  string
		files map[string]string
		env   string
		want  sampling
	}{
		{"generation_config.json", map[string]string{"generation_config.json": qwen}, "", sampling{temp: 1, topP: 0.95, topK: 20, repeatPenalty: 1}},
		{"the copy inside config.json", map[string]string{"config.json": `{"generation_config": ` + qwen + `}`}, "", sampling{temp: 1, topP: 0.95, topK: 20, repeatPenalty: 1}},
		{"its own repetition penalty", map[string]string{"generation_config.json": `{"temperature": 0.7, "repetition_penalty": 1.05}`}, "", sampling{temp: 0.7, topP: 1, repeatPenalty: 1.05}},
		{"no config", map[string]string{"config.json": `{"model_type": "qwen3"}`}, "", janisSampling},
		{"a config with no sampling in it", map[string]string{"generation_config.json": `{"eos_token_id": 2}`}, "", janisSampling},
		{"do_sample false", map[string]string{"generation_config.json": `{"do_sample": false, "temperature": 1.0}`}, "", janisSampling},
		{"JANIS_MODEL_SAMPLING=0", map[string]string{"generation_config.json": qwen}, "0", janisSampling},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			t.Setenv("JANIS_MODEL_SAMPLING", c.env)
			if got := modelSampling(writeModel(t, c.files)); got != c.want {
				t.Fatalf("sampling %+v, want %+v", got, c.want)
			}
		})
	}
}

// A request that names no sampling gets the loaded model's; one that names
// some keeps its own values.
func TestARequestWithNoSamplingGetsTheModels(t *testing.T) {
	installFake(t, "Qwen3-8B-4bit", 40960)
	home, _ := os.UserHomeDir()
	gen := filepath.Join(home, ".cache/huggingface/hub/models--mlx-community--Qwen3-8B-4bit/snapshots/abc/generation_config.json")
	if err := os.WriteFile(gen, []byte(`{"do_sample": true, "temperature": 1.0, "top_k": 20, "top_p": 0.95}`), 0o644); err != nil {
		t.Fatal(err)
	}
	eng := &fakeEngine{reply: "ok"}
	srv := newTestServer(t, eng)
	post := func(body string) {
		t.Helper()
		resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode != 200 {
			t.Fatalf("status %d", resp.StatusCode)
		}
	}
	post(`{"messages":[{"role":"user","content":"hi"}]}`)
	if o := eng.opts; o.Temp != 1 || o.TopP != 0.95 || o.TopK != 20 || o.RepeatPenalty != 1 {
		t.Fatalf("no sampling asked: temp %v top_p %v top_k %v penalty %v, want the model's 1 0.95 20 1", o.Temp, o.TopP, o.TopK, o.RepeatPenalty)
	}
	post(`{"temperature":0.2,"top_p":0.5,"top_k":5,"repetition_penalty":1.3,"messages":[{"role":"user","content":"hi"}]}`)
	if o := eng.opts; o.Temp != 0.2 || o.TopP != 0.5 || o.TopK != 5 || o.RepeatPenalty != 1.3 {
		t.Fatalf("sampling asked: temp %v top_p %v top_k %v penalty %v, want 0.2 0.5 5 1.3", o.Temp, o.TopP, o.TopK, o.RepeatPenalty)
	}
}
