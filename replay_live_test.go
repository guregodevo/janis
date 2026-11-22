//go:build ggml

package llm

import (
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// TestReplayedStepsLive decodes the same prompt greedily on one engine,
// with each step built fresh and then with recorded steps replayed
// (JANIS_GGML_REPLAY for the second run, JANIS_REPLAY_SECOND, default 1;
// cpu records on the CPU backend), and
// compares every step's logits, the second time with the first run's
// tokens forced, so the steps compared do not stop where the replies part.
// The plain steps read the caches padded the way a recorded step does
// (JANIS_GGML_PADDECODE): the two are then the same sums in the same order,
// and the logits must be equal. 300 tokens cross a 256-token bucket, so a
// step is recorded again on the way. JANIS_REPLAY_EXACT=0 drops the
// padding, to see the rounding a padded read differs by.
//
//	JANIS_BACKEND=ggml JANIS_GGML_DEVICE=gpu JANIS_REPLAY_E2E=1 go test -tags ggml . -run TestReplayedStepsLive -v
func TestReplayedStepsLive(t *testing.T) {
	if os.Getenv("JANIS_REPLAY_E2E") != "1" {
		t.Skip("set JANIS_REPLAY_E2E=1 (and JANIS_BACKEND=ggml) to run")
	}
	model := os.Getenv("JANIS_REPLAY_MODEL")
	if model == "" {
		model = "Qwen3.5-0.8B-MLX-4bit"
	}
	second := os.Getenv("JANIS_REPLAY_SECOND")
	if second == "" {
		second = "1"
	}
	home, _ := os.UserHomeDir()
	dirs, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--mlx-community--"+model+"/snapshots/*"))
	if len(dirs) == 0 {
		t.Skip(model + " not installed")
	}
	e, err := Open(dirs[0])
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	msgs := []Message{{Role: "user", Content: "Write a detailed, long essay on the history of the printing press. /no_think"}}

	run := func(replay string, force []int32) ([]int32, [][]float32) {
		t.Setenv("JANIS_GGML_REPLAY", replay)
		var logits [][]float32
		var toks []int32
		e.utilSession.OnLogits = func(l []float32) { logits = append(logits, append([]float32(nil), l...)) }
		e.utilSession.Force = force
		e.ChatUtil(msgs, Options{Temp: 0, MaxTokens: 300, OnToken: func(id int32, _ string) { toks = append(toks, id) }})
		e.utilSession.OnLogits, e.utilSession.Force = nil, nil
		e.utilSession.Reset(e.bk)
		return toks, logits
	}
	exact := os.Getenv("JANIS_REPLAY_EXACT") != "0"
	if exact {
		t.Setenv("JANIS_GGML_PADDECODE", "1")
	}
	built, bl := run("0", nil)
	replayed, _ := run(second, nil)

	// The same tokens through both: every step compared, past where the
	// greedy replies part.
	_, fl := run(second, built)
	same, steps, gap := 0, 0, 0.0
	for i := 0; i < len(bl) && i < len(fl); i++ {
		steps++
		if argmax(bl[i]) == argmax(fl[i]) {
			same++
		}
		for j := range bl[i] {
			gap = math.Max(gap, math.Abs(float64(bl[i][j]-fl[i][j])))
		}
	}
	t.Logf("%s, the same tokens: the top token agrees at %d of %d steps; largest logit difference %.4g", model, same, steps, gap)

	agree := 0
	for agree < len(built) && agree < len(replayed) && built[agree] == replayed[agree] {
		agree++
	}
	t.Logf("%s: the greedy replies agree for %d of %d tokens", model, agree, len(built))
	if exact && (gap != 0 || agree != len(built)) {
		t.Errorf("replayed steps are not the plain ones: logits differ by %.4g, replies agree for %d of %d tokens", gap, agree, len(built))
	}
	if !exact && same*100 < steps*85 {
		t.Errorf("the top token agrees at only %d of %d steps", same, steps)
	}
}

func argmax(v []float32) int {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return best
}

// TestReplayedStepsThenAPromptThatParts decodes with recorded steps, then
// reads a prompt that shares only the system prompt with the first (the
// caches cut back to where they part), then the first again: a recorded
// step's caches must survive the requests after it, and the replies must
// be the plain path's (JANIS_GGML_PADDECODE).
func TestReplayedStepsThenAPromptThatParts(t *testing.T) {
	if os.Getenv("JANIS_REPLAY_E2E") != "1" {
		t.Skip("set JANIS_REPLAY_E2E=1 (and JANIS_BACKEND=ggml) to run")
	}
	model := os.Getenv("JANIS_REPLAY_MODEL")
	if model == "" {
		model = "Qwen3.5-0.8B-MLX-4bit"
	}
	second := os.Getenv("JANIS_REPLAY_SECOND")
	if second == "" {
		second = "1"
	}
	home, _ := os.UserHomeDir()
	dirs, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--mlx-community--"+model+"/snapshots/*"))
	if len(dirs) == 0 {
		t.Skip(model + " not installed")
	}
	e, err := Open(dirs[0])
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	t.Setenv("JANIS_GGML_PADDECODE", "1")
	sys := Message{Role: "system", Content: "You are a careful assistant. Answer in plain sentences."}
	prompts := [][]Message{
		{sys, {Role: "user", Content: "Name three rivers in Europe. /no_think"}},
		{sys, {Role: "user", Content: "Name three mountains in Asia. /no_think"}},
		{sys, {Role: "user", Content: "Name three rivers in Europe. /no_think"}},
	}
	replies := func(replay string) []string {
		t.Setenv("JANIS_GGML_REPLAY", replay)
		var out []string
		for _, p := range prompts {
			out = append(out, e.Chat(p, Options{Temp: 0, MaxTokens: 40}))
			// What a server does between requests (another session's
			// call sweeps; the collector runs the finalizers).
			runtime.GC()
			e.ChatUtil([]Message{{Role: "user", Content: "Say OK. /no_think"}}, Options{Temp: 0, MaxTokens: 4})
			runtime.GC()
		}
		e.session.Reset(e.bk)
		return out
	}
	plain := replies("0")
	replayed := replies(second)
	same := 0
	for i := range plain {
		if plain[i] != replayed[i] {
			t.Errorf("prompt %d: replayed %q, plain %q", i, replayed[i], plain[i])
		} else {
			same++
		}
	}
	t.Logf("%s: %d of %d replies the same", model, same, len(plain))
}
