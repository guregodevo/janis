package llm

import (
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/guregodevo/janis/hostfit"
)

// batchEngine opens JANIS_BATCH_MODEL (Qwen3.5-0.8B-MLX-4bit by default)
// with n slots, on the test's own OS thread, as the server's engine thread.
func batchEngine(t *testing.T, n int) *Engine {
	t.Helper()
	if os.Getenv("JANIS_BATCH_E2E") != "1" {
		t.Skip("set JANIS_BATCH_E2E=1 to run")
	}
	runtime.LockOSThread()
	t.Cleanup(runtime.UnlockOSThread)
	name := os.Getenv("JANIS_BATCH_MODEL")
	if name == "" {
		name = "Qwen3.5-0.8B-MLX-4bit"
	}
	dirs, _ := filepath.Glob(filepath.Join(hostfit.CacheDir(), "models--mlx-community--"+name, "snapshots/*"))
	if len(dirs) == 0 {
		t.Skip(name + " not installed")
	}
	old := parallel
	parallel = n
	t.Cleanup(func() { parallel = old })
	e, err := Open(dirs[0])
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(e.Close)
	if e.Slots() != n {
		t.Skipf("Slots() = %d: %s does not batch", e.Slots(), name)
	}
	return e
}

// decodeTogether begins every prompt's reply, then steps them all, each
// finished once done, the way the server's batch loop does.
func decodeTogether(t *testing.T, e *Engine, prompts [][]Message, opts Options) []string {
	t.Helper()
	out := make([]string, len(prompts))
	var live []Reply
	var at []int
	for i, p := range prompts {
		r, err := e.BeginChat(p, opts, nil)
		if err != nil || r == nil {
			t.Fatalf("BeginChat %d: %v %v", i, r, err)
		}
		if r.Done() {
			out[i] = e.FinishChat(r)
			continue
		}
		live, at = append(live, r), append(at, i)
	}
	for len(live) > 0 {
		e.StepChats(live)
		for j := len(live) - 1; j >= 0; j-- {
			if live[j].Done() {
				out[at[j]] = e.FinishChat(live[j])
				live, at = slices.Delete(live, j, j+1), slices.Delete(at, j, j+1)
			}
		}
	}
	return out
}

func userPrompt(text string) []Message { return []Message{{Role: "user", Content: text}} }

// REPLIES DECODED TOGETHER ARE THE REPLIES DECODED ONE AT A TIME.
//
// Greedy decoding of three prompts of different lengths, one at a time with
// Chat, then together on three slots, each on a fresh session: the texts
// are equal.
func TestRepliesDecodedTogetherAreTheRepliesOneAtATime(t *testing.T) {
	e := batchEngine(t, 3)
	prompts := [][]Message{
		userPrompt("Name three prime numbers."),
		userPrompt("Write one sentence about the sea, and nothing else."),
		userPrompt("What is 12 times 12? Answer with the number and a short explanation of how you got it."),
	}
	opts := Options{Temp: 0, MaxTokens: 32}
	// Every reply on a fresh session, both ways: a prompt that reuses another
	// one's prefix is read in other chunks, which can move a greedy pick.
	fresh := func() {
		if e.slotDecoder != nil {
			e.slotDecoder.Settle()
		}
		e.session, e.slots = e.q35.NewSession(), nil
	}
	var alone []string
	for _, p := range prompts {
		fresh()
		alone = append(alone, e.Chat(p, opts))
	}
	fresh()
	together := decodeTogether(t, e, prompts, opts)
	for i := range prompts {
		t.Logf("prompt %d alone    %q", i, alone[i])
		t.Logf("prompt %d together %q", i, together[i])
		if together[i] != alone[i] {
			t.Errorf("prompt %d:\n alone    %q\n together %q", i, alone[i], together[i])
		}
	}
}

// SLOTS ARE REUSED AFTER THEIR BATCH ENDED.
//
// A batch of 4 decodes to its end, then a batch of 8 on the same engine:
// four slots reused, each prefilled alone while the others still held rows
// of the last batch's stacked state. Every reply has text. (Before Release,
// the first reused slot unpinned the stack under the others, and the second
// sliced freed memory.)
func TestSlotsAreReusedAfterTheirBatchEnded(t *testing.T) {
	e := batchEngine(t, 8)
	opts := Options{Temp: 0, MaxTokens: 8}
	for _, n := range []int{4, 8} {
		var prompts [][]Message
		for i := 0; i < n; i++ {
			prompts = append(prompts, userPrompt("Say the word number "+strconv.Itoa(i)+"."))
		}
		for i, r := range decodeTogether(t, e, prompts, opts) {
			if r == "" {
				t.Errorf("n=%d reply %d is empty", n, i)
			}
		}
	}
}

// BATCHED REPLY SPEED: n replies of JANIS_BATCH_TOKENS tokens (128) decoded
// together, for each n in JANIS_BATCH_SIZES ("1,4,8,16,32"). Logs the total
// decode rate; asserts nothing.
func TestBatchedReplySpeed(t *testing.T) {
	sizes := []int{1, 4, 8, 16, 32}
	if v := os.Getenv("JANIS_BATCH_SIZES"); v != "" {
		sizes = nil
		for _, f := range strings.FieldsFunc(v, func(r rune) bool { return r == ',' || r == ' ' }) {
			if n, err := strconv.Atoi(f); err == nil {
				sizes = append(sizes, n)
			}
		}
	}
	tokens := 128
	if v, err := strconv.Atoi(os.Getenv("JANIS_BATCH_TOKENS")); err == nil && v > 0 {
		tokens = v
	}
	max := 0
	for _, n := range sizes {
		max = maxInt(max, n)
	}
	e := batchEngine(t, max)
	for _, n := range sizes {
		var prompts [][]Message
		for i := 0; i < n; i++ {
			prompts = append(prompts, userPrompt("Count from "+strconv.Itoa(i)+" upward, one number per line, and do not stop."))
		}
		opts := Options{Temp: 0, MaxTokens: tokens}
		start := time.Now()
		rs := make([]Reply, n)
		for i, p := range prompts {
			rs[i], _ = e.BeginChat(p, opts, nil)
		}
		prefill := time.Since(start)
		steps, total := 0, 0
		live := slices.Clone(rs)
		for len(live) > 0 {
			e.StepChats(live)
			steps++
			for j := len(live) - 1; j >= 0; j-- {
				if live[j].Done() {
					total += len(live[j].(*chatGen).g.Finish())
					e.FinishChat(live[j])
					live = slices.Delete(live, j, j+1)
				}
			}
		}
		decode := time.Since(start) - prefill
		t.Logf("n=%d: %d tokens in %d steps, decode %.1fs = %.0f tok/s total, %.1f ms a step; prefill %.1fs",
			n, total, steps, decode.Seconds(), float64(total)/decode.Seconds(), decode.Seconds()*1000/float64(maxInt(steps, 1)), prefill.Seconds())
	}
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
