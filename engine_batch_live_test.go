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

// ON FIXED ROWS, REPLIES THAT JOIN AND LEAVE ARE THE REPLIES DECODED ALONE.
//
// JANIS_FIXED_ROWS: a reply joins the batch mid-way, another ends and its
// row is held, then a follow-up of the ended conversation joins on its
// slot, reading the state the held row kept. Greedy, each text equals the
// one Chat gives alone (the follow-up on the session that read the first
// turn). Logs how the steps ran: after the first stack, joins and leaves
// should not stack the rows again.
func TestOnFixedRowsRepliesThatJoinAndLeaveAreTheRepliesAlone(t *testing.T) {
	joinAndLeave(t, 24)
}

// ON FIXED ROWS, A ROW THAT FILLS ITS POOL IS STACKED AGAIN.
//
// Replies 600 tokens long in pools with 256 tokens of room past the
// longest row (JANIS_POOL_SPARE=256): rows fill their pool while they
// decode and when they join. The rows are stacked again, more than once,
// and every reply has its text. The texts are not compared to the replies
// alone: past a few hundred tokens a near tie flips on the rounding of
// another batch size, on fixed rows and off them alike (2026-11-23: the
// same word at the same place, both ways). Every row is taken back and stacked in
// a new pool; the texts stay the ones decoded alone. (A pool made in a
// recorded step was run again at every replay, from the freed old pool:
// a segfault, 2026-11-23.)
func TestOnFixedRowsARowThatFillsItsPoolIsStackedAgain(t *testing.T) {
	t.Setenv("JANIS_POOL_SPARE", "256")
	kinds := joinAndLeave(t, 600)
	if kinds["plain"] < 2 {
		t.Errorf("steps %v: no row stacked again after the first stack", kinds)
	}
}

// joinAndLeave decodes the replies of the fixed-row tests and returns how
// its steps ran; the texts must be the replies alone when maxTokens is
// short enough for no near tie to flip.
func joinAndLeave(t *testing.T, maxTokens int) map[string]int {
	old := fixedRows
	fixedRows = true
	t.Cleanup(func() { fixedRows = old })
	e := batchEngine(t, 3)
	opts := Options{Temp: 0, MaxTokens: maxTokens}
	a := userPrompt("Name three prime numbers.")
	b := userPrompt("Write one sentence about the sea, and nothing else.")
	if maxTokens > 100 {
		a = userPrompt("Name three prime numbers, then explain at length why each is prime.")
		b = userPrompt("Write a long story about the sea.")
	}
	fresh := func() {
		if e.slotDecoder != nil {
			e.slotDecoder.Settle()
		}
		e.session, e.slots = e.q35.NewSession(), nil
	}
	followUp := func(reply string) []Message {
		return append(append([]Message{}, a...), Message{Role: "assistant", Content: reply}, Message{Role: "user", Content: "Now name three more."})
	}
	fresh()
	aloneA := e.Chat(a, opts)
	aloneA2 := e.Chat(followUp(aloneA), opts)
	fresh()
	aloneB := e.Chat(b, opts)
	fresh()

	kinds := map[string]int{}
	begin := func(m []Message) Reply {
		r, err := e.BeginChat(m, opts, nil)
		if err != nil || r == nil {
			t.Fatalf("BeginChat: %v %v", r, err)
		}
		return r
	}
	step := func(live []Reply) {
		e.StepChats(live)
		kinds[e.slotDecoder.Last]++
	}
	ra := begin(a)
	for i := 0; i < 6; i++ {
		step([]Reply{ra})
	}
	rb := begin(b) // joins while a decodes
	live := []Reply{ra, rb}
	for !ra.Done() {
		step(live)
	}
	togetherA := e.FinishChat(ra) // a's row is held from here
	ra2 := begin(followUp(togetherA))
	if ra2.(*chatGen).slot != ra.(*chatGen).slot {
		t.Errorf("the follow-up is on slot %d, the first turn was on %d", ra2.(*chatGen).slot, ra.(*chatGen).slot)
	}
	live = []Reply{rb, ra2}
	out := map[Reply]string{}
	for len(live) > 0 {
		step(live)
		for j := len(live) - 1; j >= 0; j-- {
			if live[j].Done() {
				out[live[j]] = e.FinishChat(live[j])
				live = slices.Delete(live, j, j+1)
			}
		}
	}
	t.Logf("steps: %v", kinds)
	for _, c := range []struct{ name, alone, together string }{
		{"a", aloneA, togetherA}, {"b", aloneB, out[rb]}, {"a's follow-up", aloneA2, out[ra2]},
	} {
		t.Logf("%s alone    %q", c.name, c.alone)
		t.Logf("%s together %q", c.name, c.together)
		if c.together == "" {
			t.Errorf("%s: no text", c.name)
		}
		if maxTokens <= 100 && c.alone != c.together {
			t.Errorf("%s:\n alone    %q\n together %q", c.name, c.alone, c.together)
		}
	}
	return kinds
}
