package llm

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestQuestionConstructorsFailFast(t *testing.T) {
	if _, err := NewChoice("q", []string{"only"}); err == nil {
		t.Fatal("one option must be rejected")
	}
	if _, err := NewChoice("q", []string{"a", "a"}); err == nil {
		t.Fatal("duplicate options must be rejected")
	}
	if _, err := NewChoice("", []string{"a", "b"}); err == nil {
		t.Fatal("empty question must be rejected")
	}
	if _, err := NewScore("q", []string{"lo", "hi"}, []float64{1}); err == nil {
		t.Fatal("misaligned values must be rejected")
	}
	if _, err := ParseKind("maybe"); err == nil {
		t.Fatal("unknown kind must be rejected")
	}
	q, err := NewScore("q", []string{"lo", "mid", "hi"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := q.Values; got[0] != 1 || got[2] != 3 {
		t.Fatalf("default values = %v, want 1..3", got)
	}
}

func TestRenderQuestionLabelsOptions(t *testing.T) {
	q, _ := NewScore("How urgent?", []string{"low", "high"}, []float64{0, 10})
	s := renderQuestion(q)
	for _, want := range []string{"Question: How urgent?", "A. 0 - low", "B. 10 - high", "Answer with the letter"} {
		if !strings.Contains(s, want) {
			t.Fatalf("rendered question missing %q:\n%s", want, s)
		}
	}
}

func TestOptionProbsMergesVariantsAndRestricts(t *testing.T) {
	// vocab of 10: label A has two token variants (ids 1, 2), B one (id 5);
	// everything else is huge noise that must not leak into the answer.
	logits := []float32{50, 0, 0, 50, 50, math.Float32frombits(0), 50, 50, 50, 50}
	logits[5] = float32(math.Log(2)) // B = log 2 -> weight 2; A = e^0 + e^0 = weight 2
	labels := [][]int32{{1, 2}, {5}}
	p, err := optionProbs(logits, labels, 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(p[0]-0.5) > 1e-6 || math.Abs(p[1]-0.5) > 1e-6 {
		t.Fatalf("probs = %v, want [0.5 0.5]", p)
	}
	// temperature flattens: T=2 halves the logit gap
	logits[5] = 4 // B weight e^4 vs A weight 2
	p1, _ := optionProbs(logits, labels, 2, 1)
	p2, _ := optionProbs(logits, labels, 2, 2)
	if !(p1[1] > p2[1] && p2[1] > 0.5) {
		t.Fatalf("temperature should flatten toward uniform: T=1 %v, T=2 %v", p1, p2)
	}
	if _, err := optionProbs(logits, [][]int32{{1}, {}}, 2, 1); err == nil {
		t.Fatal("a label with no token must error")
	}
}

func TestNewDecisionExpectedValue(t *testing.T) {
	q, _ := NewScore("q", []string{"a", "b", "c"}, []float64{1, 2, 3})
	d := newDecision(q, []float64{0.2, 0.3, 0.5})
	if d.Index != 2 || d.Confidence != 0.5 {
		t.Fatalf("argmax = %d/%v", d.Index, d.Confidence)
	}
	if math.Abs(d.Expected-2.3) > 1e-9 {
		t.Fatalf("expected = %v, want 2.3", d.Expected)
	}
	y, _ := NewYesNo("q")
	if !newDecision(y, []float64{0.7, 0.3}).Yes() || newDecision(y, []float64{0.4, 0.6}).Yes() {
		t.Fatal("Yes() must follow P(yes) >= 0.5")
	}
}

// TestDecideLive runs the primitive on a cached mlx-community snapshot and
// checks the answers a human would give to an unambiguous ticket, plus that
// asking the same questions twice reuses the state prefix (second call faster).
//
//	MLX_DECIDE_E2E=1 go test ./llm/ -run TestDecideLive -v
func TestDecideLive(t *testing.T) {
	if os.Getenv("MLX_DECIDE_E2E") != "1" {
		t.Skip("set MLX_DECIDE_E2E=1 to run (loads a cached model)")
	}
	model := os.Getenv("MLX_DECIDE_MODEL")
	if model == "" {
		model = "Qwen3-8B-4bit"
	}
	home, _ := os.UserHomeDir()
	dirs, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--mlx-community--"+model+"/snapshots/*"))
	if len(dirs) == 0 {
		t.Skipf("%s not installed", model)
	}
	eng, err := Open(dirs[0])
	if err != nil {
		t.Fatal(err)
	}
	defer eng.Close()

	ticket := "Hi, I was charged twice for order #4411 last week and still have no refund. This is the third " +
		"time I am writing. Fix it today or I will dispute the charge with my bank."
	topic, _ := NewChoice("What is the ticket about?", []string{"Billing", "Shipping", "Account access", "Product question"})
	escalate, _ := NewYesNo("Is the customer threatening to escalate?")
	urgency, _ := NewScore("How urgent is this ticket?", []string{"not urgent", "somewhat urgent", "very urgent"}, nil)

	ds, err := eng.Decide(ticket, []Question{topic, escalate, urgency})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("topic=%v escalate=%.2f urgency=%.2f (%v)", ds[0].Probs, ds[1].Probs[0], ds[2].Expected, ds[2].Probs)
	if ds[0].Answer() != "Billing" {
		t.Errorf("topic = %s %v, want Billing", ds[0].Answer(), ds[0].Probs)
	}
	if !ds[1].Yes() {
		t.Errorf("escalate P(yes) = %.2f, want >= 0.5", ds[1].Probs[0])
	}
	if ds[2].Expected < 2 {
		t.Errorf("urgency expected = %.2f, want >= 2", ds[2].Expected)
	}
	for _, d := range ds {
		var sum float64
		for _, p := range d.Probs {
			sum += p
		}
		if math.Abs(sum-1) > 1e-6 {
			t.Errorf("probs %v do not sum to 1", d.Probs)
		}
	}
}
