package llm

import (
	"fmt"
	"math"
	"strings"
	"sync"

	"github.com/guregodevo/janis/chat"
	"github.com/guregodevo/janis/qwen"
)

// Decide is the engine's decision primitive: typed questions about one piece
// of state, answered from ONE prefill each with no decoding. The prompt lists
// the options as lettered choices and primes the assistant turn; the model's
// next-token distribution, restricted to the option letters, IS the answer —
// a probability per option, always one of the caller's options, never prose
// to parse. This is what a "System One" / Jev-style call does, on the local
// engine already loaded (see docs/features/DECIDE.md).
//
// Questions about the same state share the session's KV prefix (the state is
// materialized once; each further question prefills only its own suffix), on
// a dedicated session so the agent's and the utility session's prefixes are
// left intact. Hybrid recurrent models (qwen3_5) cannot truncate a prefix, so
// there every question re-prefills the state.

// Kind names a question's answer shape. A value object: parse at the
// boundary with ParseKind, never compare raw strings.
type Kind string

const (
	KindChoice Kind = "choice" // one of N labelled options
	KindYesNo  Kind = "yesno"  // P(yes); options are fixed to Yes / No
	KindScore  Kind = "score"  // ordered rubric levels; Expected is the mean level value
)

// ParseKind validates a kind coming from JSON or a CLI flag.
func ParseKind(s string) (Kind, error) {
	switch k := Kind(strings.ToLower(strings.TrimSpace(s))); k {
	case KindChoice, KindYesNo, KindScore:
		return k, nil
	}
	return "", fmt.Errorf("unknown question kind %q (choice|yesno|score)", s)
}

// maxOptions is bounded by the single-letter labels A..Z that the answer
// token is read from.
const maxOptions = 26

// Question is one typed question. Build it with NewChoice / NewYesNo /
// NewScore, which validate; a zero Question is rejected by Decide.
type Question struct {
	Kind    Kind
	Text    string
	Options []string  // the lettered options, in the order shown to the model
	Values  []float64 // KindScore only: the numeric value of each level
}

// NewChoice asks the model to pick one of 2..26 distinct options.
func NewChoice(text string, options []string) (Question, error) {
	q := Question{Kind: KindChoice, Text: text, Options: options}
	return q, q.validate()
}

// NewYesNo asks a binary question; Decision.Probs[0] is P(yes).
func NewYesNo(text string) (Question, error) {
	q := Question{Kind: KindYesNo, Text: text, Options: []string{"Yes", "No"}}
	return q, q.validate()
}

// NewScore asks for a level on an ordered rubric (low to high). values gives
// each level's number; nil means 1..n. Decision.Expected is the mean value
// under the model's distribution, so scores can be fractional.
func NewScore(text string, levels []string, values []float64) (Question, error) {
	if values == nil {
		values = make([]float64, len(levels))
		for i := range values {
			values[i] = float64(i + 1)
		}
	}
	q := Question{Kind: KindScore, Text: text, Options: levels, Values: values}
	return q, q.validate()
}

func (q Question) validate() error {
	if _, err := ParseKind(string(q.Kind)); err != nil {
		return err
	}
	if strings.TrimSpace(q.Text) == "" {
		return fmt.Errorf("question text is empty")
	}
	if n := len(q.Options); n < 2 || n > maxOptions {
		return fmt.Errorf("question %q needs 2..%d options, got %d", q.Text, maxOptions, n)
	}
	seen := map[string]bool{}
	for _, o := range q.Options {
		if strings.TrimSpace(o) == "" {
			return fmt.Errorf("question %q has an empty option", q.Text)
		}
		if seen[o] {
			return fmt.Errorf("question %q repeats option %q", q.Text, o)
		}
		seen[o] = true
	}
	if q.Kind == KindScore && len(q.Values) != len(q.Options) {
		return fmt.Errorf("question %q: %d values for %d levels", q.Text, len(q.Values), len(q.Options))
	}
	return nil
}

// Decision is the answer to one Question: a probability per option (same
// order), the argmax, and for KindScore the expected level value.
type Decision struct {
	Options    []string
	Probs      []float64
	Index      int
	Confidence float64 // Probs[Index]
	Expected   float64 // KindScore only
}

// Answer is the chosen option's text.
func (d Decision) Answer() string { return d.Options[d.Index] }

// Yes reports the KindYesNo verdict at the 0.5 threshold.
func (d Decision) Yes() bool { return d.Probs[0] >= 0.5 }

// decideSystem tells the model the answer is a single letter; the template's
// primed assistant turn (empty think block on Qwen3) means the very next token
// is that letter, which is where the distribution is read.
const decideSystem = "You are a precise classifier. You will be shown some content, then one question " +
	"about it with lettered options. Reply with the single letter of the best option and nothing else."

// decideTemp divides the option logits before the softmax. Raw next-token
// distributions from instruct models are overconfident (a fitted T of 3-15 on
// public tasks for Qwen3 0.6B-4B); a per-deployment temperature is the cheap
// fix until a calibrated head exists. 1 = raw.
var decideTemp = envFloat32("MEMDOOR_DECIDE_TEMP", 1.0)

func renderQuestion(q Question) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Question: %s\n", q.Text)
	for i, o := range q.Options {
		label := o
		if q.Kind == KindScore {
			label = fmt.Sprintf("%g - %s", q.Values[i], o)
		}
		fmt.Fprintf(&b, "%c. %s\n", 'A'+i, label)
	}
	b.WriteString("Answer with the letter of the best option.")
	return b.String()
}

func decideMessages(state string, q Question) []Message {
	return []Message{
		{Role: "system", Content: decideSystem},
		{Role: "user", Content: "Content:\n" + state + "\n\n" + renderQuestion(q)},
	}
}

// labelTokens resolves, once per engine, the token ids that spell each letter
// A..Z on its own — both the bare form and the space-prefixed form when each
// is a single token — so the read is independent of how the template ends.
func (e *Engine) labelTokens() [][]int32 {
	e.labelOnce.Do(func() {
		e.labels = make([][]int32, maxOptions)
		for i := range e.labels {
			letter := string(rune('A' + i))
			for _, text := range []string{letter, " " + letter} {
				if ids := e.tok.Encode(text); len(ids) == 1 {
					e.labels[i] = append(e.labels[i], ids[0])
				}
			}
		}
	})
	return e.labels
}

// optionProbs restricts the vocabulary distribution to the first n labels,
// merging each label's token variants by log-sum-exp, and returns the softmax
// over options at temperature temp.
func optionProbs(logits []float32, labels [][]int32, n int, temp float32) ([]float64, error) {
	if temp <= 0 {
		temp = 1
	}
	scores := make([]float64, n)
	for i := 0; i < n; i++ {
		if len(labels[i]) == 0 {
			return nil, fmt.Errorf("tokenizer has no single-token form of label %c", 'A'+i)
		}
		lse := math.Inf(-1)
		for _, id := range labels[i] {
			if int(id) >= len(logits) {
				return nil, fmt.Errorf("label token %d outside vocab %d", id, len(logits))
			}
			lse = logAddExp(lse, float64(logits[id]))
		}
		scores[i] = lse / float64(temp)
	}
	return softmax(scores), nil
}

func logAddExp(a, b float64) float64 {
	if math.IsInf(a, -1) {
		return b
	}
	if a < b {
		a, b = b, a
	}
	return a + math.Log1p(math.Exp(b-a))
}

func softmax(z []float64) []float64 {
	m := math.Inf(-1)
	for _, v := range z {
		m = math.Max(m, v)
	}
	out := make([]float64, len(z))
	var sum float64
	for i, v := range z {
		out[i] = math.Exp(v - m)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

func newDecision(q Question, probs []float64) Decision {
	d := Decision{Options: q.Options, Probs: probs}
	for i, p := range probs {
		if p > d.Confidence {
			d.Confidence, d.Index = p, i
		}
	}
	if q.Kind == KindScore {
		for i, p := range probs {
			d.Expected += p * q.Values[i]
		}
	}
	return d
}

// Decide answers every question about state. Questions are answered in
// order on the dedicated decide session, so all but the first reuse the
// state's KV prefix on pure-attention models.
func (e *Engine) Decide(state string, qs []Question) ([]Decision, error) {
	if len(qs) == 0 {
		return nil, fmt.Errorf("no questions")
	}
	for _, q := range qs {
		if err := q.validate(); err != nil {
			return nil, err
		}
	}
	labels := e.labelTokens()
	out := make([]Decision, 0, len(qs))
	mlxComputeMu.Lock()
	defer mlxComputeMu.Unlock()
	for _, q := range qs {
		ids := e.tok.Encode(chat.ApplyTemplate(e.mtype, decideMessages(state, q)))
		var logits []float32
		withGenerateNothingChunking(e.decideSession, func() {
			logits = e.decideSession.Prefill(e.bk, ids, nil)
		})
		if logits == nil {
			return nil, fmt.Errorf("prefill produced no logits")
		}
		probs, err := optionProbs(logits, labels, len(q.Options), decideTemp)
		if err != nil {
			return nil, err
		}
		out = append(out, newDecision(q, probs))
	}
	return out, nil
}

// withGenerateNothingChunking runs fn with the session's prefill chunk widened
// for a call that decodes nothing (KV warmup, Decide): chunk=128 fetches each
// MoE layer's expert union once per chunk (~30% faster prefill), and the
// reason chunking is NOT the general default — it leaves the decode slot
// cache unwarmed — cannot bite when nothing is decoded. Dense models ignore
// overrides above the global chunk size.
func withGenerateNothingChunking(sess *qwen.Session, fn func()) {
	if sess.PrefillChunk == 1 {
		sess.PrefillChunk = 128
		defer func() { sess.PrefillChunk = 1 }()
	}
	fn()
}

// decideState is the per-engine label cache; embedded in Engine.
type decideState struct {
	decideSession *qwen.Session
	labels        [][]int32
	labelOnce     sync.Once
}
