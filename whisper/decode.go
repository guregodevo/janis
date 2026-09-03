package whisper

import (
	"math"
	"sort"
	"strings"

	"memdoor/llm/engine"
)

// Greedy decoding of one 30 s chunk with whisper's logit rules: blank and
// non-speech suppression, the timestamp grammar (timestamps come in pairs,
// never decrease, open the sequence, and win when their total probability
// beats every text token). This is whisper's DecodingTask at temperature 0;
// the temperature fallback of its transcribe loop is not ported yet.

// Segment is a timestamped span of the chunk.
type Segment struct {
	Start, End float64
	Tokens     []int
	Text       string
}

// ChunkResult is what DecodeChunk returns for one chunk.
type ChunkResult struct {
	Language string
	Tokens   []int // the sampled tokens (after the prompt), specials included
	Segments []Segment
	Text     string
}

// DecodeOptions steer DecodeChunk. The zero value detects the language and
// allows the first timestamp up to 1 s in, as whisper does.
type DecodeOptions struct {
	Language            string  // ISO code; "" detects
	MaxInitialTimestamp float64 // seconds; 0 means whisper's 1.0
}

var negInf = float32(math.Inf(-1))

// DetectLanguage runs the decoder on <|startoftranscript|> alone and picks
// the most likely language token.
func (d *Decoder) DetectLanguage(b engine.Backend, st *DecoderState) string {
	logits := d.Step(b, st, []int{TokSOT})
	st.Reset(b)
	best, bestV := 0, negInf
	for i := 0; i < numLanguages; i++ {
		if v := logits[tokLangBase+i]; v > bestV {
			best, bestV = i, v
		}
	}
	return whisperLanguages[best]
}

// DecodeChunk transcribes one encoded chunk (the encoder output) greedily.
func DecodeChunk(b engine.Backend, dec *Decoder, tok *Tokenizer, audio engine.Tensor, opt DecodeOptions) *ChunkResult {
	st := dec.NewState(b, audio)
	lang := opt.Language
	if lang == "" {
		lang = dec.DetectLanguage(b, st)
	}
	prompt := []int{TokSOT, LangToken(lang), TokTranscribe}
	sampleBegin := len(prompt)
	tokens := append([]int(nil), prompt...)
	suppress := tok.nonSpeechTokens()
	maxInit := opt.MaxInitialTimestamp
	if maxInit == 0 {
		maxInit = 1.0
	}
	maxInitIdx := int(math.Round(maxInit / 0.02))

	sampleLen := dec.cfg.NCtx / 2
	for i := 0; i < sampleLen; i++ {
		var logits []float32
		if i == 0 {
			logits = dec.Step(b, st, tokens)
		} else {
			logits = dec.Step(b, st, tokens[len(tokens)-1:])
		}
		applyRules(logits, tokens, sampleBegin, suppress, tok, maxInitIdx)
		next := argmax(logits)
		if next == TokEOT {
			break
		}
		tokens = append(tokens, next)
	}
	st.Reset(b)
	if sw, ok := b.(engine.Sweeper); ok {
		sw.Unpin(st.crossK...)
		sw.Unpin(st.crossV...)
		sw.Sweep()
	}
	sampled := tokens[sampleBegin:]
	return &ChunkResult{Language: lang, Tokens: sampled, Segments: segments(tok, sampled), Text: tok.Decode(sampled)}
}

// applyRules is whisper's SuppressBlank + SuppressTokens + ApplyTimestampRules.
func applyRules(logits []float32, tokens []int, sampleBegin int, suppress []int, tok *Tokenizer, maxInitIdx int) {
	atStart := len(tokens) == sampleBegin
	if atStart {
		for _, t := range tok.Encode(" ") {
			logits[t] = negInf
		}
		logits[TokEOT] = negInf
	}
	for _, t := range suppress {
		logits[t] = negInf
	}
	logits[TokNoTimestamps] = negInf

	seq := tokens[sampleBegin:]
	lastTS := len(seq) >= 1 && IsTimestamp(seq[len(seq)-1])
	penultTS := len(seq) < 2 || IsTimestamp(seq[len(seq)-2])
	if lastTS {
		if penultTS { // a text token must follow
			fill(logits, TokTimestampBase, len(logits), negInf)
		} else { // the closing timestamp must follow
			fill(logits, 0, TokEOT, negInf)
		}
	}
	lastTSTok := -1
	for _, t := range seq {
		if IsTimestamp(t) {
			lastTSTok = t
		}
	}
	if lastTSTok >= 0 {
		floor := lastTSTok + 1 // segments have nonzero length
		if lastTS && !penultTS {
			floor = lastTSTok
		}
		fill(logits, TokTimestampBase, floor, negInf)
	}
	if atStart {
		fill(logits, 0, TokTimestampBase, negInf)
		fill(logits, TokTimestampBase+maxInitIdx+1, len(logits), negInf)
	}

	// Timestamps win when their total probability beats every text token.
	lse := logSumExp(logits)
	tsLogProb := logSumExp(logits[TokTimestampBase:]) - lse
	maxText := negInf
	for _, v := range logits[:TokTimestampBase] {
		if v > maxText {
			maxText = v
		}
	}
	if tsLogProb > maxText-lse {
		fill(logits, 0, TokTimestampBase, negInf)
	}
}

func fill(v []float32, from, to int, x float32) {
	for i := from; i < to && i < len(v); i++ {
		v[i] = x
	}
}

func logSumExp(v []float32) float32 {
	m := negInf
	for _, x := range v {
		if x > m {
			m = x
		}
	}
	if m == negInf {
		return negInf
	}
	var s float64
	for _, x := range v {
		s += math.Exp(float64(x - m))
	}
	return m + float32(math.Log(s))
}

func argmax(v []float32) int {
	best, bestV := 0, negInf
	for i, x := range v {
		if x > bestV {
			best, bestV = i, x
		}
	}
	return best
}

// segments splits sampled tokens on timestamp pairs: <|t0|> text <|t1|>.
func segments(tok *Tokenizer, sampled []int) []Segment {
	var out []Segment
	for i := 0; i < len(sampled); {
		if !IsTimestamp(sampled[i]) {
			i++
			continue
		}
		start := Timestamp(sampled[i])
		j := i + 1
		for j < len(sampled) && !IsTimestamp(sampled[j]) {
			j++
		}
		body := sampled[i+1 : j]
		end := float64(ChunkSecs)
		if j < len(sampled) {
			end = Timestamp(sampled[j])
		}
		if len(body) > 0 {
			out = append(out, Segment{Start: start, End: end, Tokens: body, Text: tok.Decode(body)})
		}
		i = j + 1
	}
	return out
}

// nonSpeechTokens is whisper's list of symbol tokens the decoder must not
// produce (they never appear in speech transcripts), plus the task specials.
func (t *Tokenizer) nonSpeechTokens() []int {
	symbols := strings.Split(`"#()*+/:;<=>@[\]^_`+"`"+`{|}~「」『』`, "")
	symbols = append(symbols, strings.Fields(`<< >> <<< >>> -- --- -( -[ (' (" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪`)...)
	misc := "♩♪♫♬♭♮♯"
	set := map[int]bool{}
	if ids := t.Encode(" -"); len(ids) > 0 {
		set[ids[0]] = true
	}
	if ids := t.Encode(" '"); len(ids) > 0 {
		set[ids[0]] = true
	}
	for _, s := range append(symbols, strings.Split(misc, "")...) {
		for _, ids := range [][]int{t.Encode(s), t.Encode(" " + s)} {
			if len(ids) == 1 || (strings.Contains(misc, s) && len(ids) > 0) {
				set[ids[0]] = true
			}
		}
	}
	for _, s := range []int{TokTranscribe, TokTranslate, TokSOT, TokStartOfPrev, TokStartOfLM, TokNoSpeech} {
		set[s] = true
	}
	out := make([]int, 0, len(set))
	for id := range set {
		out = append(out, id)
	}
	sort.Ints(out)
	return out
}
