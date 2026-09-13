package whisper

import (
	"bytes"
	"compress/zlib"
	"math"
	"math/rand"
	"sort"
	"strings"

	"memdoor/llm/engine"
)

// Greedy decoding of one 30 s chunk with whisper's logit rules: blank and
// non-speech suppression, the timestamp grammar (timestamps come in pairs,
// never decrease, open the sequence, and win when their total probability
// beats every text token). This is whisper's DecodingTask at temperature 0;
// the temperature fallback of its transcribe loop is not ported yet.

// Segment is a timestamped span of the recording.
type Segment struct {
	Start, End float64
	Tokens     []int // the segment's tokens, timestamps included
	Text       string
	Words      []Word // filled by AddWordTimestamps

	// Whisper's per-window quality signals, copied to each of its segments.
	Temperature, AvgLogprob, CompressionRatio, NoSpeechProb float64
}

// ChunkResult is what DecodeChunk returns for one chunk.
type ChunkResult struct {
	Language string
	Tokens   []int // the sampled tokens (after the prompt), specials included
	Segments []Segment
	Text     string

	Temperature      float64
	AvgLogprob       float64 // mean log-probability of the sampled tokens (EOT included)
	CompressionRatio float64 // zlib ratio of the text: high means it loops
	NoSpeechProb     float64 // <|nospeech|> at the start-of-transcript position
}

// DecodeOptions steer DecodeChunk. The zero value detects the language,
// decodes greedily and allows the first timestamp up to 1 s in, as whisper.
type DecodeOptions struct {
	Language            string  // ISO code; "" detects
	Translate           bool    // whisper's translate task: output in English
	Prompt              []int   // previous text (and any initial prompt); the last NCtx/2-1 are used
	Temperature         float64 // 0 is greedy
	MaxInitialTimestamp float64 // seconds; 0 means whisper's 1.0
	Seed                int64   // for Temperature > 0
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

// DecodeChunk transcribes one chunk. The caller owns st (from
// Decoder.NewState on the encoder output) so the alignment pass can reuse it.
func DecodeChunk(b engine.Backend, dec *Decoder, tok *Tokenizer, st *DecoderState, opt DecodeOptions) *ChunkResult {
	st.Reset(b)
	lang := opt.Language
	if lang == "" {
		lang = dec.DetectLanguage(b, st)
	}
	var initial []int
	if len(opt.Prompt) > 0 {
		p := opt.Prompt
		if max := dec.cfg.NCtx/2 - 1; len(p) > max {
			p = p[len(p)-max:]
		}
		initial = append([]int{TokStartOfPrev}, p...)
	}
	sotIndex := len(initial)
	task := TokTranscribe
	if opt.Translate {
		task = TokTranslate
	}
	initial = append(initial, TokSOT, LangToken(lang), task)
	sampleBegin := len(initial)
	tokens := append([]int(nil), initial...)
	suppress := tok.nonSpeechTokens()
	maxInit := opt.MaxInitialTimestamp
	if maxInit == 0 {
		maxInit = 1.0
	}
	maxInitIdx := int(math.Round(maxInit / 0.02))
	rng := rand.New(rand.NewSource(opt.Seed))

	res := &ChunkResult{Language: lang, Temperature: opt.Temperature}
	var sumLogprob float64
	sampleLen := dec.cfg.NCtx / 2
	for i := 0; i < sampleLen; i++ {
		if len(tokens) >= dec.cfg.NCtx { // a long prompt plus a long sample: whisper stops at n_ctx
			break
		}
		var logits []float32
		if i == 0 {
			rows := dec.StepRows(b, st, tokens, []int{sotIndex, len(tokens) - 1})
			res.NoSpeechProb = softmaxAt(rows[0], TokNoSpeech)
			logits = rows[1]
		} else {
			logits = dec.Step(b, st, tokens[len(tokens)-1:])
		}
		applyRules(logits, tokens, sampleBegin, suppress, tok, maxInitIdx)
		var next int
		if opt.Temperature > 0 {
			next = sample(logits, opt.Temperature, rng)
		} else {
			next = argmax(logits)
		}
		sumLogprob += float64(logits[next] - logSumExp(logits))
		if next == TokEOT {
			break
		}
		tokens = append(tokens, next)
	}
	st.Reset(b)
	sampled := tokens[sampleBegin:]
	res.Tokens = sampled
	res.Text = tok.Decode(sampled)
	res.Segments = segments(tok, sampled)
	res.AvgLogprob = sumLogprob / float64(len(sampled)+1)
	res.CompressionRatio = compressionRatio(res.Text)
	return res
}

// softmaxAt is the probability of one token under the logits.
func softmaxAt(logits []float32, tok int) float64 {
	return math.Exp(float64(logits[tok] - logSumExp(logits)))
}

// sample draws from softmax(logits / temperature).
func sample(logits []float32, temperature float64, rng *rand.Rand) int {
	m := negInf
	for _, v := range logits {
		if v > m {
			m = v
		}
	}
	probs := make([]float64, len(logits))
	var sum float64
	for i, v := range logits {
		if v == negInf {
			continue
		}
		probs[i] = math.Exp(float64(v-m) / temperature)
		sum += probs[i]
	}
	r := rng.Float64() * sum
	for i, p := range probs {
		r -= p
		if r <= 0 && p > 0 {
			return i
		}
	}
	return argmax(logits)
}

// compressionRatio is whisper's loop detector: text bytes over their zlib size.
func compressionRatio(text string) float64 {
	var buf bytes.Buffer
	w := zlib.NewWriter(&buf)
	w.Write([]byte(text))
	w.Close()
	if buf.Len() == 0 {
		return 0
	}
	return float64(len(text)) / float64(buf.Len())
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
		raw := sampled[i:j]
		if j < len(sampled) {
			end = Timestamp(sampled[j])
			raw = sampled[i : j+1]
		}
		if len(body) > 0 {
			out = append(out, Segment{Start: start, End: end, Tokens: raw, Text: tok.Decode(body)})
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
