package whisper

import (
	"math"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/guregodevo/janis/engine"
)

// Word timestamps, whisper's way (timing.py): a second decoder pass over
// the transcript without timestamps, the cross-attention of the alignment
// heads softmaxed over the audio frames, z-normalised over tokens, median
// filtered, averaged over heads, and dynamic time warping over the result.
// Each word starts where the path first reaches its first token.

// Word is a timed word of a segment.
type Word struct {
	Word        string
	Start, End  float64
	Probability float64
	Tokens      []int
}

const (
	tokensPerSecond = SampleRate / Hop / 2 // 50 encoder frames a second
	medfiltWidth    = 7
	prependPunct    = "\"'“¿([{-"
	appendPunct     = "\"'.。,，!！?？:：”)]}、"
	asciiPunct      = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
)

var noSpaceLanguages = map[string]bool{"zh": true, "ja": true, "th": true, "lo": true, "my": true, "yue": true}

// pieceBytes is a token's text, specials rendered as <|name|>.
func (t *Tokenizer) pieceBytes(tok int) []byte {
	if tok >= 0 && tok < len(t.tokens) {
		return t.tokens[tok]
	}
	return []byte(t.DecodeWithSpecial([]int{tok}))
}

// splitToWords ports split_to_word_tokens: subwords end where the decoded
// bytes form valid UTF-8; for languages that use spaces, subwords join into
// words unless they start with a space, are punctuation, or are special.
func (t *Tokenizer) splitToWords(tokens []int, lang string) (words []string, wordTokens [][]int) {
	var subs []string
	var subToks [][]int
	var cur []int
	var buf []byte
	for _, tok := range tokens {
		cur = append(cur, tok)
		buf = append(buf, t.pieceBytes(tok)...)
		if utf8.Valid(buf) {
			subs = append(subs, string(buf))
			subToks = append(subToks, cur)
			cur, buf = nil, nil
		}
	}
	if noSpaceLanguages[lang] {
		return subs, subToks
	}
	for i, sub := range subs {
		special := subToks[i][0] >= TokEOT
		withSpace := strings.HasPrefix(sub, " ")
		punct := strings.Contains(asciiPunct, strings.TrimSpace(sub))
		if special || withSpace || punct || len(words) == 0 {
			words = append(words, sub)
			wordTokens = append(wordTokens, subToks[i])
		} else {
			words[len(words)-1] += sub
			wordTokens[len(wordTokens)-1] = append(wordTokens[len(wordTokens)-1], subToks[i]...)
		}
	}
	return words, wordTokens
}

// medianFilter smooths each row along its last axis with reflect padding.
func medianFilter(rows [][]float64, width int) {
	pad := width / 2
	if len(rows) == 0 || len(rows[0]) <= pad {
		return
	}
	win := make([]float64, width)
	for _, row := range rows {
		n := len(row)
		src := make([]float64, n+2*pad)
		copy(src[pad:], row)
		for i := 0; i < pad; i++ {
			src[pad-1-i] = row[i+1]
			src[pad+n+i] = row[n-2-i]
		}
		for i := 0; i < n; i++ {
			copy(win, src[i:i+width])
			sort.Float64s(win)
			row[i] = win[pad]
		}
	}
}

// dtw finds the monotone path of least cost through x [N][M] (whisper's
// dtw_cpu + backtrace) and returns it as parallel row and column indices.
func dtw(x [][]float64) (rows, cols []int) {
	N, M := len(x), len(x[0])
	inf := math.Inf(1)
	cost := make([][]float64, N+1)
	trace := make([][]int8, N+1)
	for i := range cost {
		cost[i] = make([]float64, M+1)
		trace[i] = make([]int8, M+1)
		for j := range cost[i] {
			cost[i][j] = inf
			trace[i][j] = -1
		}
	}
	cost[0][0] = 0
	for j := 1; j <= M; j++ {
		for i := 1; i <= N; i++ {
			c0, c1, c2 := cost[i-1][j-1], cost[i-1][j], cost[i][j-1]
			var c float64
			var t int8
			switch {
			case c0 < c1 && c0 < c2:
				c, t = c0, 0
			case c1 < c0 && c1 < c2:
				c, t = c1, 1
			default:
				c, t = c2, 2
			}
			cost[i][j] = x[i-1][j-1] + c
			trace[i][j] = t
		}
	}
	for j := range trace[0] {
		trace[0][j] = 2
	}
	for i := range trace {
		trace[i][0] = 1
	}
	i, j := N, M
	for i > 0 || j > 0 {
		rows = append(rows, i-1)
		cols = append(cols, j-1)
		switch trace[i][j] {
		case 0:
			i--
			j--
		case 1:
			i--
		default:
			j--
		}
	}
	for a, z := 0, len(rows)-1; a < z; a, z = a+1, z-1 {
		rows[a], rows[z] = rows[z], rows[a]
		cols[a], cols[z] = cols[z], cols[a]
	}
	return rows, cols
}

// findAlignment ports whisper's find_alignment for one chunk: the words of
// textTokens with start/end seconds relative to the chunk.
func findAlignment(b engine.Backend, dec *Decoder, tok *Tokenizer, st *DecoderState, lang string, textTokens []int, numFrames int) []Word {
	if len(textTokens) == 0 {
		return nil
	}
	const sotLen = 3
	tokens := append([]int{TokSOT, LangToken(lang), TokTranscribe, TokNoTimestamps}, textTokens...)
	tokens = append(tokens, TokEOT)
	n := len(tokens)
	logits, cross := dec.ForwardAlign(b, st, tokens)

	// Probability of each text token at the position that predicted it.
	V := dec.cfg.NVocab
	N := len(textTokens)
	probs := make([]float64, N)
	for i := 0; i < N; i++ {
		row := logits[(sotLen+i)*V : (sotLen+i)*V+TokEOT]
		m := float64(row[0])
		for _, v := range row {
			m = math.Max(m, float64(v))
		}
		var sum float64
		for _, v := range row {
			sum += math.Exp(float64(v) - m)
		}
		probs[i] = math.Exp(float64(row[textTokens[i]])-m) / sum
	}

	// Attention weights [head][token][frame] over the chunk's content frames.
	H := dec.cfg.NHead
	T := st.crossK[0].Shape()[2]
	F := numFrames / 2
	if F > T {
		F = T
	}
	var heads [][][]float64
	for _, c := range cross {
		for h := 0; h < H; h++ {
			w := make([][]float64, n)
			for t := 0; t < n; t++ {
				row := make([]float64, F)
				base := (h*n + t) * T
				m := math.Inf(-1)
				for f := 0; f < F; f++ {
					row[f] = float64(c[base+f])
					m = math.Max(m, row[f])
				}
				var sum float64
				for f := range row {
					row[f] = math.Exp(row[f] - m)
					sum += row[f]
				}
				for f := range row {
					row[f] /= sum
				}
				w[t] = row
			}
			heads = append(heads, w)
		}
	}
	// z-normalise over tokens, per head and frame.
	for _, w := range heads {
		for f := 0; f < F; f++ {
			var mean, sq float64
			for t := 0; t < n; t++ {
				mean += w[t][f]
			}
			mean /= float64(n)
			for t := 0; t < n; t++ {
				d := w[t][f] - mean
				sq += d * d
			}
			std := math.Sqrt(sq / float64(n))
			for t := 0; t < n; t++ {
				w[t][f] = (w[t][f] - mean) / std
			}
		}
		medianFilter(w, medfiltWidth)
	}
	// Average the heads; keep the rows that predict text (drop the sot
	// sequence and the row after the last text token).
	matrix := make([][]float64, n-sotLen-1)
	for t := range matrix {
		row := make([]float64, F)
		for _, w := range heads {
			for f := 0; f < F; f++ {
				row[f] -= w[t+sotLen][f] // negated: dtw minimises
			}
		}
		for f := range row {
			row[f] /= float64(len(heads))
		}
		matrix[t] = row
	}
	textIdx, timeIdx := dtw(matrix)

	words, wordTokens := tok.splitToWords(append(append([]int(nil), textTokens...), TokEOT), lang)
	if len(wordTokens) <= 1 {
		return nil
	}
	boundaries := []int{0}
	for _, wt := range wordTokens[:len(wordTokens)-1] {
		boundaries = append(boundaries, boundaries[len(boundaries)-1]+len(wt))
	}
	var jumpTimes []float64
	for k := range textIdx {
		if k == 0 || textIdx[k] != textIdx[k-1] {
			jumpTimes = append(jumpTimes, float64(timeIdx[k])/tokensPerSecond)
		}
	}
	out := make([]Word, 0, len(boundaries)-1)
	for i := 0; i+1 < len(boundaries); i++ {
		lo, hi := boundaries[i], boundaries[i+1]
		if hi >= len(jumpTimes) {
			break
		}
		var p float64
		for _, v := range probs[lo:hi] {
			p += v
		}
		out = append(out, Word{Word: words[i], Tokens: wordTokens[i], Start: jumpTimes[lo], End: jumpTimes[hi], Probability: p / float64(hi-lo)})
	}
	return out
}

// mergePunctuations glues leading quotes/brackets to the next word and
// trailing punctuation to the previous one (whisper's merge_punctuations).
func mergePunctuations(a []Word) {
	for i, j := len(a)-2, len(a)-1; i >= 0; i-- {
		if strings.HasPrefix(a[i].Word, " ") && strings.Contains(prependPunct, strings.TrimSpace(a[i].Word)) {
			a[j].Word = a[i].Word + a[j].Word
			a[j].Tokens = append(append([]int(nil), a[i].Tokens...), a[j].Tokens...)
			a[i].Word, a[i].Tokens = "", nil
		} else {
			j = i
		}
	}
	for i, j := 0, 1; j < len(a); j++ {
		if !strings.HasSuffix(a[i].Word, " ") && strings.Contains(appendPunct, a[j].Word) {
			a[i].Word += a[j].Word
			a[i].Tokens = append(a[i].Tokens, a[j].Tokens...)
			a[j].Word, a[j].Tokens = "", nil
		} else {
			i = j
		}
	}
}

// AddWordTimestamps fills the Words of a chunk's segments (whisper's
// add_word_timestamps): one alignment pass over all the text, then the
// heuristics that keep words at sentence and segment boundaries from
// swallowing silence. timeOffset is the chunk's start in the recording,
// numFrames its content in mel frames; lastSpeech is the end of the last
// word before this chunk, and the new value is returned.
func AddWordTimestamps(b engine.Backend, dec *Decoder, tok *Tokenizer, st *DecoderState, lang string, segs []Segment, numFrames int, timeOffset, lastSpeech float64) float64 {
	if len(segs) == 0 {
		return lastSpeech
	}
	perSeg := make([][]int, len(segs))
	var text []int
	for i, s := range segs {
		for _, t := range s.Tokens {
			if t < TokEOT {
				perSeg[i] = append(perSeg[i], t)
			}
		}
		text = append(text, perSeg[i]...)
	}
	align := findAlignment(b, dec, tok, st, lang, text, numFrames)

	var durs []float64
	for _, w := range align {
		if d := w.End - w.Start; d != 0 {
			durs = append(durs, d)
		}
	}
	median := 0.0
	if len(durs) > 0 {
		sorted := append([]float64(nil), durs...)
		sort.Float64s(sorted)
		if n := len(sorted); n%2 == 1 {
			median = sorted[n/2]
		} else {
			median = (sorted[n/2-1] + sorted[n/2]) / 2
		}
	}
	median = math.Min(0.7, median)
	maxDur := median * 2
	if len(durs) > 0 {
		const sentenceEnd = ".。!！?？"
		for i := 1; i < len(align); i++ {
			if align[i].End-align[i].Start > maxDur {
				if strings.Contains(sentenceEnd, align[i].Word) && align[i].Word != "" {
					align[i].End = align[i].Start + maxDur
				} else if strings.Contains(sentenceEnd, align[i-1].Word) && align[i-1].Word != "" {
					align[i].Start = align[i].End - maxDur
				}
			}
		}
	}
	mergePunctuations(align)

	wi := 0
	for si := range segs {
		seg := &segs[si]
		var words []Word
		saved := 0
		for wi < len(align) && saved < len(perSeg[si]) {
			w := align[wi]
			if w.Word != "" {
				words = append(words, Word{
					Word: w.Word, Tokens: w.Tokens, Probability: w.Probability,
					Start: round2(timeOffset + w.Start), End: round2(timeOffset + w.End),
				})
			}
			saved += len(w.Tokens)
			wi++
		}
		if len(words) > 0 {
			// The first word after a pause must not be twice the median.
			if words[0].End-lastSpeech > median*4 && (words[0].End-words[0].Start > maxDur ||
				(len(words) > 1 && words[1].End-words[0].Start > maxDur*2)) {
				if len(words) > 1 && words[1].End-words[1].Start > maxDur {
					boundary := math.Max(words[1].End/2, words[1].End-maxDur)
					words[0].End, words[1].Start = boundary, boundary
				}
				words[0].Start = math.Max(0, words[0].End-maxDur)
			}
			// Prefer the segment's own start when the first word is too long.
			if seg.Start < words[0].End && seg.Start-0.5 > words[0].Start {
				words[0].Start = math.Max(0, math.Min(words[0].End-median, seg.Start))
			} else {
				seg.Start = words[0].Start
			}
			last := len(words) - 1
			if seg.End > words[last].Start && seg.End+0.5 < words[last].End {
				words[last].End = math.Max(words[last].Start+median, seg.End)
			} else {
				seg.End = words[last].End
			}
			lastSpeech = seg.End
		}
		seg.Words = words
	}
	return lastSpeech
}

func round2(v float64) float64 { return math.Round(v*100) / 100 }
