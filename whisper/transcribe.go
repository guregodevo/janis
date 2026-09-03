package whisper

import (
	"fmt"
	"math"
	"strings"

	"memdoor/llm/engine"
	"memdoor/llm/safetensors"
)

// The long-form loop of whisper's transcribe(): 30 s windows over the
// recording, each decoded with the temperature fallback (greedy first;
// warmer when the result loops or is improbable), segments cut on
// timestamp pairs, the window advanced to the last complete segment,
// previous text carried as the next window's prompt, and word timestamps
// from the alignment pass when asked.

// Model is a loaded whisper: encoder, decoder and tokenizer on one backend.
type Model struct {
	B   engine.Backend
	Enc *Encoder
	Dec *Decoder
	Tok *Tokenizer
}

// Load reads whisper-large-v3-turbo from a safetensors file onto b.
func Load(b engine.Backend, weights string) (*Model, error) {
	st, err := safetensors.Open(weights)
	if err != nil {
		return nil, err
	}
	defer st.Close()
	enc, err := LoadEncoder(b, st, TurboEncoder)
	if err != nil {
		return nil, fmt.Errorf("whisper encoder: %w", err)
	}
	dec, err := LoadDecoder(b, st, TurboDecoder)
	if err != nil {
		return nil, fmt.Errorf("whisper decoder: %w", err)
	}
	tok, err := NewTokenizer()
	if err != nil {
		return nil, err
	}
	if sw, ok := b.(engine.Sweeper); ok {
		sw.PinAll()
	}
	return &Model{B: b, Enc: enc, Dec: dec, Tok: tok}, nil
}

// TranscribeOptions are whisper's transcribe() knobs. Zero values take
// whisper's defaults: language detected, temperatures 0…1.0 in steps of
// 0.2, compression ratio 2.4, log-probability −1, no-speech 0.6, previous
// text conditioning on.
type TranscribeOptions struct {
	Language       string
	InitialPrompt  string // vocabulary or style hint; whisper's initial_prompt
	WordTimestamps bool

	Temperatures              []float64
	CompressionRatioThreshold float64
	LogprobThreshold          float64
	NoSpeechThreshold         float64
	NoConditionOnPrevious     bool
	NoFallback                bool // greedy only, no thresholds

	// HallucinationSilenceThreshold (seconds) skips silence around words
	// that look hallucinated; 0 keeps whisper's default of off.
	HallucinationSilenceThreshold float64

	Progress func(seek, total int) // mel frames done / total, per window
}

// Result is a transcription.
type Result struct {
	Language string
	Text     string
	Segments []Segment
}

const (
	framesPerSecond = SampleRate / Hop                      // 100 mel frames a second
	inputStride     = 2                                     // mel frames per encoder frame
	timePrecision   = float64(inputStride*Hop) / SampleRate // 0.02 s
)

func (o *TranscribeOptions) defaults() {
	if o.Temperatures == nil {
		o.Temperatures = []float64{0, 0.2, 0.4, 0.6, 0.8, 1.0}
	}
	if o.CompressionRatioThreshold == 0 {
		o.CompressionRatioThreshold = 2.4
	}
	if o.LogprobThreshold == 0 {
		o.LogprobThreshold = -1.0
	}
	if o.NoSpeechThreshold == 0 {
		o.NoSpeechThreshold = 0.6
	}
	if o.NoFallback {
		o.Temperatures = []float64{0}
	}
}

// Transcribe runs the long-form loop over 16 kHz mono samples.
func (m *Model) Transcribe(samples []float32, opt TranscribeOptions) (*Result, error) {
	if len(samples) == 0 {
		return nil, fmt.Errorf("whisper: no audio")
	}
	opt.defaults()
	b := m.B
	mel := LogMelAll(samples)
	contentFrames := len(mel) - NFrames
	contentDuration := float64(contentFrames*Hop) / SampleRate

	lang := opt.Language
	var allTokens []int
	if opt.InitialPrompt != "" {
		allTokens = m.Tok.Encode(" " + strings.TrimSpace(opt.InitialPrompt))
	}
	promptResetSince := 0
	lastSpeech := 0.0
	res := &Result{}
	seek := 0
	for seek < contentFrames {
		timeOffset := float64(seek*Hop) / SampleRate
		windowEnd := float64((seek+NFrames)*Hop) / SampleRate
		segmentSize := contentFrames - seek
		if segmentSize > NFrames {
			segmentSize = NFrames
		}
		segmentDuration := float64(segmentSize*Hop) / SampleRate
		audio := m.Enc.Forward(b, ChunkMel(mel, seek, segmentSize))
		st := m.Dec.NewState(b, audio)

		r := m.decodeWithFallback(st, lang, allTokens[promptResetSince:], opt)
		if lang == "" {
			lang = r.Language
			res.Language = lang
		}
		tokens := r.Tokens

		if !opt.NoFallback {
			skip := r.NoSpeechProb > opt.NoSpeechThreshold
			if r.AvgLogprob > opt.LogprobThreshold {
				skip = false // confident enough despite the no-speech signal
			}
			if skip {
				seek += segmentSize
				st.Free(b)
				continue
			}
		}

		previousSeek := seek
		var current []Segment
		newSegment := func(start, end float64, toks []int) Segment {
			var text []int
			for _, t := range toks {
				if t < TokEOT {
					text = append(text, t)
				}
			}
			return Segment{Start: start, End: end, Tokens: toks, Text: m.Tok.Decode(text),
				Temperature: r.Temperature, AvgLogprob: r.AvgLogprob, CompressionRatio: r.CompressionRatio, NoSpeechProb: r.NoSpeechProb}
		}
		n := len(tokens)
		singleEnding := n >= 2 && !IsTimestamp(tokens[n-2]) && IsTimestamp(tokens[n-1])
		var consecutive []int
		for i := 1; i < n; i++ {
			if IsTimestamp(tokens[i-1]) && IsTimestamp(tokens[i]) {
				consecutive = append(consecutive, i)
			}
		}
		if len(consecutive) > 0 {
			slices := consecutive
			if singleEnding {
				slices = append(slices, n)
			}
			last := 0
			for _, cur := range slices {
				sl := tokens[last:cur]
				current = append(current, newSegment(timeOffset+Timestamp(sl[0]), timeOffset+Timestamp(sl[len(sl)-1]), sl))
				last = cur
			}
			if singleEnding {
				seek += segmentSize
			} else {
				seek += (tokens[last-1] - TokTimestampBase) * inputStride
			}
		} else {
			duration := segmentDuration
			lastTS := -1
			for _, t := range tokens {
				if IsTimestamp(t) {
					lastTS = t
				}
			}
			if lastTS > TokTimestampBase {
				duration = Timestamp(lastTS)
			}
			current = append(current, newSegment(timeOffset, timeOffset+duration, tokens))
			seek += segmentSize
		}

		if opt.WordTimestamps {
			AddWordTimestamps(b, m.Dec, m.Tok, st, lang, current, segmentSize, timeOffset, lastSpeech)
			if !singleEnding {
				if end, ok := lastWordEnd(current); ok && end > timeOffset {
					seek = int(math.Round(end * framesPerSecond))
				}
			}
			if thr := opt.HallucinationSilenceThreshold; thr > 0 {
				var skipWindow bool
				current, seek, skipWindow = skipHallucinations(current, seek, previousSeek, segmentSize, timeOffset, windowEnd, segmentDuration, contentDuration, contentFrames, lastSpeech, thr)
				if skipWindow {
					st.Free(b)
					continue
				}
			}
			if end, ok := lastWordEnd(current); ok {
				lastSpeech = end
			}
		}
		st.Free(b)

		res.Segments = append(res.Segments, current...)
		for _, s := range current {
			allTokens = append(allTokens, s.Tokens...)
		}
		if opt.NoConditionOnPrevious || r.Temperature > 0.5 {
			promptResetSince = len(allTokens)
		}
		if opt.Progress != nil {
			done := seek
			if done > contentFrames {
				done = contentFrames
			}
			opt.Progress(done, contentFrames)
		}
	}
	var sb strings.Builder
	for _, s := range res.Segments {
		sb.WriteString(s.Text)
	}
	res.Text = sb.String()
	if res.Language == "" {
		res.Language = lang
	}
	return res, nil
}

// decodeWithFallback is whisper's: try each temperature until the result
// neither loops (compression ratio) nor rambles (average log-probability);
// silence is accepted as is.
func (m *Model) decodeWithFallback(st *DecoderState, lang string, prompt []int, opt TranscribeOptions) *ChunkResult {
	var r *ChunkResult
	for i, t := range opt.Temperatures {
		r = DecodeChunk(m.B, m.Dec, m.Tok, st, DecodeOptions{Language: lang, Prompt: prompt, Temperature: t, Seed: int64(i)})
		if lang == "" {
			lang = r.Language
		}
		if opt.NoFallback {
			return r
		}
		needsFallback := r.CompressionRatio > opt.CompressionRatioThreshold || r.AvgLogprob < opt.LogprobThreshold
		if r.NoSpeechProb > opt.NoSpeechThreshold {
			needsFallback = false
		}
		if !needsFallback {
			break
		}
	}
	return r
}

func lastWordEnd(segs []Segment) (float64, bool) {
	for i := len(segs) - 1; i >= 0; i-- {
		if n := len(segs[i].Words); n > 0 {
			return segs[i].Words[n-1].End, true
		}
	}
	return 0, false
}

// skipHallucinations ports the hallucination_silence_threshold branch of
// whisper's loop: anomalous words are very long, very short or improbable;
// a segment made of them and surrounded by silence is dropped and the
// window moved past it.
func skipHallucinations(current []Segment, seek, previousSeek, segmentSize int, timeOffset, windowEnd, segmentDuration, contentDuration float64, contentFrames int, lastSpeech, thr float64) ([]Segment, int, bool) {
	anomalyScore := func(w Word) float64 {
		score := 0.0
		d := w.End - w.Start
		if w.Probability < 0.15 {
			score++
		}
		if d < 0.133 {
			score += (0.133 - d) * 15
		}
		if d > 2.0 {
			score += d - 2.0
		}
		return score
	}
	isAnomaly := func(s *Segment) bool {
		if s == nil || len(s.Words) == 0 {
			return false
		}
		var words []Word
		for _, w := range s.Words {
			if !strings.Contains(prependPunct+appendPunct, w.Word) {
				words = append(words, w)
			}
		}
		if len(words) > 8 {
			words = words[:8]
		}
		score := 0.0
		for _, w := range words {
			score += anomalyScore(w)
		}
		return score >= 3 || score+0.01 >= float64(len(words))
	}
	nextWords := func(segs []Segment) *Segment {
		for i := range segs {
			if len(segs[i].Words) > 0 {
				return &segs[i]
			}
		}
		return nil
	}
	n := len(current)
	singleEnding := false // the caller only reaches here after word timing; recompute from words
	_ = singleEnding
	if end, ok := lastWordEnd(current); ok && end > timeOffset {
		if windowEnd-end > thr {
			seek = int(math.Round(end * framesPerSecond))
		} else {
			seek = previousSeek + segmentSize
		}
	}
	if first := nextWords(current); first != nil && isAnomaly(first) {
		if gap := first.Start - timeOffset; gap > thr {
			return current, previousSeek + int(math.Round(gap*framesPerSecond)), true
		}
	}
	halLastEnd := lastSpeech
	for si := 0; si < n; si++ {
		seg := &current[si]
		if len(seg.Words) == 0 {
			continue
		}
		if isAnomaly(seg) {
			next := nextWords(current[si+1:])
			halNextStart := timeOffset + segmentDuration
			if next != nil {
				halNextStart = next.Words[0].Start
			}
			silenceBefore := seg.Start-halLastEnd > thr || seg.Start < thr || seg.Start-timeOffset < 2.0
			silenceAfter := halNextStart-seg.End > thr || isAnomaly(next) || windowEnd-seg.End < 2.0
			if silenceBefore && silenceAfter {
				seek = int(math.Round(math.Max(timeOffset+1, seg.Start) * framesPerSecond))
				if contentDuration-seg.End < thr {
					seek = contentFrames
				}
				current = current[:si]
				break
			}
		}
		halLastEnd = seg.End
	}
	return current, seek, false
}
