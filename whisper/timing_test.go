package whisper

import (
	"encoding/json"
	"math"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/guregodevo/janis/safetensors"
)

func TestDTWDiagonal(t *testing.T) {
	// A cost matrix cheapest on its diagonal walks the diagonal.
	x := make([][]float64, 4)
	for i := range x {
		x[i] = make([]float64, 4)
		for j := range x[i] {
			x[i][j] = math.Abs(float64(i - j))
		}
	}
	rows, cols := dtw(x)
	for k := range rows {
		if rows[k] != k || cols[k] != k {
			t.Fatalf("path %v %v", rows, cols)
		}
	}
}

func TestSplitToWords(t *testing.T) {
	tok, err := NewTokenizer()
	if err != nil {
		t.Fatal(err)
	}
	ids := tok.Encode(" The death, then, of a woman")
	words, wt := tok.splitToWords(append(ids, TokEOT), "en")
	got := strings.Join(words, "|")
	if got != " The| death|,| then|,| of| a| woman|<|endoftext|>" {
		t.Errorf("words %q", got)
	}
	if len(wt) != len(words) {
		t.Errorf("%d words, %d token groups", len(words), len(wt))
	}
}

// Milestone 4 receipt: the words of the oracle clip are mlx_whisper's, and
// their boundaries agree within two encoder frames (40 ms) for at least nine
// in ten, never more than five frames apart — the residue is F16 attention
// scores feeding a tie-breaking DTW, on both sides.
func TestWordTimestampsMatchOracle(t *testing.T) {
	wav, err := os.ReadFile("testdata/audio_10s.wav")
	if err != nil {
		t.Skip(err)
	}
	raw, err := os.ReadFile("testdata/words_oracle.json")
	if err != nil {
		t.Skip(err)
	}
	var oracle struct {
		Words []struct {
			W string  `json:"w"`
			S float64 `json:"s"`
			E float64 `json:"e"`
		} `json:"words"`
	}
	if err := json.Unmarshal(raw, &oracle); err != nil {
		t.Fatal(err)
	}
	samples, _ := ReadWAV16k(wav)
	all := LogMelAll(samples)
	content := len(all) - NFrames
	mel := ChunkMel(all, 0, content)
	b := newTestBackend(t)
	defer b.Close()
	st, err := safetensors.Open(turboWeights(t))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	enc, _ := LoadEncoder(b, st, TurboEncoder)
	dec, _ := LoadDecoder(b, st, TurboDecoder)
	tok, _ := NewTokenizer()
	if sw, ok := any(b).(interface{ PinAll() }); ok {
		sw.PinAll()
	}
	audio := enc.Forward(b, mel)
	ds := dec.NewState(b, audio)
	defer ds.Free(b)
	res := DecodeChunk(b, dec, tok, ds, DecodeOptions{})
	t0 := time.Now()
	AddWordTimestamps(b, dec, tok, ds, res.Language, res.Segments, content, 0, 0)
	t.Logf("word timestamps in %s", time.Since(t0).Round(time.Millisecond))
	var got []Word
	for _, s := range res.Segments {
		got = append(got, s.Words...)
	}
	if len(got) != len(oracle.Words) {
		t.Fatalf("%d words, oracle has %d:\n%v", len(got), len(oracle.Words), got)
	}
	// Compare boundaries (a word's start; the last word's end), not words:
	// one moved boundary shifts two words.
	worst, off, boundaries := 0.0, 0, len(got)+1
	for i, w := range got {
		o := oracle.Words[i]
		if strings.TrimSpace(w.Word) != strings.TrimSpace(o.W) {
			t.Errorf("word %d: %q vs oracle %q", i, w.Word, o.W)
		}
		d := math.Abs(w.Start - o.S)
		if i == len(got)-1 {
			d = math.Max(d, math.Abs(w.End-o.E))
		}
		worst = math.Max(worst, d)
		if d > 0.04 {
			off++
			t.Logf("word %d: %q starts %.2f, oracle %.2f", i, w.Word, w.Start, o.S)
		}
	}
	t.Logf("%d words, %d of %d boundaries beyond 40 ms, worst %.2f s", len(got), off, boundaries, worst)
	if off > boundaries/10 || worst > 0.1 {
		t.Errorf("%d of %d boundaries beyond 40 ms, worst %.2f s", off, boundaries, worst)
	}
}
