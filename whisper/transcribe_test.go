package whisper

import (
	"encoding/json"
	"math"
	"os"
	"strings"
	"testing"
	"time"

	"memdoor/llm/mlxc"
)

// Milestone 5 receipt: the long-form loop on the oracle clip gives
// mlx_whisper's transcript, its segment, and its words.
func TestTranscribeOracleClip(t *testing.T) {
	wav, err := os.ReadFile("testdata/audio_10s.wav")
	if err != nil {
		t.Skip(err)
	}
	raw, err := os.ReadFile("testdata/words_oracle.json")
	if err != nil {
		t.Skip(err)
	}
	var oracle struct {
		Text     string `json:"text"`
		Language string `json:"language"`
		Words    []struct {
			W string  `json:"w"`
			S float64 `json:"s"`
			E float64 `json:"e"`
		} `json:"words"`
		Segments []struct {
			Start, End float64
		} `json:"segments"`
	}
	if err := json.Unmarshal(raw, &oracle); err != nil {
		t.Fatal(err)
	}
	samples, _ := ReadWAV16k(wav)
	b := mlxc.New()
	defer b.Close()
	t0 := time.Now()
	m, err := Load(b, turboWeights(t))
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("model loaded in %s", time.Since(t0).Round(time.Millisecond))
	t0 = time.Now()
	res, err := m.Transcribe(samples, TranscribeOptions{WordTimestamps: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("transcribed %.1f s of audio in %s", float64(len(samples))/SampleRate, time.Since(t0).Round(time.Millisecond))
	if res.Language != oracle.Language || res.Text != oracle.Text {
		t.Errorf("got %s %q\nwant %s %q", res.Language, res.Text, oracle.Language, oracle.Text)
	}
	if len(res.Segments) != len(oracle.Segments) {
		t.Fatalf("%d segments, oracle %d", len(res.Segments), len(oracle.Segments))
	}
	for i, s := range res.Segments {
		if o := oracle.Segments[i]; math.Abs(s.Start-o.Start) > 0.05 || math.Abs(s.End-o.End) > 0.05 {
			t.Errorf("segment %d [%.2f→%.2f], oracle [%.2f→%.2f]", i, s.Start, s.End, o.Start, o.End)
		}
		t.Logf("segment [%.2f→%.2f] temp %.1f logprob %.2f ratio %.2f nospeech %.3f", s.Start, s.End, s.Temperature, s.AvgLogprob, s.CompressionRatio, s.NoSpeechProb)
	}
	var got []Word
	for _, s := range res.Segments {
		got = append(got, s.Words...)
	}
	if len(got) != len(oracle.Words) {
		t.Fatalf("%d words, oracle %d", len(got), len(oracle.Words))
	}
	off := 0
	for i, w := range got {
		if strings.TrimSpace(w.Word) != strings.TrimSpace(oracle.Words[i].W) {
			t.Errorf("word %d %q vs %q", i, w.Word, oracle.Words[i].W)
		}
		if math.Abs(w.Start-oracle.Words[i].S) > 0.04 {
			off++
		}
	}
	if off > len(got)/10 {
		t.Errorf("%d of %d word starts beyond 40 ms", off, len(got))
	}
}
