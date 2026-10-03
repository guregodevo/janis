package whisper

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/guregodevo/janis/safetensors"
)

// Milestone 3 receipt: the Go decoder's greedy transcript of the oracle clip
// is mlx_whisper's, word for word, with English detected.
func TestDecodeMatchesOracle(t *testing.T) {
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
			Tokens     []int
		} `json:"segments"`
	}
	if err := json.Unmarshal(raw, &oracle); err != nil {
		t.Fatal(err)
	}
	samples, err := ReadWAV16k(wav)
	if err != nil {
		t.Fatal(err)
	}
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
	enc, err := LoadEncoder(b, st, TurboEncoder)
	if err != nil {
		t.Fatal(err)
	}
	dec, err := LoadDecoder(b, st, TurboDecoder)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := NewTokenizer()
	if err != nil {
		t.Fatal(err)
	}
	if sw, ok := any(b).(interface{ PinAll() }); ok {
		sw.PinAll()
	}
	t0 := time.Now()
	audio := enc.Forward(b, mel)
	t.Logf("encode in %s", time.Since(t0).Round(time.Millisecond))
	t0 = time.Now()
	ds := dec.NewState(b, audio)
	defer ds.Free(b)
	res := DecodeChunk(b, dec, tok, ds, DecodeOptions{})
	t.Logf("decode in %s: %d tokens, %d segments", time.Since(t0).Round(time.Millisecond), len(res.Tokens), len(res.Segments))
	t.Logf("tokens: %s", tok.DecodeWithSpecial(res.Tokens))
	for _, s := range res.Segments {
		t.Logf("  [%6.2f → %6.2f] %s", s.Start, s.End, s.Text)
	}
	if res.Language != oracle.Language {
		t.Errorf("language %q, want %q", res.Language, oracle.Language)
	}
	if got, want := normalize(res.Text), normalize(oracle.Text); got != want {
		t.Errorf("text:\n got %q\nwant %q", got, want)
	}
	var want []int
	for _, s := range oracle.Segments {
		want = append(want, s.Tokens...)
	}
	if fmt.Sprint(res.Tokens) != fmt.Sprint(want) {
		t.Errorf("tokens differ from mlx_whisper's:\n got %v\nwant %v", res.Tokens, want)
	}
}

var nonWord = regexp.MustCompile(`[^a-z0-9' ]+`)

func normalize(s string) string {
	s = nonWord.ReplaceAllString(strings.ToLower(s), " ")
	return strings.Join(strings.Fields(s), " ")
}
