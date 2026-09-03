package whisper

import (
	"encoding/json"
	"math"
	"os"
	"sort"
	"strings"
	"testing"
	"time"

	"memdoor/llm/mlxc"
)

// TestTranscribeLongForm compares the Go loop with a Python word cache on a
// real recording: WHISPER_LONG=<16k wav>,<words.v2.json>[,<initial prompt>].
// It logs the receipt and writes the Go words beside the wav as
// <wav>.go-words.json; it is skipped without the variable.
func TestTranscribeLongForm(t *testing.T) {
	spec := os.Getenv("WHISPER_LONG")
	if spec == "" {
		t.Skip("WHISPER_LONG not set")
	}
	parts := strings.SplitN(spec, ",", 3)
	wav, err := os.ReadFile(parts[0])
	if err != nil {
		t.Fatal(err)
	}
	var py []struct {
		W string  `json:"w"`
		S float64 `json:"s"`
		E float64 `json:"e"`
	}
	raw, err := os.ReadFile(parts[1])
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &py); err != nil {
		t.Fatal(err)
	}
	prompt := ""
	if len(parts) > 2 {
		prompt = parts[2]
	}
	samples, err := ReadWAV16k(wav)
	if err != nil {
		t.Fatal(err)
	}
	b := mlxc.New()
	defer b.Close()
	m, err := Load(b, turboWeights(t))
	if err != nil {
		t.Fatal(err)
	}
	t0 := time.Now()
	res, err := m.Transcribe(samples, TranscribeOptions{WordTimestamps: true, InitialPrompt: prompt, Progress: func(seek, total int) {
		if seek%3000 < 1000 {
			t.Logf("  %5.1f%% at %s", 100*float64(seek)/float64(total), time.Since(t0).Round(time.Second))
		}
	}})
	if err != nil {
		t.Fatal(err)
	}
	dur := float64(len(samples)) / SampleRate
	el := time.Since(t0)
	t.Logf("transcribed %.0f s of audio in %s (%.0fx realtime); %d segments, language %s", dur, el.Round(time.Second), dur/el.Seconds(), len(res.Segments), res.Language)
	type jw struct {
		W string  `json:"w"`
		S float64 `json:"s"`
		E float64 `json:"e"`
	}
	var got []jw
	for _, s := range res.Segments {
		for _, w := range s.Words {
			got = append(got, jw{strings.TrimSpace(w.Word), w.Start, w.End})
		}
	}
	out, _ := json.Marshal(got)
	os.WriteFile(parts[0]+".go-words.json", out, 0o644)

	// Match words by text within a 1 s window and compare their starts.
	norm := func(s string) string {
		return strings.ToLower(strings.Trim(s, ".,!?;:\"'"))
	}
	var diffs []float64
	j := 0
	matched := 0
	for _, p := range py {
		for j < len(got) && got[j].S < p.S-1.0 {
			j++
		}
		for k := j; k < len(got) && got[k].S <= p.S+1.0; k++ {
			if norm(got[k].W) == norm(p.W) {
				diffs = append(diffs, math.Abs(got[k].S-p.S))
				matched++
				j = k + 1
				break
			}
		}
	}
	sort.Float64s(diffs)
	pct := func(q float64) float64 {
		if len(diffs) == 0 {
			return 0
		}
		return diffs[int(q*float64(len(diffs)-1))]
	}
	t.Logf("words: Go %d, Python %d, matched %d (%.1f%%); start diff median %.3f s, p90 %.3f s, p99 %.3f s", len(got), len(py), matched, 100*float64(matched)/float64(len(py)), pct(0.5), pct(0.9), pct(0.99))
	t.Logf("first Go words: %v", got[:min(8, len(got))])
	t.Logf("text head: %.300s", res.Text)
}
