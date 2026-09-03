package whisper

import (
	"encoding/binary"
	"math"
	"os"
	"testing"
)

// Milestone 1 receipt: the Go log-mel of the oracle clip matches
// mlx_whisper's to within 1e-3 over the first 500 frames.
func TestLogMelMatchesOracle(t *testing.T) {
	wav, err := os.ReadFile("testdata/audio_10s.wav")
	if err != nil {
		t.Skip(err)
	}
	samples, err := ReadWAV16k(wav)
	if err != nil {
		t.Fatal(err)
	}
	mel := LogMel(PadOrTrim(samples))
	if len(mel) != NFrames || len(mel[0]) != NMels {
		t.Fatalf("shape %dx%d, want %dx%d", len(mel), len(mel[0]), NFrames, NMels)
	}
	oracle, err := os.ReadFile("testdata/mel_oracle_500.f32")
	if err != nil {
		t.Skip(err)
	}
	const frames = 500
	var maxErr, sumErr float64
	for tIdx := 0; tIdx < frames; tIdx++ {
		for m := 0; m < NMels; m++ {
			off := (tIdx*NMels + m) * 4
			want := math.Float32frombits(binary.LittleEndian.Uint32(oracle[off:]))
			d := math.Abs(float64(mel[tIdx][m] - want))
			sumErr += d
			if d > maxErr {
				maxErr = d
			}
		}
	}
	mean := sumErr / float64(frames*NMels)
	t.Logf("log-mel vs oracle: max abs err %.5f, mean %.6f (500 frames)", maxErr, mean)
	if maxErr > 1e-3 {
		t.Fatalf("max abs error %.5f exceeds 1e-3", maxErr)
	}
}
