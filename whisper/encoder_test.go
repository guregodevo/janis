package whisper

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/guregodevo/janis/engine"
	"github.com/guregodevo/janis/safetensors"
)

func turboWeights(t *testing.T) string {
	home, _ := os.UserHomeDir()
	m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo/snapshots/*/weights.safetensors"))
	if len(m) == 0 {
		t.Skip("whisper-large-v3-turbo not downloaded")
	}
	return m[0]
}

// Milestone 2 receipt: the Go encoder's output for the oracle clip matches
// mlx_whisper's — cosine similarity > 0.999 per row over the first 100 rows.
func TestEncoderMatchesOracle(t *testing.T) {
	wav, err := os.ReadFile("testdata/audio_10s.wav")
	if err != nil {
		t.Skip(err)
	}
	oracle, err := os.ReadFile("testdata/enc_oracle_100.f32")
	if err != nil {
		t.Skip(err)
	}
	samples, err := ReadWAV16k(wav)
	if err != nil {
		t.Fatal(err)
	}
	mel := LogMel(PadOrTrim(samples))

	b := newTestBackend(t)
	defer b.Close()
	st, err := safetensors.Open(turboWeights(t))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	t0 := time.Now()
	enc, err := LoadEncoder(b, st, TurboEncoder)
	if err != nil {
		t.Fatal(err)
	}
	if sw, ok := any(b).(interface{ PinAll() }); ok {
		sw.PinAll()
	}
	t.Logf("encoder loaded in %s", time.Since(t0).Round(time.Millisecond))
	t0 = time.Now()
	out := enc.Forward(b, mel)
	got := b.Floats(b.Cast(out, engine.F32))
	t.Logf("encoder forward in %s", time.Since(t0).Round(time.Millisecond))
	if shape := out.Shape(); shape[0] != TurboEncoder.NCtx || shape[1] != TurboEncoder.NState {
		t.Fatalf("shape %v", shape)
	}
	const rows = 100
	d := TurboEncoder.NState
	minCos, maxAbs := 1.0, 0.0
	for r := 0; r < rows; r++ {
		var dot, na, nb float64
		for i := 0; i < d; i++ {
			g := float64(got[r*d+i])
			w := float64(math.Float32frombits(binary.LittleEndian.Uint32(oracle[(r*d+i)*4:])))
			dot += g * w
			na += g * g
			nb += w * w
			if e := math.Abs(g - w); e > maxAbs {
				maxAbs = e
			}
		}
		if c := dot / math.Sqrt(na*nb); c < minCos {
			minCos = c
		}
	}
	t.Logf("encoder vs oracle: min row cosine %.5f, max abs diff %.4f (%d rows)", minCos, maxAbs, rows)
	if minCos < 0.999 {
		t.Fatalf("min row cosine %.5f below 0.999", minCos)
	}
}
