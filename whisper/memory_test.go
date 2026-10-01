//go:build darwin && arm64 && cgo

package whisper

import (
	"os"
	"runtime"
	"testing"

	"memdoor/llm/mlxc"
)

// The test below reads the MLX allocator's active/cache numbers
// (mlxc.MemoryStatsMB); on the CPU backend there is no GPU memory to
// account for, so it runs on Apple Silicon only.

// A transcript must not leave anything behind on the GPU: the encoder
// pinned every block's output and never released it, and Close freed
// only the stream — ~300 MB per minute of audio, kept until the process
// died (three silent gateway deaths on 2026-09-13).
func TestTranscribeLeavesNoMemoryBehind(t *testing.T) {
	wav, err := os.ReadFile("testdata/audio_10s.wav")
	if err != nil {
		t.Skip(err)
	}
	samples, _ := ReadWAV16k(wav)
	b := mlxc.New()
	m, err := Load(b, turboWeights(t))
	if err != nil {
		t.Fatal(err)
	}
	loaded, _, _ := mlxc.MemoryStatsMB()
	runtime.GC()
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	if live := float64(ms.HeapInuse) / (1 << 20); live > 200 {
		t.Fatalf("the loaded model keeps %.0f MB of Go heap: the weights live on the GPU, a host copy is duplication", live)
	}
	var after [2]float64
	for i := range after {
		if _, err := m.Transcribe(samples, TranscribeOptions{WordTimestamps: true}); err != nil {
			t.Fatal(err)
		}
		after[i], _, _ = mlxc.MemoryStatsMB()
	}
	t.Logf("active MB: loaded %.0f, after pass 1 %.0f, after pass 2 %.0f", loaded, after[0], after[1])
	if after[0] > loaded+1 || after[1] > after[0]+1 {
		t.Fatalf("a transcript left memory behind: loaded %.0f MB, after passes %.0f / %.0f MB", loaded, after[0], after[1])
	}
	b.Close()
	if active, cache, _ := mlxc.MemoryStatsMB(); active > 1 || cache > 1 {
		t.Fatalf("Close left %.0f MB active and %.0f MB cached", active, cache)
	}
}
