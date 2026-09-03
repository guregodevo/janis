// Package whisper is memdoor's port of whisper-large-v3-turbo onto the MLX
// engine (ADR-0006: Python never ships). Receipt-first: every stage diffs
// against an oracle dumped from mlx_whisper for a fixed clip.
//
// This file is the front end: 16 kHz mono samples → the log-mel spectrogram
// the encoder consumes, exactly as whisper computes it — 400-point Hann
// window (periodic), hop 160, reflect padding of 200, power spectrum, the
// 128 slaney mel filters shipped with the model, log10 clamped at 1e-10,
// floored at (max − 8), then (x + 4) / 4.
package whisper

import (
	"encoding/binary"
	"fmt"
	"math"

	_ "embed"
)

const (
	SampleRate = 16000
	NFFT       = 400
	Hop        = 160
	NMels      = 128
	ChunkSecs  = 30
	// NFrames is the mel frames per 30 s chunk the encoder expects.
	NFrames = ChunkSecs * SampleRate / Hop // 3000
	nBins   = NFFT/2 + 1                   // 201
)

//go:embed assets/mel_filters_128.f32
var melFilterBytes []byte

// melFilters is [NMels][nBins] float32, row-major, as saved by numpy.
var melFilters = func() [][]float32 {
	if len(melFilterBytes) != NMels*nBins*4 {
		panic(fmt.Sprintf("mel filters: %d bytes, want %d", len(melFilterBytes), NMels*nBins*4))
	}
	f := make([][]float32, NMels)
	for m := range f {
		f[m] = make([]float32, nBins)
		for b := range f[m] {
			off := (m*nBins + b) * 4
			f[m][b] = math.Float32frombits(binary.LittleEndian.Uint32(melFilterBytes[off:]))
		}
	}
	return f
}()

// hann is the periodic Hann window whisper uses (numpy.hanning(401)[:-1]).
var hann = func() []float64 {
	w := make([]float64, NFFT)
	for i := range w {
		w[i] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(NFFT))
	}
	return w
}()

// dft tables: cos/sin for k in [0,nBins), n in [0,NFFT). NFFT=400 is not a
// power of two; a direct real DFT over 201 bins is ~80k MACs per frame,
// ~1 s for a 30 s chunk in pure Go — fine for a first version (the engine's
// FFT can replace it later without changing the contract).
var dftCos, dftSin = func() ([][]float64, [][]float64) {
	c := make([][]float64, nBins)
	s := make([][]float64, nBins)
	for k := 0; k < nBins; k++ {
		c[k] = make([]float64, NFFT)
		s[k] = make([]float64, NFFT)
		for n := 0; n < NFFT; n++ {
			a := 2 * math.Pi * float64(k) * float64(n) / float64(NFFT)
			c[k][n] = math.Cos(a)
			s[k][n] = math.Sin(a)
		}
	}
	return c, s
}()

// PadOrTrim returns exactly ChunkSecs of samples: zero-padded or cut.
func PadOrTrim(samples []float32) []float32 {
	want := ChunkSecs * SampleRate
	out := make([]float32, want)
	copy(out, samples)
	return out
}

// LogMel computes the [NFrames][NMels] log-mel spectrogram of exactly
// ChunkSecs*SampleRate samples (use PadOrTrim first) — whisper's decode
// path, where silence is padded into the audio and lands at the floor.
func LogMel(samples []float32) [][]float32 {
	if len(samples) != ChunkSecs*SampleRate {
		panic("LogMel: pass exactly 30 s of samples (PadOrTrim)")
	}
	return logMel(samples)
}

// LogMelAll is whisper's transcribe path: the spectrogram of the whole
// recording followed by 30 s of silence, so the floor is computed once over
// everything. Content frames are len(frames) - NFrames; feed the encoder
// windows of it with ChunkMel.
func LogMelAll(samples []float32) [][]float32 {
	padded := make([]float32, len(samples)+ChunkSecs*SampleRate)
	copy(padded, samples)
	return logMel(padded)
}

// ChunkMel is one encoder input: frames[seek : seek+size] followed by zero
// rows up to NFrames, as whisper's transcribe pads a window's mel (not its
// audio) — the encoder sees zeros, not the floor, past the content.
func ChunkMel(frames [][]float32, seek, size int) [][]float32 {
	out := make([][]float32, NFrames)
	for i := range out {
		if i < size && seek+i < len(frames) {
			out[i] = frames[seek+i]
		} else {
			out[i] = make([]float32, NMels)
		}
	}
	return out
}

// logMel is the spectrogram of any length of audio: len/Hop frames (the
// last one dropped, as whisper does).
func logMel(samples []float32) [][]float32 {
	nFrames := len(samples) / Hop
	// Reflect-pad 200 samples each side, like torch.stft(center=True).
	const pad = NFFT / 2
	x := make([]float64, len(samples)+2*pad)
	for i := 0; i < pad; i++ {
		x[pad-1-i] = float64(samples[i+1])
		x[pad+len(samples)+i] = float64(samples[len(samples)-2-i])
	}
	for i, s := range samples {
		x[pad+i] = float64(s)
	}
	power := make([][]float64, nFrames)
	frame := make([]float64, NFFT)
	for t := 0; t < nFrames; t++ {
		off := t * Hop
		for n := 0; n < NFFT; n++ {
			frame[n] = x[off+n] * hann[n]
		}
		p := make([]float64, nBins)
		for k := 0; k < nBins; k++ {
			var re, im float64
			ck, sk := dftCos[k], dftSin[k]
			for n := 0; n < NFFT; n++ {
				re += frame[n] * ck[n]
				im -= frame[n] * sk[n]
			}
			p[k] = re*re + im*im
		}
		power[t] = p
	}
	// mel = filters @ power, then log10 with the whisper clamp/floor/scale.
	out := make([][]float32, nFrames)
	maxv := math.Inf(-1)
	for t := 0; t < nFrames; t++ {
		row := make([]float32, NMels)
		for m := 0; m < NMels; m++ {
			var acc float64
			f := melFilters[m]
			for k := 0; k < nBins; k++ {
				acc += float64(f[k]) * power[t][k]
			}
			if acc < 1e-10 {
				acc = 1e-10
			}
			v := math.Log10(acc)
			row[m] = float32(v)
			if v > maxv {
				maxv = v
			}
		}
		out[t] = row
	}
	floor := float32(maxv - 8.0)
	for t := range out {
		for m := range out[t] {
			if out[t][m] < floor {
				out[t][m] = floor
			}
			out[t][m] = (out[t][m] + 4) / 4
		}
	}
	return out
}

// ReadWAV16k decodes a 16-bit PCM mono WAV at 16 kHz into float32 samples
// in [-1, 1]. Anything else → error (ffmpeg produces exactly this format).
func ReadWAV16k(b []byte) ([]float32, error) {
	if len(b) < 12 || string(b[0:4]) != "RIFF" || string(b[8:12]) != "WAVE" {
		return nil, fmt.Errorf("not a RIFF/WAVE file")
	}
	var channels, bits int
	var rate uint32
	var data []byte
	for off := 12; off+8 <= len(b); {
		id := string(b[off : off+4])
		size := int(binary.LittleEndian.Uint32(b[off+4:]))
		body := b[off+8:]
		if size > len(body) {
			size = len(body)
		}
		switch id {
		case "fmt ":
			if size < 16 {
				return nil, fmt.Errorf("short fmt chunk")
			}
			channels = int(binary.LittleEndian.Uint16(body[2:]))
			rate = binary.LittleEndian.Uint32(body[4:])
			bits = int(binary.LittleEndian.Uint16(body[14:]))
		case "data":
			data = body[:size]
		}
		off += 8 + size + size%2
	}
	if channels != 1 || bits != 16 || rate != SampleRate {
		return nil, fmt.Errorf("want 16-bit mono %d Hz, got %d-bit %d-ch %d Hz", SampleRate, bits, channels, rate)
	}
	out := make([]float32, len(data)/2)
	for i := range out {
		out[i] = float32(int16(binary.LittleEndian.Uint16(data[2*i:]))) / 32768
	}
	return out, nil
}
