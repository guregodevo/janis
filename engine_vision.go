package llm

import (
	"fmt"
	"image"
	"strings"

	"memdoor/llm/chat"
	"memdoor/llm/engine"
	"memdoor/llm/qwen"
	"memdoor/llm/qwen35"
	"memdoor/llm/safetensors"
)

// Seeing: the loaded model answers a question about images with its own
// vision tower (qwen35/vision.go), in this process, on the same weights the
// brain runs on. The tower loads on first use and stays (about 0.9 GB);
// each call prefills a throwaway cache set so the agent's conversational
// session keeps its prefix.

// How much of a frame the eyes take: about 400 image tokens for one frame
// (a 9:16 short resizes to 480×832; the processor's own ceiling would make
// it 1,605), a shared budget across a judge's candidates, and a floor a
// frame never drops below.
const (
	seeVisionPixels    = 480 * 864
	seeVisionPixelsAll = 1_200_000
	seeVisionPixelsMin = 256 * 256
)

// CanSee reports whether the loaded model has a vision tower.
func (e *Engine) CanSee() bool { return e.q35 != nil }

// See asks one question about the images (all of them in one prompt) and
// returns the model's answer, greedy, at most maxTokens long.
func (e *Engine) See(images []image.Image, question string, maxTokens int) (string, error) {
	if e.q35 == nil {
		return "", fmt.Errorf("the loaded model (%s) cannot see", e.mtype)
	}
	if len(images) == 0 {
		return "", fmt.Errorf("see: no images")
	}
	if maxTokens <= 0 {
		maxTokens = 200
	}
	mlxComputeMu.Lock()
	defer mlxComputeMu.Unlock()
	b := e.bk
	if e.tower == nil {
		st, err := safetensors.OpenModel(e.modelDir)
		if err != nil {
			return "", err
		}
		tower, err := qwen35.LoadVisionTower(b, st, qwen35.Qwen35Vision)
		st.Close()
		if err != nil {
			return "", fmt.Errorf("vision tower: %w", err)
		}
		pinAll(b)
		e.tower = tower
	}

	// The prompt: one <|vision_start|><|image_pad|><|vision_end|> per image,
	// then the question, through the family's chat template (thinking off).
	var content strings.Builder
	for range images {
		content.WriteString("<|vision_start|><|image_pad|><|vision_end|>")
	}
	content.WriteString(question)
	prompt := chat.ApplyTemplate(e.mtype, []chat.Message{{Role: "user", Content: content.String()}})
	raw := e.tok.EncodeSpecial(prompt)

	// Encode the images and expand each single pad to the image's token count.
	// Pixel budget per image: seeVisionPixels alone, shared when several
	// frames sit in one prompt (the thumbnail judge), never under the floor.
	budget := seeVisionPixels
	if n := len(images); n > 1 && seeVisionPixelsAll/n < budget {
		budget = seeVisionPixelsAll / n
	}
	if budget < seeVisionPixelsMin {
		budget = seeVisionPixelsMin
	}
	var feats []engine.Tensor
	var grids []qwen35.Grid
	var ids []int32
	img := 0
	for _, id := range raw {
		if id != qwen35.ImageTokenID || img >= len(images) {
			ids = append(ids, id)
			continue
		}
		p := qwen35.PreprocessBudget(images[img], qwen35.Qwen35Vision, budget)
		img++
		feat := e.tower.Forward(b, p)
		// The tower sweeps per block, which frees every unpinned tensor on
		// the backend — including the features of the images encoded before
		// this one. Pinned here, unpinned when the call ends (live 2026-09-03:
		// the eight-frame thumbnail judge died on "expected a non-empty
		// mlx_array", the gateway with it).
		if sw, ok := b.(engine.Sweeper); ok {
			sw.Pin(feat)
		}
		feats = append(feats, feat)
		grids = append(grids, qwen35.Grid{T: p.T, H: p.H / qwen35.Qwen35Vision.Merge, W: p.W / qwen35.Qwen35Vision.Merge})
		for i := 0; i < p.Tokens(); i++ {
			ids = append(ids, qwen35.ImageTokenID)
		}
	}

	caches := e.q35.NewCaches()
	defer func() {
		if sw, ok := b.(engine.Sweeper); ok {
			for _, c := range caches {
				sw.Unpin(c.Tensors()...)
			}
			sw.Unpin(feats...)
			sw.Sweep()
		}
	}()
	logits, delta := e.q35.PrefillWithImages(b, ids, feats, grids, caches)
	tok := argmax32(b.Floats(b.Cast(logits, engine.F32)))
	offset := len(ids) + delta
	var out []int32
	for len(out) < maxTokens && !e.stops[tok] {
		out = append(out, tok)
		lg := e.q35.Step(b, tok, offset, caches)
		offset++
		if sw, ok := b.(engine.Sweeper); ok {
			var keep []engine.Tensor
			for _, c := range caches {
				keep = append(keep, c.Tensors()...)
			}
			b.Eval(append(keep, lg)...)
			sw.Pin(keep...)
			sw.Pin(lg)
			sw.Sweep()
		}
		tok = argmax32(b.Floats(b.Cast(lg, engine.F32)))
	}
	return strings.TrimSpace(e.tok.Decode(out)), nil
}

func argmax32(v []float32) int32 {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return int32(best)
}

var _ = qwen.SampleParams{} // the decode above is greedy on purpose: a judge, not a writer
