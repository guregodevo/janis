package qwen35

import (
	"fmt"
	"image"
	"image/draw"
	"math"

	xdraw "golang.org/x/image/draw"

	"memdoor/llm/engine"
	"memdoor/llm/qwen"
	"memdoor/llm/safetensors"
)

// The vision tower of Qwen3.5 (the Qwen3-VL design) on engine.Backend, so
// the one loaded model both thinks and sees (ADR-0006: Python never ships).
//
// An image becomes 16×16 patches in 2×2 merge blocks; each patch's pixels,
// duplicated over a temporal pair, are one 1536-vector. The tower is a
// matmul patch embedding plus bias, a learned 48×48 position grid
// bilinearly interpolated to the image grid, 2-D rotary on (row, col), 27
// pre-LN blocks of full attention over the image's patches and a GELU-tanh
// MLP, then a merger that folds each 2×2 block into one 4096-wide token the
// language model reads in place of <|image_pad|>. Every stage diffs against
// an mlx-vlm oracle in testdata (vision_test.go).

// VisionConfig is Qwen3.5-9B's vision_config.
type VisionConfig struct {
	Depth, Hidden, Heads, Intermediate int
	Patch, Temporal, Merge, Channels   int
	PosGrid                            int // sqrt(num_position_embeddings)
	OutHidden                          int
	LNEps                              float32
}

var Qwen35Vision = VisionConfig{Depth: 27, Hidden: 1152, Heads: 16, Intermediate: 4304,
	Patch: 16, Temporal: 2, Merge: 2, Channels: 3, PosGrid: 48, OutHidden: 4096, LNEps: 1e-6}

// PatchDim is one patch's vector length: channels × temporal × patch².
func (c VisionConfig) PatchDim() int { return c.Channels * c.Temporal * c.Patch * c.Patch }

// Image-size limits of the processor (preprocessor_config.json).
const (
	visionMinPixels = 65536
	visionMaxPixels = 16777216
)

// smartResize is the processor's rule: both sides multiples of patch×merge,
// the pixel count inside [min, max], the aspect ratio kept.
func smartResize(h, w int, factor, minPixels, maxPixels int) (int, int) {
	round := func(v float64) int { return int(math.Round(v/float64(factor))) * factor }
	hb, wb := round(float64(h)), round(float64(w))
	if hb*wb > maxPixels {
		beta := math.Sqrt(float64(h*w) / float64(maxPixels))
		hb = max(factor, int(math.Floor(float64(h)/beta/float64(factor)))*factor)
		wb = max(factor, int(math.Floor(float64(w)/beta/float64(factor)))*factor)
	} else if hb*wb < minPixels {
		beta := math.Sqrt(float64(minPixels) / float64(h*w))
		hb = int(math.Ceil(float64(h)*beta/float64(factor))) * factor
		wb = int(math.Ceil(float64(w)*beta/float64(factor))) * factor
	}
	return hb, wb
}

// Patches is a preprocessed image: N patch vectors in merge-block order and
// the grid (t, h, w) in patches.
type Patches struct {
	Data     []float32 // [N × PatchDim]
	T, H, W  int
	ResizedH int
	ResizedW int
}

// N is the patch count; Tokens is what the language model sees after merging.
func (p Patches) N() int      { return p.T * p.H * p.W }
func (p Patches) Tokens() int { return p.N() / 4 }

// Preprocess resizes (bicubic) to the processor's grid, normalises to
// (x/255 − 0.5)/0.5 and patchifies in the processor's order: merge blocks
// row-major, the 2×2 patches inside each block row-major, and within a
// patch channel, then temporal copy, then row, then column.
func Preprocess(img image.Image, cfg VisionConfig) Patches {
	return PreprocessBudget(img, cfg, visionMaxPixels)
}

// PreprocessBudget is Preprocess with a pixel ceiling below the processor's
// 16.7M: a 960×1707 frame under the default ceiling is 6,420 patches and
// 1,605 language tokens per image — the tower's attention over it and the
// prefill behind it took the gateway down (live 2026-09-03 21:35, the
// `see` check on a finished short). A judge does not need that many.
func PreprocessBudget(img image.Image, cfg VisionConfig, maxPixels int) Patches {
	b := img.Bounds()
	factor := cfg.Patch * cfg.Merge
	if maxPixels <= 0 || maxPixels > visionMaxPixels {
		maxPixels = visionMaxPixels
	}
	rh, rw := smartResize(b.Dy(), b.Dx(), factor, visionMinPixels, maxPixels)
	rgba := image.NewRGBA(image.Rect(0, 0, rw, rh))
	if rh == b.Dy() && rw == b.Dx() {
		draw.Draw(rgba, rgba.Bounds(), img, b.Min, draw.Src)
	} else {
		xdraw.CatmullRom.Scale(rgba, rgba.Bounds(), img, b, xdraw.Src, nil)
	}
	gh, gw := rh/cfg.Patch, rw/cfg.Patch
	pd := cfg.PatchDim()
	out := make([]float32, gh*gw*pd)
	n := 0
	for br := 0; br < gh/cfg.Merge; br++ {
		for bc := 0; bc < gw/cfg.Merge; bc++ {
			for ir := 0; ir < cfg.Merge; ir++ {
				for ic := 0; ic < cfg.Merge; ic++ {
					py0 := (br*cfg.Merge + ir) * cfg.Patch
					px0 := (bc*cfg.Merge + ic) * cfg.Patch
					v := out[n*pd : (n+1)*pd]
					i := 0
					for c := 0; c < cfg.Channels; c++ {
						for t := 0; t < cfg.Temporal; t++ {
							for y := 0; y < cfg.Patch; y++ {
								row := rgba.Pix[(py0+y)*rgba.Stride+px0*4:]
								for x := 0; x < cfg.Patch; x++ {
									v[i] = (float32(row[x*4+c])/255 - 0.5) / 0.5
									i++
								}
							}
						}
					}
					n++
				}
			}
		}
	}
	return Patches{Data: out, T: 1, H: gh, W: gw, ResizedH: rh, ResizedW: rw}
}

type visionLinear struct {
	wT   engine.Tensor // [in, out]
	bias engine.Tensor
}

func loadVisionLinear(b engine.Backend, st *safetensors.File, prefix string) (*visionLinear, error) {
	w, err := qwen.LoadTensor(b, st, prefix+".weight")
	if err != nil {
		return nil, err
	}
	bias, err := qwen.LoadTensor(b, st, prefix+".bias")
	if err != nil {
		return nil, err
	}
	l := &visionLinear{wT: b.Transpose(b.Cast(w, engine.F16), 1, 0), bias: b.Cast(bias, engine.F16)}
	b.Eval(l.wT, l.bias)
	return l, nil
}

func (l *visionLinear) forward(b engine.Backend, x engine.Tensor) engine.Tensor {
	return b.Add(b.MatMul(x, l.wT), l.bias)
}

type visionNorm struct {
	w, bias engine.Tensor
	eps     float32
}

func loadVisionNorm(b engine.Backend, st *safetensors.File, prefix string, eps float32) (*visionNorm, error) {
	w, err := qwen.LoadTensor(b, st, prefix+".weight")
	if err != nil {
		return nil, err
	}
	bias, err := qwen.LoadTensor(b, st, prefix+".bias")
	if err != nil {
		return nil, err
	}
	n := &visionNorm{w: b.Cast(w, engine.F16), bias: b.Cast(bias, engine.F16), eps: eps}
	b.Eval(n.w, n.bias)
	return n, nil
}

func (n *visionNorm) forward(b engine.Backend, x engine.Tensor) engine.Tensor {
	return b.LayerNorm(x, n.w, n.bias, n.eps)
}

type visionBlock struct {
	norm1, norm2 *visionNorm
	qkv, proj    *visionLinear
	fc1, fc2     *visionLinear
}

// VisionTower is the loaded tower.
type VisionTower struct {
	cfg      VisionConfig
	tensors  []engine.Tensor // everything loaded, for Free
	patchW   engine.Tensor   // [PatchDim, Hidden] in the patch vector's order
	patchB   engine.Tensor
	posEmbed engine.Tensor // [PosGrid², Hidden]
	blocks   []*visionBlock
	mergeLN  *visionNorm
	mergeFC1 *visionLinear
	mergeFC2 *visionLinear
}

// LoadVisionTower reads vision_tower.* from the checkpoint.
func LoadVisionTower(b engine.Backend, st *safetensors.File, cfg VisionConfig) (*VisionTower, error) {
	v := &VisionTower{cfg: cfg}
	w, err := qwen.LoadTensor(b, st, "vision_tower.patch_embed.proj.weight") // [out, T, P, P, C]
	if err != nil {
		return nil, err
	}
	shape := w.Shape()
	if len(shape) != 5 || shape[1] != cfg.Temporal || shape[2] != cfg.Patch || shape[4] != cfg.Channels {
		return nil, fmt.Errorf("patch_embed weight: want [out,%d,%d,%d,%d], got %v", cfg.Temporal, cfg.Patch, cfg.Patch, cfg.Channels, shape)
	}
	// The patch vector is ordered (c, t, y, x); reorder the kernel to match:
	// [out, T, P, P, C] → [out, C, T, P, P] → [out, PatchDim] → transpose.
	w = b.Transpose(w, 0, 4, 1, 2, 3)
	w = b.Reshape(w, shape[0], cfg.PatchDim())
	v.patchW = b.Transpose(b.Cast(w, engine.F16), 1, 0)
	if v.patchB, err = qwen.LoadTensor(b, st, "vision_tower.patch_embed.proj.bias"); err != nil {
		return nil, err
	}
	v.patchB = b.Cast(v.patchB, engine.F16)
	if v.posEmbed, err = qwen.LoadTensor(b, st, "vision_tower.pos_embed.weight"); err != nil {
		return nil, err
	}
	v.posEmbed = b.Cast(v.posEmbed, engine.F16)
	b.Eval(v.patchW, v.patchB, v.posEmbed)
	for i := 0; i < cfg.Depth; i++ {
		p := fmt.Sprintf("vision_tower.blocks.%d", i)
		blk := &visionBlock{}
		if blk.norm1, err = loadVisionNorm(b, st, p+".norm1", cfg.LNEps); err != nil {
			return nil, err
		}
		if blk.norm2, err = loadVisionNorm(b, st, p+".norm2", cfg.LNEps); err != nil {
			return nil, err
		}
		if blk.qkv, err = loadVisionLinear(b, st, p+".attn.qkv"); err != nil {
			return nil, err
		}
		if blk.proj, err = loadVisionLinear(b, st, p+".attn.proj"); err != nil {
			return nil, err
		}
		if blk.fc1, err = loadVisionLinear(b, st, p+".mlp.linear_fc1"); err != nil {
			return nil, err
		}
		if blk.fc2, err = loadVisionLinear(b, st, p+".mlp.linear_fc2"); err != nil {
			return nil, err
		}
		v.blocks = append(v.blocks, blk)
	}
	if v.mergeLN, err = loadVisionNorm(b, st, "vision_tower.merger.norm", cfg.LNEps); err != nil {
		return nil, err
	}
	if v.mergeFC1, err = loadVisionLinear(b, st, "vision_tower.merger.linear_fc1"); err != nil {
		return nil, err
	}
	if v.mergeFC2, err = loadVisionLinear(b, st, "vision_tower.merger.linear_fc2"); err != nil {
		return nil, err
	}
	v.tensors = append(v.tensors, v.patchW, v.patchB, v.posEmbed, v.mergeLN.w, v.mergeLN.bias,
		v.mergeFC1.wT, v.mergeFC1.bias, v.mergeFC2.wT, v.mergeFC2.bias)
	for _, blk := range v.blocks {
		v.tensors = append(v.tensors, blk.norm1.w, blk.norm1.bias, blk.norm2.w, blk.norm2.bias,
			blk.qkv.wT, blk.qkv.bias, blk.proj.wT, blk.proj.bias, blk.fc1.wT, blk.fc1.bias, blk.fc2.wT, blk.fc2.bias)
	}
	return v, nil
}

// Free releases the tower's weights (about 0.9 GB) on a sweeping backend;
// the tower must not be used afterwards. Loading again takes ~300 ms.
func (v *VisionTower) Free(b engine.Backend) {
	if sw, ok := b.(engine.Sweeper); ok {
		sw.Unpin(v.tensors...)
		sw.Sweep()
	}
}

// geluTanh is the tower's activation: 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))).
func geluTanh(b engine.Backend, x engine.Tensor) engine.Tensor {
	x3 := b.Mul(b.Mul(x, x), x)
	inner := b.ScalarMul(b.Add(x, b.ScalarMul(x3, 0.044715)), float32(math.Sqrt(2/math.Pi)))
	return b.Mul(b.ScalarMul(x, 0.5), b.ScalarAdd(b.Tanh(inner), 1))
}

// PatchEmbed is the first stage: patches → [N, Hidden] (stage receipt).
func (v *VisionTower) PatchEmbed(b engine.Backend, p Patches) engine.Tensor {
	x := b.Cast(b.FromFloats(p.Data, p.N(), v.cfg.PatchDim()), engine.F16)
	return b.Add(b.MatMul(x, v.patchW), v.patchB)
}

// posEmbedFor interpolates the learned grid to the image grid, in merge-block
// order (the tower's fast_pos_embed_interpolate).
func (v *VisionTower) posEmbedFor(b engine.Backend, p Patches) engine.Tensor {
	g := v.cfg.PosGrid
	lin := func(n int) []float64 {
		out := make([]float64, n)
		for i := range out {
			if n == 1 {
				out[i] = 0
			} else {
				out[i] = float64(g-1) * float64(i) / float64(n-1)
			}
		}
		return out
	}
	hs, ws := lin(p.H), lin(p.W)
	// Four corner lookups with bilinear weights, then permuted to block order.
	n := p.H * p.W
	idx := [4][]int32{}
	wgt := [4][]float32{}
	for k := range idx {
		idx[k] = make([]int32, n)
		wgt[k] = make([]float32, n)
	}
	order := blockOrder(p.H, p.W, v.cfg.Merge) // row-major index → block-order position
	for y := 0; y < p.H; y++ {
		hf := int(hs[y])
		hc := min(hf+1, g-1)
		dh := hs[y] - float64(hf)
		for x := 0; x < p.W; x++ {
			wf := int(ws[x])
			wc := min(wf+1, g-1)
			dw := ws[x] - float64(wf)
			pos := order[y*p.W+x]
			idx[0][pos], wgt[0][pos] = int32(hf*g+wf), float32((1-dh)*(1-dw))
			idx[1][pos], wgt[1][pos] = int32(hf*g+wc), float32((1-dh)*dw)
			idx[2][pos], wgt[2][pos] = int32(hc*g+wf), float32(dh*(1-dw))
			idx[3][pos], wgt[3][pos] = int32(hc*g+wc), float32(dh*dw)
		}
	}
	var sum engine.Tensor
	for k := 0; k < 4; k++ {
		rows := b.TakeAxis(v.posEmbed, b.FromInt32(idx[k], n), 0)           // [n, Hidden]
		term := b.Mul(rows, b.Cast(b.FromFloats(wgt[k], n, 1), engine.F16)) // broadcast weight
		if sum == nil {
			sum = term
		} else {
			sum = b.Add(sum, term)
		}
	}
	if p.T > 1 {
		tiled := sum
		for t := 1; t < p.T; t++ {
			tiled = b.Concat(tiled, sum, 0)
		}
		sum = tiled
	}
	return sum
}

// blockOrder maps a row-major patch index to its position in merge-block
// order: blocks row-major, patches inside a block row-major.
func blockOrder(h, w, merge int) []int {
	out := make([]int, h*w)
	pos := 0
	for br := 0; br < h/merge; br++ {
		for bc := 0; bc < w/merge; bc++ {
			for ir := 0; ir < merge; ir++ {
				for ic := 0; ic < merge; ic++ {
					out[(br*merge+ir)*w+bc*merge+ic] = pos
					pos++
				}
			}
		}
	}
	return out
}

// rotary returns cos and sin [N, headDim] for the 2-D rotary: each patch's
// (row, col) in the grid, the first half of the angles from the row, the
// second from the column, both tiled to the head.
func (v *VisionTower) rotary(b engine.Backend, p Patches) (cos, sin engine.Tensor) {
	headDim := v.cfg.Hidden / v.cfg.Heads
	half := headDim / 2 // 36 angles: 18 from the row, 18 from the column
	nf := half / 2
	inv := make([]float64, nf)
	for i := range inv {
		inv[i] = 1 / math.Pow(10000, float64(2*i)/float64(half))
	}
	n := p.N()
	cs := make([]float32, n*headDim)
	sn := make([]float32, n*headDim)
	order := blockOrder(p.H, p.W, v.cfg.Merge)
	for t := 0; t < p.T; t++ {
		for y := 0; y < p.H; y++ {
			for x := 0; x < p.W; x++ {
				pos := t*p.H*p.W + order[y*p.W+x]
				row := cs[pos*headDim : (pos+1)*headDim]
				srow := sn[pos*headDim : (pos+1)*headDim]
				for i := 0; i < nf; i++ {
					ah, aw := float64(y)*inv[i], float64(x)*inv[i]
					// angles [h(18) | w(18)] tiled twice over the head
					row[i], srow[i] = float32(math.Cos(ah)), float32(math.Sin(ah))
					row[nf+i], srow[nf+i] = float32(math.Cos(aw)), float32(math.Sin(aw))
					row[half+i], srow[half+i] = row[i], srow[i]
					row[half+nf+i], srow[half+nf+i] = row[nf+i], srow[nf+i]
				}
			}
		}
	}
	return b.Cast(b.FromFloats(cs, 1, 1, n, headDim), engine.F16), b.Cast(b.FromFloats(sn, 1, 1, n, headDim), engine.F16)
}

// applyRotary is x·cos + rotate_half(x)·sin over the last axis of [1, H, N, d].
func applyRotary(b engine.Backend, x, cos, sin engine.Tensor) engine.Tensor {
	d := x.Shape()[3]
	x1 := b.Slice(x, 3, 0, d/2)
	x2 := b.Slice(x, 3, d/2, d)
	rot := b.Concat(b.ScalarMul(x2, -1), x1, 3)
	return b.Add(b.Mul(x, cos), b.Mul(rot, sin))
}

// Forward runs the tower: patches → merged image tokens [N/4, OutHidden].
// It evaluates and sweeps per block, so every unpinned tensor the caller
// holds on this backend is freed by the time it returns: pin what must
// survive (earlier images' features, a session's caches).
func (v *VisionTower) Forward(b engine.Backend, p Patches) engine.Tensor {
	cfg := v.cfg
	n := p.N()
	H, dh := cfg.Heads, cfg.Hidden/cfg.Heads
	scale := float32(1 / math.Sqrt(float64(dh)))
	x := b.Add(v.PatchEmbed(b, p), v.posEmbedFor(b, p))
	cos, sin := v.rotary(b, p)
	b.Eval(x, cos, sin)
	sw, _ := b.(engine.Sweeper)
	pin := func(ts ...engine.Tensor) {
		if sw != nil {
			sw.Pin(ts...)
			sw.Sweep()
		}
	}
	pin(x, cos, sin)
	// Each block's output is pinned across the sweep and the previous
	// block's released: pinning every block's x for good leaked the whole
	// tower's activations per image (the same defect as whisper's encoder,
	// found 2026-09-13).
	prev := x
	for _, blk := range v.blocks {
		h := blk.norm1.forward(b, x)
		qkv := b.Reshape(blk.qkv.forward(b, h), n, 3, H, dh) // [N, 3, H, dh]
		split := func(i int) engine.Tensor {
			return b.Transpose(b.Reshape(b.Slice(qkv, 1, i, i+1), 1, n, H, dh), 0, 2, 1, 3) // [1, H, N, dh]
		}
		q := applyRotary(b, split(0), cos, sin)
		k := applyRotary(b, split(1), cos, sin)
		val := split(2)
		o := b.SDPA(q, k, val, scale, false)
		o = b.Reshape(b.Transpose(o, 0, 2, 1, 3), n, cfg.Hidden)
		x = b.Add(x, blk.proj.forward(b, o))
		x = b.Add(x, blk.fc2.forward(b, geluTanh(b, blk.fc1.forward(b, blk.norm2.forward(b, x)))))
		b.Eval(x)
		if sw != nil {
			sw.Pin(x)
			sw.Unpin(prev)
			sw.Sweep()
		}
		prev = x
	}
	m := b.Reshape(v.mergeLN.forward(b, x), n/(cfg.Merge*cfg.Merge), cfg.Hidden*cfg.Merge*cfg.Merge)
	out := v.mergeFC2.forward(b, b.Gelu(v.mergeFC1.forward(b, m)))
	b.Eval(out)
	// The features are the caller's (engine_vision pins them); the last
	// block's x and the rotary tables go with the caller's next sweep.
	if sw != nil {
		sw.Unpin(x, cos, sin)
	}
	return out
}
