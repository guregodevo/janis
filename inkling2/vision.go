package inkling2

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/guregodevo/janis/engine"
	"github.com/guregodevo/janis/safetensors"
)

// VisionConfig is config.json's vision_config.
type VisionConfig struct {
	Hidden, Layers, Heads int
	Patch, Merge          int // pixels a patch side, patches a merged block side
	Grid                  int // side of the learned position grid
	Eps                   float32
	RopeTheta             float64
	ImageToken            int32
}

// Block is the side, in pixels, of the square one image token covers.
func (v VisionConfig) Block() int { return v.Patch * v.Merge }

// LoadVisionConfig reads the vision tower's config; ok is false when the
// model has none.
func LoadVisionConfig(modelDir string) (cfg VisionConfig, ok bool, err error) {
	raw, err := os.ReadFile(filepath.Join(modelDir, "config.json"))
	if err != nil {
		return cfg, false, fmt.Errorf("read config.json: %w", err)
	}
	var r struct {
		ImageToken *int32 `json:"image_token_id"`
		Vision     *struct {
			Hidden    int      `json:"hidden_size"`
			Layers    int      `json:"num_hidden_layers"`
			Heads     int      `json:"num_attention_heads"`
			Patch     *int     `json:"native_patch_size"`
			Merge     *int     `json:"adapter_merge_size"`
			Grid      *int     `json:"posemb_grid_size"`
			Eps       *float64 `json:"layer_norm_eps"`
			RopeTheta *float64 `json:"rope_theta"`
			Bias      bool     `json:"projector_bias"`
		} `json:"vision_config"`
	}
	if err := json.Unmarshal(raw, &r); err != nil {
		return cfg, false, fmt.Errorf("parse config.json: %w", err)
	}
	if r.Vision == nil {
		return cfg, false, nil
	}
	v := r.Vision
	// The reference's defaults, for the fields a config leaves out.
	cfg = VisionConfig{Hidden: v.Hidden, Layers: v.Layers, Heads: v.Heads,
		Patch: 20, Merge: 2, Grid: 32, Eps: 1e-5, RopeTheta: 10000, ImageToken: 200054}
	if v.Patch != nil {
		cfg.Patch = *v.Patch
	}
	if v.Merge != nil {
		cfg.Merge = *v.Merge
	}
	if v.Grid != nil {
		cfg.Grid = *v.Grid
	}
	if v.Eps != nil {
		cfg.Eps = float32(*v.Eps)
	}
	if v.RopeTheta != nil {
		cfg.RopeTheta = *v.RopeTheta
	}
	if r.ImageToken != nil {
		cfg.ImageToken = *r.ImageToken
	}
	if v.Bias {
		return cfg, false, fmt.Errorf("%s: a vision projector with a bias is not supported", modelDir)
	}
	if cfg.Hidden <= 0 || cfg.Layers <= 0 || cfg.Heads <= 0 || cfg.Hidden%cfg.Heads != 0 || cfg.Hidden/cfg.Heads%4 != 0 {
		return cfg, false, fmt.Errorf("%s: vision_config has no usable hidden size, layers or heads", modelDir)
	}
	return cfg, true, nil
}

// linear is one of the tower's linear layers: floats in the mlx-lm
// conversions, quantized in the MLX-VLM ones.
type linear struct {
	W    projection
	Bias engine.Tensor // [out] or nil
}

func (l linear) forward(c *ctx, x engine.Tensor) engine.Tensor {
	y := l.W.Forward(c, x)
	if l.Bias != nil {
		y = c.Add(y, l.Bias)
	}
	return y
}

type visionLayer struct {
	LNBefore, LNAfter [2]engine.Tensor // weight, bias
	Q, K, V, Out      linear
	FC1, FC2          linear
	Lambda1, Lambda2  engine.Tensor
}

// Vision is the image encoder: a ViT over 20-pixel patches with a 2-D
// rotary position in its attention, whose patch states are merged 2 by 2
// and projected into the text model's embedding space. One image token
// stands for one merged block.
type Vision struct {
	Cfg            VisionConfig
	Patch          projection    // patch pixels -> hidden
	Positions      engine.Tensor // [grid*grid, hidden]
	Class          engine.Tensor // [1, hidden]: class + its position
	PreLN          [2]engine.Tensor
	Layers         []visionLayer
	Merge1, Merge2 linear

	c *ctx
}

// LoadVision loads the tower from the model's checkpoint.
func (m *Model) LoadVision(st *safetensors.File, cfg VisionConfig) (*Vision, error) {
	const p = "model.vision_tower."
	c := m.c
	ld := loader{c: c, st: st, cfg: m.Cfg}
	lin := func(name string, bias bool) linear {
		l := linear{W: ld.linear(name)}
		if bias {
			l.Bias = ld.tensor(name + ".bias")
		}
		return l
	}
	ln := func(name string) [2]engine.Tensor {
		return [2]engine.Tensor{ld.tensor(name + ".weight"), ld.tensor(name + ".bias")}
	}
	v := &Vision{Cfg: cfg, c: c}
	v.Patch = ld.linear(p + "embeddings.patch_embedding")
	v.Positions = ld.tensor(p + "embeddings.position_embedding.weight")
	class, classPos := ld.tensor(p+"embeddings.class_embedding"), ld.tensor(p+"embeddings.cls_position_embedding")
	v.PreLN = ln(p + "pre_layernorm")
	v.Merge1, v.Merge2 = lin(p+"merger.linear_1", false), lin(p+"merger.linear_2", false)
	for i := 0; i < cfg.Layers && ld.err == nil; i++ {
		lp := fmt.Sprintf("%slayers.%d.", p, i)
		v.Layers = append(v.Layers, visionLayer{
			LNBefore: ln(lp + "layernorm_before"), LNAfter: ln(lp + "layernorm_after"),
			Q: lin(lp+"self_attn.q_proj", true), K: lin(lp+"self_attn.k_proj", true),
			V: lin(lp+"self_attn.v_proj", true), Out: lin(lp+"self_attn.out_proj", true),
			FC1: lin(lp+"mlp.fc1", true), FC2: lin(lp+"mlp.fc2", true),
			Lambda1: ld.tensor(lp + "lambda_1"), Lambda2: ld.tensor(lp + "lambda_2"),
		})
	}
	if ld.err != nil {
		return nil, ld.err
	}
	v.Class = c.Reshape(c.Add(class, classPos), 1, cfg.Hidden)
	return v, nil
}

// gelu is the exact GELU, in float32 as the reference computes it.
func (c *ctx) gelu(x engine.Tensor) engine.Tensor {
	f := c.f32(x)
	cdf := c.ScalarMul(c.ScalarAdd(c.Erf(c.ScalarMul(f, float32(1/math.Sqrt2))), 1), 0.5)
	return c.model(c.Mul(f, cdf))
}

// Encode turns one image into its tokens' embeddings: pixels is
// [3, height, width] row-major, already normalized, with both sides a
// multiple of Block. The result is [rows*cols, text hidden], one row per
// block in raster order.
func (v *Vision) Encode(pixels []float32, height, width int) (engine.Tensor, error) {
	c, cfg := v.c, v.Cfg
	p, m := cfg.Patch, cfg.Merge
	if height <= 0 || width <= 0 || height%(p*m) != 0 || width%(p*m) != 0 || len(pixels) != 3*height*width {
		return nil, fmt.Errorf("an image of %dx%d (%d values) is not a multiple of %d", width, height, len(pixels), p*m)
	}
	rows, cols := height/(p*m), width/(p*m)
	n := rows * cols * m * m // patches

	// The merge blocks in raster order, then the patches of each block; a
	// patch is its pixels in raster order, the channels of each together.
	patches := make([]float32, n*p*p*3)
	rowOf, colOf := make([]int, n), make([]int, n)
	at, k := 0, 0
	for r := 0; r < rows; r++ {
		for col := 0; col < cols; col++ {
			for i := 0; i < m; i++ {
				for j := 0; j < m; j++ {
					rowOf[k], colOf[k] = r*m+i, col*m+j
					k++
					for y := 0; y < p; y++ {
						for x := 0; x < p; x++ {
							py, px := (r*m+i)*p+y, (col*m+j)*p+x
							for ch := 0; ch < 3; ch++ {
								patches[at] = pixels[ch*height*width+py*width+px]
								at++
							}
						}
					}
				}
			}
		}
	}
	h := v.Patch.Forward(c, c.model(c.FromFloats(patches, n, p*p*3))) // [n, hidden]

	// The learned positions are a Grid x Grid table, read bilinearly at
	// each patch's place in this image's grid.
	side := cfg.Grid
	taps := func(index []int, size int) ([][2]int32, [][2]float32) {
		ts, ws := make([][2]int32, len(index)), make([][2]float32, len(index))
		for i, ix := range index {
			src := (float64(ix)+0.5)*float64(side)/float64(size) - 0.5
			lo := math.Floor(src)
			for t := 0; t < 2; t++ {
				tap := lo + float64(t)
				ws[i][t] = float32(math.Max(1-math.Abs(src-tap), 0))
				ts[i][t] = int32(math.Min(math.Max(tap, 0), float64(side-1)))
			}
		}
		return ts, ws
	}
	rowTaps, rowW := taps(rowOf, rows*m)
	colTaps, colW := taps(colOf, cols*m)
	index, weight := make([]int32, n*4), make([]float32, n*4)
	for i := 0; i < n; i++ {
		for a := 0; a < 2; a++ {
			for b := 0; b < 2; b++ {
				index[i*4+a*2+b] = rowTaps[i][a]*int32(side) + colTaps[i][b]
				weight[i*4+a*2+b] = rowW[i][a] * colW[i][b]
			}
		}
	}
	pos := c.f32(c.Reshape(c.TakeAxis(v.Positions, c.FromInt32(index, n*4), 0), n, 4, cfg.Hidden))
	pos = c.Mul(pos, c.FromFloats(weight, n, 4, 1))
	pos = c.ScalarMul(c.Mean(pos, 1), 4)
	h = c.Concat(v.Class, c.Add(h, c.model(pos)), 0) // [1+n, hidden]
	h = c.LayerNorm(h, v.PreLN[0], v.PreLN[1], cfg.Eps)

	// The rotary angles: the class token at (0, 0), patch (r, c) at
	// (r+1, c+1); the column's frequencies, then the row's, each twice.
	N, heads := n+1, cfg.Heads
	hd := cfg.Hidden / heads
	half := hd / 2
	cos, sin := make([]float32, N*hd), make([]float32, N*hd)
	for t := 0; t < N; t++ {
		row, col := 0.0, 0.0
		if t > 0 {
			row, col = float64(rowOf[t-1]+1), float64(colOf[t-1]+1)
		}
		for f := 0; f < half/2; f++ {
			inv := 1 / math.Pow(cfg.RopeTheta, float64(2*f)/float64(half))
			for rep := 0; rep < 2; rep++ {
				w, hgt := t*hd+2*f+rep, t*hd+half+2*f+rep
				cos[w], sin[w] = float32(math.Cos(col*inv)), float32(math.Sin(col*inv))
				cos[hgt], sin[hgt] = float32(math.Cos(row*inv)), float32(math.Sin(row*inv))
			}
		}
	}
	cosT, sinT := c.FromFloats(cos, N, 1, hd), c.FromFloats(sin, N, 1, hd)
	rotate := func(x engine.Tensor) engine.Tensor {
		s := c.f32(c.Reshape(x, N, heads, hd))
		pairs := c.Reshape(s, N, heads, hd/2, 2)
		turned := c.Concat(c.ScalarMul(c.Slice(pairs, 3, 1, 2), -1), c.Slice(pairs, 3, 0, 1), 3)
		out := c.Add(c.Mul(s, cosT), c.Mul(c.Reshape(turned, N, heads, hd), sinT))
		return c.Reshape(c.Transpose(c.model(out), 1, 0, 2), 1, heads, N, hd)
	}
	scale := float32(1 / math.Sqrt(float64(hd)))
	for _, l := range v.Layers {
		x := c.LayerNorm(h, l.LNBefore[0], l.LNBefore[1], cfg.Eps)
		q, k := rotate(l.Q.forward(c, x)), rotate(l.K.forward(c, x))
		val := c.Reshape(c.Transpose(c.Reshape(l.V.forward(c, x), N, heads, hd), 1, 0, 2), 1, heads, N, hd)
		a := c.SDPA(q, k, val, scale, false)
		a = c.Reshape(c.Transpose(c.Reshape(a, heads, N, hd), 1, 0, 2), N, cfg.Hidden)
		h = c.Add(h, c.Mul(l.Out.forward(c, a), l.Lambda1))
		x = c.LayerNorm(h, l.LNAfter[0], l.LNAfter[1], cfg.Eps)
		h = c.Add(h, c.Mul(l.FC2.forward(c, c.gelu(l.FC1.forward(c, x))), l.Lambda2))
	}
	blocks := c.Reshape(c.Slice(h, 0, 1, N), rows*cols, m*m*cfg.Hidden)
	return v.Merge2.forward(c, c.gelu(v.Merge1.forward(c, blocks))), nil
}
