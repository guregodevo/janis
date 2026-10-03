package whisper

import (
	"fmt"
	"math"

	"github.com/guregodevo/janis/engine"
	"github.com/guregodevo/janis/qwen"
	"github.com/guregodevo/janis/safetensors"
)

// The audio encoder of whisper-large-v3-turbo on engine.Backend: two
// convolutions (kernel 3; the second at stride 2) with GELU, sinusoidal
// positions, 32 pre-LN transformer blocks, a final LayerNorm. Weights stay
// F16 as shipped and the maths runs in F16, as the oracle did.
//
// Conv1d is not an engine op; it is expressed as three shifted matmuls (one
// per tap) summed, and stride 2 as a gather of the even frames — exact, and
// nothing new to port to each backend.

// EncoderConfig is whisper-large-v3-turbo's audio side.
type EncoderConfig struct {
	NMels, NState, NHead, NLayer, NCtx int
	LNEps                              float32
}

var TurboEncoder = EncoderConfig{NMels: 128, NState: 1280, NHead: 20, NLayer: 32, NCtx: 1500, LNEps: 1e-5}

type linear struct {
	wT   engine.Tensor // [in, out] — transposed once at load
	bias engine.Tensor // nil when the layer has none (whisper's key)
}

func loadLinear(b engine.Backend, st *safetensors.File, prefix string, withBias bool) (*linear, error) {
	w, err := qwen.LoadTensor(b, st, prefix+".weight")
	if err != nil {
		return nil, err
	}
	l := &linear{wT: b.Transpose(w, 1, 0)}
	b.Eval(l.wT)
	freeTensor(b, w)
	if withBias {
		if l.bias, err = qwen.LoadTensor(b, st, prefix+".bias"); err != nil {
			return nil, err
		}
	}
	return l, nil
}

func (l *linear) forward(b engine.Backend, x engine.Tensor) engine.Tensor {
	y := b.MatMul(x, l.wT)
	if l.bias != nil {
		y = b.Add(y, l.bias)
	}
	return y
}

type layerNorm struct {
	w, bias engine.Tensor
	eps     float32
}

func loadLayerNorm(b engine.Backend, st *safetensors.File, prefix string, eps float32) (*layerNorm, error) {
	w, err := qwen.LoadTensor(b, st, prefix+".weight")
	if err != nil {
		return nil, err
	}
	bias, err := qwen.LoadTensor(b, st, prefix+".bias")
	if err != nil {
		return nil, err
	}
	return &layerNorm{w: w, bias: bias, eps: eps}, nil
}

func (ln *layerNorm) forward(b engine.Backend, x engine.Tensor) engine.Tensor {
	return b.LayerNorm(x, ln.w, ln.bias, ln.eps)
}

// conv3 is a kernel-3, padding-1 convolution held as three [in, out] tap
// matrices plus bias. Weight in the checkpoint is [out, 3, in].
type conv3 struct {
	taps [3]engine.Tensor
	bias engine.Tensor
}

func loadConv3(b engine.Backend, st *safetensors.File, prefix string) (*conv3, error) {
	w, err := qwen.LoadTensor(b, st, prefix+".weight")
	if err != nil {
		return nil, err
	}
	shape := w.Shape() // [out, 3, in]
	if len(shape) != 3 || shape[1] != 3 {
		return nil, fmt.Errorf("%s: want [out,3,in], got %v", prefix, shape)
	}
	c := &conv3{}
	for k := 0; k < 3; k++ {
		tap := b.Reshape(b.Slice(w, 1, k, k+1), shape[0], shape[2]) // [out, in]
		c.taps[k] = b.Transpose(tap, 1, 0)                          // [in, out]
		b.Eval(c.taps[k])
	}
	freeTensor(b, w)
	if c.bias, err = qwen.LoadTensor(b, st, prefix+".bias"); err != nil {
		return nil, err
	}
	return c, nil
}

// forward applies the convolution to x [L, in] with padding 1 and the given
// stride, returning [ceil(L/stride), out].
func (c *conv3) forward(b engine.Backend, x engine.Tensor, stride int) engine.Tensor {
	shape := x.Shape()
	L, in := shape[0], shape[1]
	zero := b.Cast(b.FromFloats(make([]float32, in), 1, in), engine.F16)
	xp := b.Concat(b.Concat(zero, x, 0), zero, 0) // [L+2, in]
	outL := (L + stride - 1) / stride
	var idx engine.Tensor
	if stride > 1 {
		ids := make([]int32, outL)
		for i := range ids {
			ids[i] = int32(i * stride)
		}
		idx = b.FromInt32(ids, outL)
	}
	var y engine.Tensor
	for k := 0; k < 3; k++ {
		xk := b.Slice(xp, 0, k, k+L) // frame t sees xp[t+k] = x[t+k-1]
		if idx != nil {
			xk = b.TakeAxis(xk, idx, 0)
		}
		t := b.MatMul(xk, c.taps[k])
		if y == nil {
			y = t
		} else {
			y = b.Add(y, t)
		}
	}
	return b.Add(y, c.bias)
}

type encoderBlock struct {
	attnLN, mlpLN *layerNorm
	q, k, v, out  *linear
	mlp1, mlp2    *linear
}

type Encoder struct {
	cfg    EncoderConfig
	conv1  *conv3
	conv2  *conv3
	pos    engine.Tensor // [NCtx, NState] sinusoids
	blks   []*encoderBlock
	lnPost *layerNorm
}

// sinusoids reproduces whisper's fixed positional table.
func sinusoids(length, channels int) []float32 {
	half := channels / 2
	inc := math.Log(10000) / float64(half-1)
	out := make([]float32, length*channels)
	for p := 0; p < length; p++ {
		for i := 0; i < half; i++ {
			t := float64(p) * math.Exp(-float64(i)*inc)
			out[p*channels+i] = float32(math.Sin(t))
			out[p*channels+half+i] = float32(math.Cos(t))
		}
	}
	return out
}

// LoadEncoder reads the encoder weights from the model's safetensors.
func LoadEncoder(b engine.Backend, st *safetensors.File, cfg EncoderConfig) (*Encoder, error) {
	e := &Encoder{cfg: cfg}
	var err error
	if e.conv1, err = loadConv3(b, st, "encoder.conv1"); err != nil {
		return nil, err
	}
	if e.conv2, err = loadConv3(b, st, "encoder.conv2"); err != nil {
		return nil, err
	}
	e.pos = b.Cast(b.FromFloats(sinusoids(cfg.NCtx, cfg.NState), cfg.NCtx, cfg.NState), engine.F16)
	b.Eval(e.pos)
	for i := 0; i < cfg.NLayer; i++ {
		p := fmt.Sprintf("encoder.blocks.%d", i)
		blk := &encoderBlock{}
		if blk.attnLN, err = loadLayerNorm(b, st, p+".attn_ln", cfg.LNEps); err != nil {
			return nil, err
		}
		if blk.mlpLN, err = loadLayerNorm(b, st, p+".mlp_ln", cfg.LNEps); err != nil {
			return nil, err
		}
		if blk.q, err = loadLinear(b, st, p+".attn.query", true); err != nil {
			return nil, err
		}
		if blk.k, err = loadLinear(b, st, p+".attn.key", false); err != nil {
			return nil, err
		}
		if blk.v, err = loadLinear(b, st, p+".attn.value", true); err != nil {
			return nil, err
		}
		if blk.out, err = loadLinear(b, st, p+".attn.out", true); err != nil {
			return nil, err
		}
		if blk.mlp1, err = loadLinear(b, st, p+".mlp1", true); err != nil {
			return nil, err
		}
		if blk.mlp2, err = loadLinear(b, st, p+".mlp2", true); err != nil {
			return nil, err
		}
		e.blks = append(e.blks, blk)
	}
	if e.lnPost, err = loadLayerNorm(b, st, "encoder.ln_post", cfg.LNEps); err != nil {
		return nil, err
	}
	return e, nil
}

// attention is whisper's multi-head attention over x [L, NState]: no mask
// here (the encoder sees the whole chunk).
func (blk *encoderBlock) attention(b engine.Backend, x engine.Tensor, cfg EncoderConfig) engine.Tensor {
	L := x.Shape()[0]
	dh := cfg.NState / cfg.NHead
	split := func(t engine.Tensor) engine.Tensor {
		return b.Transpose(b.Reshape(t, 1, L, cfg.NHead, dh), 0, 2, 1, 3) // [1, H, L, dh]
	}
	q := split(blk.q.forward(b, x))
	k := split(blk.k.forward(b, x))
	v := split(blk.v.forward(b, x))
	o := b.SDPA(q, k, v, float32(1/math.Sqrt(float64(dh))), false)
	o = b.Reshape(b.Transpose(o, 0, 2, 1, 3), L, cfg.NState)
	return blk.out.forward(b, o)
}

// Forward encodes one 30 s chunk: mel [NFrames, NMels] → [NCtx, NState].
func (e *Encoder) Forward(b engine.Backend, mel [][]float32) engine.Tensor {
	flat := make([]float32, 0, len(mel)*e.cfg.NMels)
	for _, row := range mel {
		flat = append(flat, row...)
	}
	x := b.Cast(b.FromFloats(flat, len(mel), e.cfg.NMels), engine.F16)
	x = b.Gelu(e.conv1.forward(b, x, 1))
	x = b.Gelu(e.conv2.forward(b, x, 2))
	x = b.Add(x, e.pos)
	sw, _ := b.(engine.Sweeper)
	var prev engine.Tensor
	for _, blk := range e.blks {
		x = b.Add(x, blk.attention(b, blk.attnLN.forward(b, x), e.cfg))
		h := blk.mlp2.forward(b, b.Gelu(blk.mlp1.forward(b, blk.mlpLN.forward(b, x))))
		x = b.Add(x, h)
		// Evaluate per block and release the block's intermediates: the
		// backend tracks every tensor it creates until a Sweep, and the cost
		// of a block grew from 140 ms to over a second as that list grew
		// (profiled 2026-09-03). Pin x, sweep the rest — the decode loop's rule.
		// The PREVIOUS block's x is unpinned first: pinning every block's
		// output and never releasing it leaked 32 × [1500, 1280] f16 per
		// 30 s chunk — ~300 MB per minute of audio, kept until the process
		// died (measured 2026-09-13: three silent gateway deaths that day).
		b.Eval(x)
		if sw != nil {
			sw.Pin(x)
			if prev != nil {
				sw.Unpin(prev)
			}
			sw.Sweep()
		}
		prev = x
	}
	x = e.lnPost.forward(b, x)
	b.Eval(x)
	// The result is the caller's, alive until its next Sweep (NewState
	// takes its cross-attention from it and sweeps); the last block's x
	// goes with that sweep too.
	if sw != nil && prev != nil {
		sw.Unpin(prev)
	}
	return x
}

// freeTensor releases a load-time temporary on backends that manage memory
// explicitly (the optional Sweeper capability); a no-op elsewhere.
func freeTensor(b engine.Backend, ts ...engine.Tensor) {
	if f, ok := b.(interface{ Free(ts ...engine.Tensor) }); ok {
		f.Free(ts...)
	}
}
