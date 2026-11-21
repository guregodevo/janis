package inkling2

import (
	"fmt"
	"strings"
	"sync"

	"github.com/guregodevo/janis/engine"
	"github.com/guregodevo/janis/safetensors"
)

// nvfp4Unit is what MLX divides an nvfp4 global scale by: the largest
// 4-bit value (6) times the largest 8-bit scale (448).
const nvfp4Unit = 2688

// expertStack is one projection of every expert of a layer, stacked
// [experts, out, in] and quantized in one of MLX's floating-point modes.
//
// The stack is on the device (W, Scales), or left on disk (stream): then
// each step reads the rows of the experts its tokens were routed to, and
// no more. The experts are nearly all of the model (15.8 of 16.1 GB on the
// published 4-bit one) and a token uses 8 of the 128 of a layer, so a host
// with less memory than the model runs it from the file cache.
type expertStack struct {
	W, Scales engine.Tensor
	// Global is each expert's own scale over its nvfp4 block scales, already
	// divided by nvfp4Unit: float32 [experts]. Nil when the stack has none.
	Global          engine.Tensor
	GroupSize, Bits int
	Mode            string

	stream           *safetensors.File
	name             string
	wShape, sShape   []int // of one expert
	wBytes, sBytes   int
	wBuf, sBuf       []byte // the rows of one step, reused
	readBytes, reads int64
}

// expertModes are the formats by (bits, group size), as the reference tells
// them apart: the checkpoint's quantization config does not name them.
var expertModes = map[[2]int]string{{4, 16}: "nvfp4", {4, 32}: "mxfp4", {8, 32}: "mxfp8"}

func (ld *loader) experts(prefix string, in int) *expertStack {
	if ld.err != nil {
		return nil
	}
	e := &expertStack{name: prefix}
	wdt, ws, err := ld.st.Info(prefix + ".weight")
	if err != nil {
		ld.err = err
		return nil
	}
	sdt, ss, err := ld.st.Info(prefix + ".scales")
	if err != nil {
		ld.err = err
		return nil
	}
	if wdt != "U32" || sdt != "U8" || len(ws) != 3 || len(ss) != 3 {
		ld.err = fmt.Errorf("%s: weight %s %v and scales %s %v are not a stack of nvfp4 or mxfp8 experts", prefix, wdt, ws, sdt, ss)
		return nil
	}
	if _, _, err := ld.st.Info(prefix + ".biases"); err == nil {
		ld.err = fmt.Errorf("%s: experts quantized with zero points (affine) are not supported", prefix)
		return nil
	}
	e.GroupSize = in / ss[2]
	e.Bits = ws[2] * 32 / in
	mode, ok := expertModes[[2]int{e.Bits, e.GroupSize}]
	if !ok {
		ld.err = fmt.Errorf("%s: %d-bit experts in groups of %d are in no format janis reads", prefix, e.Bits, e.GroupSize)
		return nil
	}
	e.Mode = mode
	e.wShape, e.sShape = ws[1:], ss[1:]
	e.wBytes, e.sBytes = ws[1]*ws[2]*4, ss[1]*ss[2]
	if ld.cfg.StreamExperts {
		e.stream = ld.st
	} else {
		e.W, e.Scales = ld.tensor(prefix+".weight"), ld.tensor(prefix+".scales")
	}
	if _, _, err := ld.st.Info(prefix + ".global_scale"); err == nil {
		e.Global = ld.c.ScalarMul(ld.f32(prefix+".global_scale"), 1.0/nvfp4Unit)
	}
	return e
}

// expertReads bounds the rows read from disk at once.
const expertReads = 8

// rows reads the experts' rows from disk into one stack on the device,
// [len(experts), out, in], in the order given.
func (e *expertStack) rows(c *ctx, experts []int32) (w, scales engine.Tensor, err error) {
	n := len(experts)
	if cap(e.wBuf) < n*e.wBytes {
		e.wBuf, e.sBuf = make([]byte, n*e.wBytes), make([]byte, n*e.sBytes)
	}
	wBuf, sBuf := e.wBuf[:n*e.wBytes], e.sBuf[:n*e.sBytes]
	errs := make([]error, 2*n)
	sem := make(chan struct{}, expertReads)
	var wg sync.WaitGroup
	for i, x := range experts {
		for j, read := range []func() error{
			func() error {
				return e.stream.ReadStackedRow(e.name+".weight", int(x), wBuf[i*e.wBytes:(i+1)*e.wBytes])
			},
			func() error {
				return e.stream.ReadStackedRow(e.name+".scales", int(x), sBuf[i*e.sBytes:(i+1)*e.sBytes])
			},
		} {
			wg.Add(1)
			sem <- struct{}{}
			go func(at int, read func() error) {
				defer wg.Done()
				errs[at] = read()
				<-sem
			}(2*i+j, read)
		}
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return nil, nil, err
		}
	}
	e.reads += int64(n)
	e.readBytes += int64(len(wBuf) + len(sBuf))
	// FromRaw copies, so the buffers serve the next step.
	w = c.FromRaw(engine.U32, wBuf, append([]int{n}, e.wShape...)...)
	scales = c.FromRaw(engine.U8, sBuf, append([]int{n}, e.sShape...)...)
	return w, scales, nil
}

// forward applies to each row of x [1, L, k, 1, in] the expert r
// routes it to -> [1, L, k, 1, out], in the model dtype.
func (e *expertStack) forward(c *ctx, x engine.Tensor, r routed) engine.Tensor {
	w, scales, at := e.W, e.Scales, r.idx
	if e.stream != nil {
		var err error
		if w, scales, err = e.rows(c, r.experts); err != nil {
			// The file was read at the load; a row that cannot be read now
			// is a disk failing under a running model.
			panic(fmt.Sprintf("inkling2: %v", err))
		}
		at = r.local
	}
	y := c.ops.GatherModeMatmul(x, w, scales, at, e.GroupSize, e.Bits, e.Mode)
	if e.Global == nil {
		return y
	}
	s := r.idx.Shape()
	g := c.Reshape(c.TakeAxis(e.Global, r.idx, 0), s[0], s[1], s[2], 1, 1)
	return c.model(c.Mul(c.f32(y), g))
}

// routed is where a step's tokens were sent: idx [1, L, k] names an expert
// of the layer for each of a token's k places. With the experts on disk,
// experts lists the distinct ones, in the order of the stack read for the
// step, and local is idx in that stack's numbering.
type routed struct {
	idx     engine.Tensor
	experts []int32
	local   engine.Tensor
}

// route reads the router's choice back from the device when the experts
// are on disk: the rows to read must be known before the next operation.
func (m *MoE) route(c *ctx, idx engine.Tensor) routed {
	r := routed{idx: idx}
	if m.GateUp.stream == nil {
		return r
	}
	ids := c.Ints(idx)
	at := map[int32]int32{}
	local := make([]int32, len(ids))
	for i, id := range ids {
		slot, ok := at[id]
		if !ok {
			slot = int32(len(r.experts))
			at[id] = slot
			r.experts = append(r.experts, id)
		}
		local[i] = slot
	}
	r.local = c.FromInt32(local, idx.Shape()...)
	return r
}

// MoE is a layer's MLP: a router scores every expert for each token, the
// TopK best run, and their outputs are summed by score.
type MoE struct {
	RouterT, RouterBias engine.Tensor // float32 [hidden, experts], [experts]
	GateUp, Down        *expertStack
	TopK, Intermediate  int
	GateCap, UpCap      float32
	RouteScale          float32
}

func loadMoE(c *ctx, st *safetensors.File, prefix string, cfg Config) (*MoE, error) {
	ld := loader{c: c, st: st, cfg: cfg}
	m := &MoE{
		TopK: cfg.TopK, Intermediate: cfg.Intermediate,
		GateCap: cfg.GateCap, UpCap: cfg.UpCap, RouteScale: cfg.RouteScale,
	}
	if w := ld.f32(prefix + ".gate.weight"); w != nil {
		m.RouterT = c.Transpose(w, 1, 0)
	}
	m.RouterBias = ld.f32(prefix + ".gate.e_score_correction_bias")
	m.GateUp = ld.experts(prefix+".experts.gate_up_proj", cfg.Hidden)
	m.Down = ld.experts(prefix+".experts.down_proj", cfg.Intermediate)
	return m, ld.err
}

// Forward runs the block on x [1, L, hidden] in the model dtype.
func (m *MoE) Forward(c *ctx, x engine.Tensor) engine.Tensor {
	L, hidden, k, I := x.Shape()[1], x.Shape()[2], m.TopK, m.Intermediate

	// The bias only chooses the experts; the weights are the plain scores
	// of the chosen, normalized to sum to RouteScale.
	scores := c.Sigmoid(c.MatMul(c.f32(x), m.RouterT)) // [1, L, experts]
	idx := c.ops.TopK(c.Add(scores, m.RouterBias), k)  // [1, L, k]
	weights := c.ops.TakeAlong(scores, idx, 2)
	weights = c.ops.Div(weights, c.ScalarAdd(c.sumLast(weights), 1e-20))
	weights = c.ScalarMul(weights, m.RouteScale)

	// gate_up's output rows alternate gate, up.
	r := m.route(c, idx)
	gu := c.Reshape(m.GateUp.forward(c, c.Reshape(x, 1, L, 1, 1, hidden), r), 1, L, k, I, 2)
	gate := c.f32(c.Reshape(c.Slice(gu, 4, 0, 1), 1, L, k, 1, I))
	up := c.f32(c.Reshape(c.Slice(gu, 4, 1, 2), 1, L, k, 1, I))
	// SwiGLU with both branches softly capped: tanh(x/cap)*cap.
	act := c.Mul(c.ScalarMul(c.Tanh(c.ScalarMul(gate, 1/m.GateCap)), m.GateCap), c.Sigmoid(gate))
	act = c.Mul(act, c.ScalarMul(c.Tanh(c.ScalarMul(up, 1/m.UpCap)), m.UpCap))

	out := c.f32(c.Reshape(m.Down.forward(c, c.model(act), r), 1, L, k, hidden))
	out = c.Mul(out, c.Reshape(weights, 1, L, k, 1))
	return c.model(c.ScalarMul(c.Mean(out, 2), float32(k)))
}

// ExpertBytes is the size of the model's expert weights, from the header.
func ExpertBytes(st *safetensors.File) int64 {
	var total int64
	for _, name := range st.Names() {
		if !strings.Contains(name, ".mlp.experts.") {
			continue
		}
		dt, shape, err := st.Info(name)
		if err != nil {
			continue
		}
		n := int64(map[string]int{"U32": 4, "U8": 1, "F32": 4}[dt])
		for _, d := range shape {
			n *= int64(d)
		}
		total += n
	}
	return total
}
