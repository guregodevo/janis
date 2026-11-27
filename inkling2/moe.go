package inkling2

import (
	"container/list"
	"fmt"
	"strings"
	"sync"
	"time"

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
	Global engine.Tensor
	// paired: the backend gives the stack's output rows as the first
	// projection's, then the second's, not alternating (PairRows).
	paired bool
	// Biases are the zero points of an affine stack (Mode "affine"), nil
	// for the floating-point modes.
	Biases          engine.Tensor
	GroupSize, Bits int
	Mode            string

	cache            *expertCache // of the experts read from disk
	held             map[int32]*cachedExpert
	stream           *safetensors.File
	name             string
	wShape, sShape   []int // of one expert
	wBytes, sBytes   int
	wBuf, sBuf, bBuf []byte       // one expert's rows, reused
	sDType           engine.DType // of the scales (and the zero points)
	readBytes, reads int64
	readNanos        int64
}

// expertModes are the formats by (bits, group size), as the reference tells
// them apart: the checkpoint's quantization config does not name them.
var expertModes = map[[2]int]string{{4, 16}: "nvfp4", {4, 32}: "mxfp4", {8, 32}: "mxfp8"}

func (ld *loader) experts(prefix string, in int) *expertStack {
	if ld.err != nil {
		return nil
	}
	e := &expertStack{name: ld.name(prefix)}
	wdt, ws, err := ld.st.Info(e.name + ".weight")
	if err != nil {
		ld.err = err
		return nil
	}
	sdt, ss, err := ld.st.Info(e.name + ".scales")
	if err != nil {
		ld.err = err
		return nil
	}
	if wdt != "U32" || len(ws) != 3 || len(ss) != 3 {
		ld.err = fmt.Errorf("%s: weight %s %v and scales %s %v are not a stack of quantized experts", e.name, wdt, ws, sdt, ss)
		return nil
	}
	e.GroupSize = in / ss[2]
	e.Bits = ws[2] * 32 / in
	_, _, berr := ld.st.Info(e.name + ".biases")
	switch {
	case berr == nil && sdt != "U8":
		e.Mode = "affine"
	case berr != nil && sdt == "U8":
		mode, ok := expertModes[[2]int{e.Bits, e.GroupSize}]
		if !ok {
			ld.err = fmt.Errorf("%s: %d-bit experts in groups of %d are in no format janis reads", e.name, e.Bits, e.GroupSize)
			return nil
		}
		e.Mode = mode
	default:
		ld.err = fmt.Errorf("%s: scales of %s are neither an affine stack's nor a floating-point one's", e.name, sdt)
		return nil
	}
	var ok bool
	if e.sDType, ok = map[string]engine.DType{"U8": engine.U8, "F32": engine.F32, "F16": engine.F16, "BF16": engine.BF16}[sdt]; !ok {
		ld.err = fmt.Errorf("%s: scales of %s", e.name, sdt)
		return nil
	}
	size := map[engine.DType]int{engine.U8: 1, engine.F32: 4, engine.F16: 2, engine.BF16: 2}[e.sDType]
	e.wShape, e.sShape = ws[1:], ss[1:]
	e.wBytes, e.sBytes = ws[1]*ws[2]*4, ss[1]*ss[2]*size
	if ld.cfg.StreamExperts {
		e.stream, e.cache, e.held = ld.st, ld.c.experts, map[int32]*cachedExpert{}
	} else {
		e.W, e.Scales = ld.tensor(prefix+".weight"), ld.tensor(prefix+".scales")
		if e.Mode == "affine" {
			e.Biases = ld.tensor(prefix + ".biases")
		}
	}
	if _, _, err := ld.st.Info(e.name + ".global_scale"); err == nil {
		e.Global = ld.c.ScalarMul(ld.f32(prefix+".global_scale"), 1.0/nvfp4Unit)
	}
	return e
}

// matmul is x against the stack (w, scales, biases) by the stack's format.
func (e *expertStack) matmul(c *ctx, x, w, scales, biases, idx engine.Tensor) engine.Tensor {
	if e.Mode == "affine" {
		return c.GatherQuantMatmul(x, w, scales, biases, idx, true, e.GroupSize, e.Bits)
	}
	return c.ops.GatherModeMatmul(x, w, scales, idx, e.GroupSize, e.Bits, e.Mode)
}

// expertCache keeps the experts read from disk on the device, the least
// recently used going first once they pass its budget. Without it every
// step read its experts again: 1.84 GB a token on the published model,
// 44% of a request's time waiting for the reads and as much again
// uploading them. It is one cache for the whole model: the budget is the
// model's, and a layer whose routing varies more takes more of it.
type expertCache struct {
	budget, bytes int64
	lru           *list.List // front: most recently used; of *cachedExpert
	hits, misses  int64
}

type cachedExpert struct {
	stack     *expertStack
	expert    int32
	w, scales engine.Tensor // [1, out, in]
	biases    engine.Tensor // affine stacks only
	el        *list.Element
}

// get returns the experts' weights on the device, read from disk for the
// ones the cache does not hold.
func (e *expertStack) get(c *ctx, experts []int32) (map[int32]*cachedExpert, error) {
	out := make(map[int32]*cachedExpert, len(experts))
	var missing []int32
	for _, x := range experts {
		if ce, ok := e.held[x]; ok {
			e.cache.lru.MoveToFront(ce.el)
			e.cache.hits++
			out[x] = ce
		} else {
			missing = append(missing, x)
		}
	}
	if len(missing) == 0 {
		return out, nil
	}
	start := time.Now()
	n := len(missing)
	affine := e.Mode == "affine"
	if cap(e.wBuf) < n*e.wBytes {
		e.wBuf, e.sBuf = make([]byte, n*e.wBytes), make([]byte, n*e.sBytes)
		if affine {
			e.bBuf = make([]byte, n*e.sBytes)
		}
	}
	errs := make([]error, 3*n)
	sem := make(chan struct{}, expertReads)
	var wg sync.WaitGroup
	for i, x := range missing {
		reads := []func() error{
			func() error {
				return e.stream.ReadStackedRow(e.name+".weight", int(x), e.wBuf[i*e.wBytes:(i+1)*e.wBytes])
			},
			func() error {
				return e.stream.ReadStackedRow(e.name+".scales", int(x), e.sBuf[i*e.sBytes:(i+1)*e.sBytes])
			},
		}
		if affine {
			reads = append(reads, func() error {
				return e.stream.ReadStackedRow(e.name+".biases", int(x), e.bBuf[i*e.sBytes:(i+1)*e.sBytes])
			})
		}
		for j, read := range reads {
			wg.Add(1)
			sem <- struct{}{}
			go func(at int, read func() error) {
				defer wg.Done()
				errs[at] = read()
				<-sem
			}(3*i+j, read)
		}
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}
	e.readNanos += time.Since(start).Nanoseconds()
	e.reads += int64(n)
	e.readBytes += int64(n * (e.wBytes + e.sBytes))
	sw, _ := c.Backend.(engine.Sweeper)
	for i, x := range missing {
		// FromRaw copies, so the buffers serve the next step.
		ce := &cachedExpert{stack: e, expert: x,
			w:      c.FromRaw(engine.U32, e.wBuf[i*e.wBytes:(i+1)*e.wBytes], append([]int{1}, e.wShape...)...),
			scales: c.FromRaw(e.sDType, e.sBuf[i*e.sBytes:(i+1)*e.sBytes], append([]int{1}, e.sShape...)...),
		}
		if affine {
			ce.biases = c.FromRaw(e.sDType, e.bBuf[i*e.sBytes:(i+1)*e.sBytes], append([]int{1}, e.sShape...)...)
		}
		if sw != nil { // a step's sweep must not free it
			sw.Pin(ce.w, ce.scales, ce.biases)
		}
		ce.el = e.cache.lru.PushFront(ce)
		e.held[x] = ce
		e.cache.bytes += int64(e.wBytes + e.sBytes)
		e.cache.misses++
		out[x] = ce
	}
	// Evict past the budget, never what this step is about to use.
	using := len(experts)
	for e.cache.bytes > e.cache.budget && e.cache.lru.Len() > using {
		old := e.cache.lru.Back().Value.(*cachedExpert)
		if old.stack == e {
			if _, mine := out[old.expert]; mine {
				break
			}
		}
		e.cache.lru.Remove(old.el)
		delete(old.stack.held, old.expert)
		e.cache.bytes -= int64(old.stack.wBytes + old.stack.sBytes)
		if sw != nil {
			sw.Unpin(old.w, old.scales, old.biases)
		}
	}
	return out, nil
}

// expertReads bounds the rows read from disk at once.
const expertReads = 8

// forward applies to each row of x [1, L, k, 1, in] the expert r routes it
// to -> [1, L, k, 1, out], in the model dtype.
func (e *expertStack) forward(c *ctx, x engine.Tensor, r routed) engine.Tensor {
	var y engine.Tensor
	if e.stream == nil {
		y = e.matmul(c, x, e.W, e.Scales, e.Biases, r.idx)
	} else {
		held, err := e.get(c, r.experts)
		if err != nil {
			// The file was read at the load; a row that cannot be read now
			// is a disk failing under a running model.
			panic(fmt.Sprintf("inkling2: %v", err))
		}
		// One product per routed row, each with its expert as a stack of
		// one: nothing is copied to bring a step's experts together.
		s := x.Shape()
		L, k, in := s[1], r.idx.Shape()[2], s[4]
		first := c.FromInt32([]int32{0}, 1, 1, 1)
		var rows engine.Tensor
		for l := 0; l < L; l++ {
			var row engine.Tensor
			for j := 0; j < k; j++ {
				ce := held[r.ids[l*k+j]]
				xi := c.Slice(x, 1, l, l+1)
				if s[2] > 1 {
					xi = c.Slice(xi, 2, j, j+1)
				}
				yi := e.matmul(c, c.Reshape(xi, 1, 1, 1, 1, in), ce.w, ce.scales, ce.biases, first)
				if row == nil {
					row = yi
				} else {
					row = c.Concat(row, yi, 2)
				}
			}
			if rows == nil {
				rows = row
			} else {
				rows = c.Concat(rows, row, 1)
			}
		}
		y = rows
	}
	if e.Global == nil {
		return y
	}
	s := r.idx.Shape()
	g := c.Reshape(c.TakeAxis(e.Global, r.idx, 0), s[0], s[1], s[2], 1, 1)
	return c.model(c.Mul(c.f32(y), g))
}

// routed is where a step's tokens were sent: idx [1, L, k] names an expert
// of the layer for each of a token's k places. With the experts on disk,
// ids is idx read back and experts the distinct ones.
type routed struct {
	idx     engine.Tensor
	ids     []int32
	experts []int32
}

// route reads the router's choice back from the device when the experts
// are on disk: the rows to read must be known before the next operation.
func (m *MoE) route(c *ctx, idx engine.Tensor) routed {
	r := routed{idx: idx}
	if m.Down.stream == nil {
		return r
	}
	r.ids = c.Ints(idx)
	seen := map[int32]bool{}
	for _, id := range r.ids {
		if !seen[id] {
			seen[id] = true
			r.experts = append(r.experts, id)
		}
	}
	return r
}

// MoE is a layer's MLP: a router scores every expert for each token, the
// TopK best run, and their outputs are summed by score.
type MoE struct {
	RouterT, RouterBias engine.Tensor // float32 [hidden, experts], [experts]
	// GateUp holds both projections, their output rows alternating gate,
	// up; a checkpoint that keeps them apart has Gate and Up instead.
	GateUp, Gate, Up, Down *expertStack
	TopK, Intermediate     int
	GateCap, UpCap         float32
	RouteScale             float32
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
	if cfg.AltLayout {
		m.Gate = ld.experts(prefix+".experts.gate_proj", cfg.Hidden)
		m.Up = ld.experts(prefix+".experts.up_proj", cfg.Hidden)
	} else {
		m.GateUp = ld.experts(prefix+".experts.gate_up_proj", cfg.Hidden)
	}
	m.Down = ld.experts(prefix+".experts.down_proj", cfg.Intermediate)
	if g := m.GateUp; g != nil && c.rel != nil && g.W != nil && g.Mode != "affine" {
		g.paired = c.rel.PairRows(g.W)
	}
	return m, ld.err
}

// Forward runs the block on x [1, L, hidden] in the model dtype.
func (m *MoE) Forward(c *ctx, x engine.Tensor) engine.Tensor {
	L, hidden, k, I := x.Shape()[1], x.Shape()[2], m.TopK, m.Intermediate

	// The bias only chooses the experts; the weights are the plain scores
	// of the chosen, normalized to sum to RouteScale.
	scores := c.Sigmoid(c.MatMul(c.f32(x), m.RouterT)) // [1, L, experts]
	idx := c.ops.TopK(c.Add(scores, m.RouterBias), k)  // [1, L, k]
	// One token on a backend with these as operations: the two caps'
	// factors leave the activation for the weights, which are linear in it.
	fused := c.rel != nil && L == 1
	var weights engine.Tensor
	if fused {
		weights = c.rel.RouteWeights(scores, idx, m.RouteScale*m.GateCap*m.UpCap)
	} else {
		weights = c.ops.TakeAlong(scores, idx, 2)
		weights = c.ops.Div(weights, c.ScalarAdd(c.sumLast(weights), 1e-20))
		weights = c.ScalarMul(weights, m.RouteScale)
	}

	r := m.route(c, idx)
	xe := c.Reshape(x, 1, L, 1, 1, hidden)
	var act engine.Tensor
	if fused && m.GateUp != nil && m.GateUp.paired {
		act = c.rel.CappedSwiGLU(c.f32(m.GateUp.forward(c, xe, r)), m.GateCap, m.UpCap)
	} else {
		var gate, up engine.Tensor
		switch {
		case m.GateUp != nil && m.GateUp.paired:
			gu := m.GateUp.forward(c, xe, r) // [1, L, k, 1, 2I]: gate's rows, then up's
			gate, up = c.f32(c.Slice(gu, 4, 0, I)), c.f32(c.Slice(gu, 4, I, 2*I))
		case m.GateUp != nil:
			gu := c.Reshape(m.GateUp.forward(c, xe, r), 1, L, k, I, 2)
			gate = c.f32(c.Reshape(c.Slice(gu, 4, 0, 1), 1, L, k, 1, I))
			up = c.f32(c.Reshape(c.Slice(gu, 4, 1, 2), 1, L, k, 1, I))
		default:
			gate, up = c.f32(m.Gate.forward(c, xe, r)), c.f32(m.Up.forward(c, xe, r))
		}
		// SwiGLU with both branches softly capped: tanh(x/cap)*cap.
		if fused {
			act = c.Mul(c.Tanh(c.ScalarMul(gate, 1/m.GateCap)), c.Sigmoid(gate))
			act = c.Mul(act, c.Tanh(c.ScalarMul(up, 1/m.UpCap)))
		} else {
			act = c.Mul(c.ScalarMul(c.Tanh(c.ScalarMul(gate, 1/m.GateCap)), m.GateCap), c.Sigmoid(gate))
			act = c.Mul(act, c.ScalarMul(c.Tanh(c.ScalarMul(up, 1/m.UpCap)), m.UpCap))
		}
	}

	out := c.f32(c.Reshape(m.Down.forward(c, c.model(act), r), 1, L, k, hidden))
	if c.rel != nil && L == 1 {
		return c.model(c.rel.WeightedSum(out, weights))
	}
	out = c.Mul(out, c.Reshape(weights, 1, L, k, 1))
	return c.model(c.ScalarMul(c.Mean(out, 2), float32(k)))
}

// ExpertBytes is the size of the model's expert weights, from the header.
func ExpertBytes(st *safetensors.File) int64 {
	var total int64
	for _, name := range st.Names() {
		if !strings.Contains(name, ".mlp.experts.") && !strings.Contains(name, ".mlp.switch_mlp.") {
			continue
		}
		dt, shape, err := st.Info(name)
		if err != nil {
			continue
		}
		n := int64(map[string]int{"U32": 4, "U8": 1, "F32": 4, "F16": 2, "BF16": 2}[dt])
		for _, d := range shape {
			n *= int64(d)
		}
		total += n
	}
	return total
}

// stacks are the block's expert stacks.
func (m *MoE) stacks() []*expertStack {
	var out []*expertStack
	for _, e := range []*expertStack{m.GateUp, m.Gate, m.Up, m.Down} {
		if e != nil {
			out = append(out, e)
		}
	}
	return out
}
