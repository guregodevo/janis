package qwen

import (
	"container/list"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"memdoor/llm/engine"
	"memdoor/llm/safetensors"
)

// moeReadConcurrency bounds in-flight expert-row preads per store. Apple SSDs
// sustain multiple concurrent reads at full bandwidth; unbounded goroutines
// would thrash the page cache under memory pressure.
const moeReadConcurrency = 8

type exKey struct {
	layer int
	proj  string
	e     int
}

type exEntry struct {
	key exKey
	lin *QuantLinear
	sz  int64
}

// ExpertStore lazily materializes individual MoE experts from a (sharded)
// safetensors file, keeping only a bounded LRU set resident. This is what lets
// a 30B-class MoE run on a 16 GB Mac: per token we touch ~8 of 128 experts via
// StackedRow + FromRaw instead of loading all 16.8 GB (which OOMs). The cache is
// keyed per (layer, projection, expert) so a hot expert is materialized once.
type ExpertStore struct {
	b            engine.Backend
	sw           engine.Sweeper // non-nil on MLX: cached experts must be pinned
	st           *safetensors.File
	groupSize    int
	bits         int
	budget       int64
	bytes        int64
	cache        map[exKey]*list.Element
	lru          *list.List // front = most-recently-used; Value is *exEntry
	Hits, Misses int

	readSem   chan struct{} // bounds concurrent expert-row preads
	readBytes atomic.Int64
	readNanos atomic.Int64

	slots map[int]*layerSlots // per-layer expert slot cache (decode hot path)
}

// NewExpertStore builds a store over st, bounding resident expert weights to
// budgetBytes (older experts are evicted LRU once exceeded).
func NewExpertStore(b engine.Backend, st *safetensors.File, groupSize, bits int, budgetBytes int64) *ExpertStore {
	sw, _ := b.(engine.Sweeper)
	return &ExpertStore{
		b: b, sw: sw, st: st, groupSize: groupSize, bits: bits, budget: budgetBytes,
		cache: map[exKey]*list.Element{}, lru: list.New(),
		readSem: make(chan struct{}, moeReadConcurrency),
	}
}

// Expert returns the quantized projection (gate_proj/up_proj/down_proj) for one
// expert of one layer — cached on a hit, materialized from disk on a miss.
func (s *ExpertStore) Expert(layer int, proj string, e int) (*QuantLinear, error) {
	k := exKey{layer, proj, e}
	if el, ok := s.cache[k]; ok {
		s.Hits++
		s.lru.MoveToFront(el)
		return el.Value.(*exEntry).lin, nil
	}
	s.Misses++
	lin, sz, err := s.load(layer, proj, e)
	if err != nil {
		return nil, err
	}
	s.insert(k, lin, sz)
	return lin, nil
}

// insert pins lin into the cache under k and evicts LRU entries once the byte
// budget is exceeded (keeping a minimum floor so a token's own working set
// cannot evict itself mid-assembly).
func (s *ExpertStore) insert(k exKey, lin *QuantLinear, sz int64) {
	if s.sw != nil { // pin so the decode loop's per-step Sweep won't free it
		s.sw.Pin(lin.Weight, lin.Scales, lin.Biases)
	}
	el := s.lru.PushFront(&exEntry{key: k, lin: lin, sz: sz})
	s.cache[k] = el
	s.bytes += sz
	for s.bytes > s.budget && s.lru.Len() > 3 {
		back := s.lru.Back()
		ent := back.Value.(*exEntry)
		s.lru.Remove(back)
		delete(s.cache, ent.key)
		s.bytes -= ent.sz
		if s.sw != nil {
			s.sw.Unpin(ent.lin.Weight, ent.lin.Scales, ent.lin.Biases)
		}
	}
}

// moeProjs are the three projections of one expert's SwiGLU FFN.
var moeProjs = [3]string{"gate_proj", "up_proj", "down_proj"}

// EnsureCached materializes every (projection, expert) pair the current token
// needs into the GPU cache, reading ALL missing rows — across projections and
// tensors — in one bounded-parallel batch. After it returns, Stack calls for
// these experts are pure cache hits: no disk reads, no host-to-device upload.
// This is what converts the 40-60% token-to-token expert overlap of top-8
// routing into saved IO instead of repeated fetches.
func (s *ExpertStore) EnsureCached(layer int, experts []int) error {
	var miss [3][]int
	for pi, proj := range moeProjs {
		for _, e := range experts {
			if _, ok := s.cache[exKey{layer, proj, e}]; !ok {
				miss[pi] = append(miss[pi], e)
			}
		}
	}
	if len(miss[0])+len(miss[1])+len(miss[2]) == 0 {
		return nil
	}

	var reads [3][3]stackedRows // [projection][weight, scales, biases]
	var errs [3][3]error
	var wg sync.WaitGroup
	start := time.Now()
	for pi, proj := range moeProjs {
		if len(miss[pi]) == 0 {
			continue
		}
		base := fmt.Sprintf("model.layers.%d.mlp.switch_mlp.%s", layer, proj)
		for ti, suffix := range [3]string{".weight", ".scales", ".biases"} {
			wg.Add(1)
			go func(pi, ti int, name string, experts []int) {
				defer wg.Done()
				reads[pi][ti], errs[pi][ti] = s.readRows(name, experts)
			}(pi, ti, base+suffix, miss[pi])
		}
	}
	wg.Wait()
	s.readNanos.Add(time.Since(start).Nanoseconds())
	for pi := range moeProjs {
		for ti := range errs[pi] {
			if err := errs[pi][ti]; err != nil {
				return err
			}
		}
	}

	// Upload + insert sequentially — the backend is single-threaded.
	for pi, proj := range moeProjs {
		for mi, e := range miss[pi] {
			var ts [3]engine.Tensor
			var sz int64
			for ti := range ts {
				r := reads[pi][ti]
				edt, err := stDType(r.dt)
				if err != nil {
					return fmt.Errorf("expert layer %d %s: %w", layer, proj, err)
				}
				ts[ti] = s.b.FromRaw(edt, r.rows[mi], r.rowShape...)
				sz += int64(len(r.rows[mi]))
			}
			lin := &QuantLinear{Weight: ts[0], Scales: ts[1], Biases: ts[2],
				GroupSize: s.groupSize, Bits: s.bits}
			s.insert(exKey{layer, proj, e}, lin, sz)
			s.Misses++
		}
	}
	return nil
}

// Resident reports the current resident expert-weight byte count.
func (s *ExpertStore) Resident() int64 { return s.bytes }

// StackRaw gathers the active experts' weight/scales/biases directly from disk
// into one stacked [K, ...] tensor per projection — reading only those experts'
// bytes (via StackedRow) and concatenating them into a single FromRaw. Unlike
// Stack it never materializes per-expert tensors or duplicates a resident cache,
// so peak memory is just the transient per-token stacks (~tens of MB); hot
// experts stay fast via the OS page cache. This is the path that fits in 16 GB.
func (s *ExpertStore) StackRaw(layer int, proj string, experts []int) (w, sc, bi engine.Tensor, err error) {
	ps, err := s.StackProjs(layer, experts, proj)
	if err != nil {
		return nil, nil, nil, err
	}
	return ps[0].W, ps[0].S, ps[0].B, nil
}

// StackedProj is one projection's stacked weight/scales/biases tensors.
type StackedProj struct {
	W, S, B engine.Tensor
}

// StackProjs gathers the active experts for several projections of one layer.
// Every tensor's rows (3 per projection) fetch in one concurrent read phase, so
// cold misses across projections share the SSD queue instead of arriving in
// three serial batches. The upload phase stays sequential — the backend may not
// be called from multiple goroutines.
func (s *ExpertStore) StackProjs(layer int, experts []int, projs ...string) ([]StackedProj, error) {
	names := make([]string, 0, len(projs)*3)
	for _, p := range projs {
		base := fmt.Sprintf("model.layers.%d.mlp.switch_mlp.%s", layer, p)
		names = append(names, base+".weight", base+".scales", base+".biases")
	}

	reads := make([]stackedRows, len(names))
	errs := make([]error, len(names))
	var wg sync.WaitGroup
	start := time.Now()
	for i, name := range names {
		wg.Add(1)
		go func(i int, name string) {
			defer wg.Done()
			reads[i], errs[i] = s.readRows(name, experts)
		}(i, name)
	}
	wg.Wait()
	s.readNanos.Add(time.Since(start).Nanoseconds())
	for i, e := range errs {
		if e != nil {
			return nil, fmt.Errorf("expert %s: %w", names[i], e)
		}
	}

	out := make([]StackedProj, len(projs))
	for i := range projs {
		var p StackedProj
		var err error
		if p.W, err = s.upload(reads[i*3], len(experts)); err != nil {
			return nil, fmt.Errorf("expert %s: %w", names[i*3], err)
		}
		if p.S, err = s.upload(reads[i*3+1], len(experts)); err != nil {
			return nil, fmt.Errorf("expert %s: %w", names[i*3+1], err)
		}
		if p.B, err = s.upload(reads[i*3+2], len(experts)); err != nil {
			return nil, fmt.Errorf("expert %s: %w", names[i*3+2], err)
		}
		out[i] = p
	}
	return out, nil
}

// stackedRows is one tensor's rows for the active experts, in expert order.
type stackedRows struct {
	dt       string
	rowShape []int
	rows     [][]byte
}

// readRows fetches one row per expert with bounded parallel preads, so cold
// rows come off the SSD concurrently instead of queueing behind each other.
// Hot rows still resolve from the OS page cache at memcpy speed.
func (s *ExpertStore) readRows(name string, experts []int) (stackedRows, error) {
	out := stackedRows{rows: make([][]byte, len(experts))}
	errs := make([]error, len(experts))
	var wg sync.WaitGroup
	var mu sync.Mutex
	var bytes int64
	for i, e := range experts {
		wg.Add(1)
		s.readSem <- struct{}{}
		go func(i, e int) {
			defer wg.Done()
			defer func() { <-s.readSem }()
			dt, shp, raw, err := s.st.StackedRow(name, e)
			if err != nil {
				errs[i] = err
				return
			}
			out.rows[i] = raw
			atomic.AddInt64(&bytes, int64(len(raw)))
			mu.Lock()
			out.dt, out.rowShape = dt, shp
			mu.Unlock()
		}(i, e)
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return stackedRows{}, err
		}
	}
	s.readBytes.Add(bytes)
	return out, nil
}

// upload assembles ordered rows into one stacked device tensor.
func (s *ExpertStore) upload(r stackedRows, nExperts int) (engine.Tensor, error) {
	edt, err := stDType(r.dt)
	if err != nil {
		return nil, err
	}
	var buf []byte
	if len(r.rows) > 0 {
		buf = make([]byte, 0, len(r.rows)*len(r.rows[0]))
	}
	for _, row := range r.rows {
		buf = append(buf, row...)
	}
	shape := append([]int{nExperts}, r.rowShape...)
	return s.b.FromRaw(edt, buf, shape...), nil
}

// IOStats reports cumulative expert-stream reads: total bytes and wall time
// spent awaiting row fetches. Wall time counts each readRows batch once, not
// per row, so it approximates the decode loop's actual IO stall.
func (s *ExpertStore) IOStats() (bytes int64, dur time.Duration) {
	return s.readBytes.Load(), time.Duration(s.readNanos.Load())
}

// Stack returns the active experts' weight/scales/biases stacked along a new
// leading axis ([K, ...]), for one fused GatherQuantMatmul call. Per-expert
// tensors come from the LRU cache; only the stacking is done per call.
func (s *ExpertStore) Stack(layer int, proj string, experts []int) (w, sc, bi engine.Tensor, err error) {
	for i, e := range experts {
		lin, lerr := s.Expert(layer, proj, e)
		if lerr != nil {
			return nil, nil, nil, lerr
		}
		ew := s.b.ExpandDims(lin.Weight, 0)
		es := s.b.ExpandDims(lin.Scales, 0)
		eb := s.b.ExpandDims(lin.Biases, 0)
		if i == 0 {
			w, sc, bi = ew, es, eb
		} else {
			w, sc, bi = s.b.Concat(w, ew, 0), s.b.Concat(sc, es, 0), s.b.Concat(bi, eb, 0)
		}
	}
	return w, sc, bi, nil
}

func (s *ExpertStore) load(layer int, proj string, e int) (*QuantLinear, int64, error) {
	base := fmt.Sprintf("model.layers.%d.mlp.switch_mlp.%s", layer, proj)
	w, wsz, err := s.tensor(base+".weight", e)
	if err != nil {
		return nil, 0, err
	}
	sc, ssz, err := s.tensor(base+".scales", e)
	if err != nil {
		return nil, 0, err
	}
	bi, bsz, err := s.tensor(base+".biases", e)
	if err != nil {
		return nil, 0, err
	}
	return &QuantLinear{Weight: w, Scales: sc, Biases: bi, GroupSize: s.groupSize, Bits: s.bits},
		wsz + ssz + bsz, nil
}

// MoEBlock is the Qwen3-MoE sparse FFN: a router picks the top-k experts per
// token; each selected expert's SwiGLU FFN runs (its weights streamed from the
// ExpertStore) and the outputs are score-weighted and summed. Mirrors mlx-lm's
// Qwen3MoeSparseMoeBlock: softmax over all experts, top-k, optional renorm.
type MoEBlock struct {
	Router     *QuantLinear // model.layers.N.mlp.gate (quantized)
	Store      *ExpertStore
	Layer      int
	TopK       int
	NumExperts int
	NormTopk   bool
}

// LoadMoEBlock loads the router for one layer; experts come from the shared store.
func LoadMoEBlock(b engine.Backend, st *safetensors.File, layer int, cfg Config, store *ExpertStore) (*MoEBlock, error) {
	prefix := fmt.Sprintf("model.layers.%d.mlp.gate", layer)
	gs, bits := cfg.QuantFor(prefix)
	router, err := LoadQuantLinear(b, st, prefix, gs, bits, false)
	if err != nil {
		return nil, fmt.Errorf("load router layer %d: %w", layer, err)
	}
	return &MoEBlock{Router: router, Store: store, Layer: layer,
		TopK: cfg.TopK, NumExperts: cfg.NumExperts, NormTopk: cfg.NormTopk}, nil
}

// topK returns the indices and values of the k largest entries of row.
func topK(row []float32, k int) (idx []int, val []float32) {
	order := make([]int, len(row))
	for i := range order {
		order[i] = i
	}
	for i := 0; i < k && i < len(order); i++ {
		best := i
		for j := i + 1; j < len(order); j++ {
			if row[order[j]] > row[order[best]] {
				best = j
			}
		}
		order[i], order[best] = order[best], order[i]
	}
	if k > len(order) {
		k = len(order)
	}
	idx = make([]int, k)
	val = make([]float32, k)
	for i := 0; i < k; i++ {
		idx[i] = order[i]
		val[i] = row[order[i]]
	}
	return idx, val
}

// Forward computes the sparse MoE FFN for x ([seq, hidden] or higher-rank with
// hidden last). Top-k routing is done host-side (the router logits are small),
// then only the selected experts' weights are materialized per token.
func (m *MoEBlock) Forward(b engine.Backend, x engine.Tensor) engine.Tensor {
	shp := x.Shape() // [..lead.., hidden], lead is usually [1, seq]
	nd := len(shp)
	lead := shp[:nd-1]
	nTok := 1
	for _, d := range lead {
		nTok *= d
	}

	probs := b.Softmax(m.Router.Forward(b, x), -1) // [..lead.., numExperts]
	pf := b.Floats(probs)

	// Host-side top-k for every token at once. Collect the UNIQUE experts this
	// layer touches (materialized once, not per token), the rhs index map into
	// that unique set, and the per-(token,slot) scores.
	K := m.TopK
	pos := make(map[int]int)
	var uniq []int
	ri := make([]int32, nTok*K)
	scores := make([]float32, nTok*K)
	for t := 0; t < nTok; t++ {
		idx, sc := topK(pf[t*m.NumExperts:(t+1)*m.NumExperts], K)
		if m.NormTopk {
			var s float32
			for _, v := range sc {
				s += v
			}
			if s > 0 {
				for i := range sc {
					sc[i] /= s
				}
			}
		}
		for j, e := range idx {
			p, ok := pos[e]
			if !ok {
				p = len(uniq)
				pos[e] = p
				uniq = append(uniq, e)
			}
			ri[t*K+j] = int32(p)
			scores[t*K+j] = sc[j]
		}
	}

	// Materialize the layer's unique experts, then one fused GatherQuantMatmul
	// over the whole sequence per projection.
	//
	// Single-token steps prefer the persistent slot cache: hits skip both the
	// disk read and the host-to-device upload, and misses land via ONE batched
	// ScatterRows per tensor, keeping the op count flat. (A per-expert cache
	// assembled with Concat chains was measured 20x slower — 24 small MLX ops
	// per layer per token dwarf the saved reads.) Multi-token chunks touch too
	// many experts to cache and use transient per-chunk stacks.
	var gW, gS, gB, uW, uS, uB, dW, dS, dB engine.Tensor
	slotted := false
	if nTok == 1 && slotCacheEnabled() {
		if projs, slotIdx, ok, serr := m.Store.SlotStack(m.Layer, uniq); serr != nil {
			panic(serr)
		} else if ok {
			for j := range ri {
				ri[j] = slotIdx[ri[j]]
			}
			gW, gS, gB = projs[0].W, projs[0].S, projs[0].B
			uW, uS, uB = projs[1].W, projs[1].S, projs[1].B
			dW, dS, dB = projs[2].W, projs[2].S, projs[2].B
			slotted = true
		}
	}
	if !slotted {
		ps, err := m.Store.StackProjs(m.Layer, uniq, "gate_proj", "up_proj", "down_proj")
		if err != nil {
			panic(err)
		}
		gW, gS, gB = ps[0].W, ps[0].S, ps[0].B
		uW, uS, uB = ps[1].W, ps[1].S, ps[1].B
		dW, dS, dB = ps[2].W, ps[2].S, ps[2].B
	}
	gs, bits := m.Store.groupSize, m.Store.bits
	hidden := shp[nd-1]
	// mlx-lm convention: x -> [..lead.., 1, 1, hidden] (double expand); each
	// gather_qmm yields [..lead.., K, 1, N]; squeeze the M=1 axis at the end.
	ridx := b.FromInt32(ri, append(append([]int{}, lead...), K)...)
	xe := b.ExpandDims(b.ExpandDims(x, nd-1), nd-1)                // [..lead.., 1, 1, hidden]
	g := b.GatherQuantMatmul(xe, gW, gS, gB, ridx, true, gs, bits) // [..lead.., K, 1, inter]
	u := b.GatherQuantMatmul(xe, uW, uS, uB, ridx, true, gs, bits)
	hh := b.Mul(b.SiLU(g), u)                                      // [..lead.., K, 1, inter]
	d := b.GatherQuantMatmul(hh, dW, dS, dB, ridx, true, gs, bits) // [..lead.., K, 1, hidden]
	// squeeze the M=1 axis -> [..lead.., K, hidden], then score-weighted sum
	// over the K experts (axis nd-1): mean(d*scores)*K.
	d2 := b.Reshape(d, append(append([]int{}, lead...), K, hidden)...)
	scT := b.FromFloats(scores, append(append([]int{}, lead...), K, 1)...)
	out := b.ScalarMul(b.Mean(b.Mul(d2, scT), nd-1), float32(K)) // [..lead.., hidden]

	// Decode (nTok==1): stream-free. Don't eval or free here — the heavy expert
	// matmuls stay lazy and batch into the single per-token eval, so there's no
	// per-layer GPU sync. The ~900 MB of one token's 48 stacks stays tracked and
	// is reclaimed by the decode loop's per-step Sweep. This is the hot path.
	//
	// Prefill (nTok>1): the lazy graph would otherwise hold all 48 layers' large
	// per-sequence stacks at once -> OOM. Eval per layer and free its stacks to
	// bound peak to one layer; a one-time cost on the prompt, not the hot path.
	if nTok > 1 {
		b.Eval(out)
		if f, ok := b.(engine.Freer); ok {
			f.Free(gW, gS, gB, uW, uS, uB, dW, dS, dB, g, u, hh, d)
		}
	}
	return out
}

func (s *ExpertStore) tensor(name string, e int) (engine.Tensor, int64, error) {
	dt, shape, raw, err := s.st.StackedRow(name, e)
	if err != nil {
		return nil, 0, err
	}
	edt, err := stDType(dt)
	if err != nil {
		return nil, 0, fmt.Errorf("expert %s: %w", name, err)
	}
	return s.b.FromRaw(edt, raw, shape...), int64(len(raw)), nil
}
