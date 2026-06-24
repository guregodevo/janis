package qwen

import (
	"container/list"
	"fmt"

	"memdoor/llm/engine"
	"memdoor/llm/safetensors"
)

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
}

// NewExpertStore builds a store over st, bounding resident expert weights to
// budgetBytes (older experts are evicted LRU once exceeded).
func NewExpertStore(b engine.Backend, st *safetensors.File, groupSize, bits int, budgetBytes int64) *ExpertStore {
	sw, _ := b.(engine.Sweeper)
	return &ExpertStore{
		b: b, sw: sw, st: st, groupSize: groupSize, bits: bits, budget: budgetBytes,
		cache: map[exKey]*list.Element{}, lru: list.New(),
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
	return lin, nil
}

// Resident reports the current resident expert-weight byte count.
func (s *ExpertStore) Resident() int64 { return s.bytes }

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
	router, err := LoadQuantLinear(b, st, prefix, cfg.GroupSize, cfg.Bits, false)
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
	shp := x.Shape()
	hidden := shp[len(shp)-1]
	seq := 1
	for _, d := range shp[:len(shp)-1] {
		seq *= d
	}
	x2 := x
	if len(shp) != 2 {
		x2 = b.Reshape(x, seq, hidden)
	}

	probs := b.Softmax(m.Router.Forward(b, x2), -1) // [seq, numExperts]
	pf := b.Floats(probs)

	var out engine.Tensor
	for t := 0; t < seq; t++ {
		idx, sc := topK(pf[t*m.NumExperts:(t+1)*m.NumExperts], m.TopK)
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
		xt := b.Slice(x2, 0, t, t+1) // [1, hidden]
		var acc engine.Tensor
		for j, e := range idx {
			ge, err := m.Store.Expert(m.Layer, "gate_proj", e)
			if err != nil {
				panic(err)
			}
			ue, err := m.Store.Expert(m.Layer, "up_proj", e)
			if err != nil {
				panic(err)
			}
			de, err := m.Store.Expert(m.Layer, "down_proj", e)
			if err != nil {
				panic(err)
			}
			h := b.Mul(b.SiLU(ge.Forward(b, xt)), ue.Forward(b, xt))
			ye := b.ScalarMul(de.Forward(b, h), sc[j]) // [1, hidden]
			if acc == nil {
				acc = ye
			} else {
				acc = b.Add(acc, ye)
			}
		}
		if out == nil {
			out = acc
		} else {
			out = b.Concat(out, acc, 0)
		}
	}
	if len(shp) != 2 {
		out = b.Reshape(out, shp...)
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
