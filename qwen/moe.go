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
	return &ExpertStore{
		b: b, st: st, groupSize: groupSize, bits: bits, budget: budgetBytes,
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
	el := s.lru.PushFront(&exEntry{key: k, lin: lin, sz: sz})
	s.cache[k] = el
	s.bytes += sz
	for s.bytes > s.budget && s.lru.Len() > 3 {
		back := s.lru.Back()
		ent := back.Value.(*exEntry)
		s.lru.Remove(back)
		delete(s.cache, ent.key)
		s.bytes -= ent.sz
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
