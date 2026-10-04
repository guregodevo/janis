package qwen

import (
	"fmt"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/guregodevo/janis/engine"
)

// slotCacheEnabled gates the decode-path expert slot cache — on by default
// (measured 2.0 → 3.7 tok/s on a 16 GB M4 streaming Qwen3-30B-A3B), with
// JANIS_MOE_SLOT_CACHE=0 as the escape hatch back to transient stacks.
func slotCacheEnabled() bool { return os.Getenv("JANIS_MOE_SLOT_CACHE") != "0" }

// asyncOverlapEnabled gates the decode IO/compute overlap (hit gather
// async-evaluated while miss reads run). Opt-in while being validated:
// JANIS_MOE_ASYNC_OVERLAP=1.
func asyncOverlapEnabled() bool { return os.Getenv("JANIS_MOE_ASYNC_OVERLAP") == "1" }

// slotsPerLayer is the expert-slot count per MoE layer. 16 slots hold two
// tokens' top-8 sets; with 40-60% token-to-token expert reuse that converts
// most of the overlap into cache hits while pinning only
// slots × layers × expert-size bytes (~2 GB on a 48-layer 30B-A3B).
// JANIS_MOE_SLOTS overrides for tuning.
func slotsPerLayer() int {
	if n, _ := strconv.Atoi(os.Getenv("JANIS_MOE_SLOTS")); n > 0 {
		return n
	}
	return 16
}

// layerSlots is one MoE layer's expert slot cache: three persistent stacked
// tensors per projection (weight/scales/biases, leading axis = slot) plus
// host-side LFU bookkeeping. Missed experts land via one batched ScatterRows
// per tensor, so the op count per token stays flat regardless of hit rate.
type layerSlots struct {
	projs      [3]StackedProj // gate, up, down; leading axis = slot
	expertSlot map[int]int
	slotExpert []int // -1 = free
	freq       []int64
	lastUse    []int64
	tick       int64
}

// SlotStack resolves the given experts to slot indices in the layer's
// persistent slot tensors, reading and scattering in any missing experts.
// Returns the three projections' slot tensors and the slot index per expert
// (aligned with the experts argument). ok is false when the backend cannot
// scatter rows (caller falls back to transient stacks).
func (s *ExpertStore) SlotStack(layer int, experts []int) (projs [3]StackedProj, slotIdx []int32, ok bool, err error) {
	return s.slotStackFill(layer, experts, true)
}

// slotStackFill is SlotStack's implementation; countHits=false suppresses hit
// accounting for callers that already counted them via SlotHits (the async-
// overlap decode path probes first, then fills).
func (s *ExpertStore) slotStackFill(layer int, experts []int, countHits bool) (projs [3]StackedProj, slotIdx []int32, ok bool, err error) {
	sc, scOK := s.b.(engine.RowScatterer)
	if !scOK {
		return projs, nil, false, nil
	}
	nSlots := slotsPerLayer()
	if len(experts) > nSlots {
		return projs, nil, false, nil // a single token needs more than the cache holds
	}

	if s.slots == nil {
		s.slots = map[int]*layerSlots{}
	}
	ls := s.slots[layer]

	// Split hits from misses against the current slot map.
	slotIdx = make([]int32, len(experts))
	var missPos []int // positions in experts that need a read
	if ls != nil {
		ls.tick++
		for i, e := range experts {
			if sl, hit := ls.expertSlot[e]; hit {
				slotIdx[i] = int32(sl)
				ls.freq[sl]++
				ls.lastUse[sl] = ls.tick
				if countHits {
					s.Hits++
				}
			} else {
				missPos = append(missPos, i)
			}
		}
	} else {
		for i := range experts {
			missPos = append(missPos, i)
		}
	}
	if len(missPos) == 0 {
		return ls.projs, slotIdx, true, nil
	}

	missExperts := make([]int, len(missPos))
	for j, p := range missPos {
		missExperts[j] = experts[p]
	}

	// One concurrent read phase for all nine tensors' missing rows.
	var reads [3][3]stackedRows
	var errs [3][3]error
	var wg sync.WaitGroup
	start := time.Now()
	for pi, proj := range moeProjs {
		base := fmt.Sprintf("model.layers.%d.mlp.switch_mlp.%s", layer, proj)
		for ti, suffix := range [3]string{".weight", ".scales", ".biases"} {
			wg.Add(1)
			go func(pi, ti int, name string) {
				defer wg.Done()
				reads[pi][ti], errs[pi][ti] = s.readRows(name, missExperts)
			}(pi, ti, base+suffix)
		}
	}
	wg.Wait()
	s.readNanos.Add(time.Since(start).Nanoseconds())
	for pi := range moeProjs {
		for ti := range errs[pi] {
			if e := errs[pi][ti]; e != nil {
				return projs, nil, false, e
			}
		}
	}

	if ls == nil {
		ls = s.newLayerSlots(nSlots, reads)
		s.slots[layer] = ls
		ls.tick = 1
	}

	// Assign a victim slot per miss: free slots first, then the least
	// frequently used (oldest on ties), never one this token already holds.
	inUse := map[int]bool{}
	for _, e := range experts {
		if sl, hit := ls.expertSlot[e]; hit {
			inUse[sl] = true
		}
	}
	victims := make([]int32, len(missPos))
	for j := range missPos {
		v := ls.pickVictim(inUse)
		inUse[v] = true
		victims[j] = int32(v)
		if old := ls.slotExpert[v]; old >= 0 {
			delete(ls.expertSlot, old)
		}
		e := missExperts[j]
		ls.slotExpert[v] = e
		ls.expertSlot[e] = v
		ls.freq[v] = 1
		ls.lastUse[v] = ls.tick
		slotIdx[missPos[j]] = victims[j]
		s.Misses++
	}

	// One batched scatter per tensor lands every missed row; the replaced
	// slot-tensor handles are re-pinned so the decode loop's Sweep keeps them.
	for pi := range moeProjs {
		up := [3]*engine.Tensor{&ls.projs[pi].W, &ls.projs[pi].S, &ls.projs[pi].B}
		for ti, dst := range up {
			r := reads[pi][ti]
			edt, derr := stDType(r.dt)
			if derr != nil {
				return projs, nil, false, derr
			}
			var buf []byte
			for _, row := range r.rows {
				buf = append(buf, row...)
			}
			updates := s.b.FromRaw(edt, buf, append([]int{len(missExperts)}, r.rowShape...)...)
			old := *dst
			*dst = sc.ScatterRows(old, victims, updates)
			if s.sw != nil {
				s.sw.Pin(*dst)
				s.sw.Unpin(old)
			}
		}
	}
	return ls.projs, slotIdx, true, nil
}

// SlotHits reports which of the given experts are resident in the layer's slot
// cache, WITHOUT mutating the cache: no insertion, no eviction, no LFU decay —
// prefill uses it to read only its cache misses while decode's hot set stays
// intact. Returns the slot tensors, a per-expert slot index (-1 = miss), and
// ok=false when the layer has no slot cache yet.
func (s *ExpertStore) SlotHits(layer int, experts []int) (projs [3]StackedProj, slotIdx []int32, ok bool) {
	ls := s.slots[layer]
	if ls == nil {
		return projs, nil, false
	}
	slotIdx = make([]int32, len(experts))
	for i, e := range experts {
		if sl, hit := ls.expertSlot[e]; hit {
			slotIdx[i] = int32(sl)
			s.Hits++
		} else {
			slotIdx[i] = -1
		}
	}
	return ls.projs, slotIdx, true
}

// newLayerSlots allocates the persistent slot tensors for one layer, shaped
// from the first read batch and zero-filled.
func (s *ExpertStore) newLayerSlots(nSlots int, reads [3][3]stackedRows) *layerSlots {
	ls := &layerSlots{
		expertSlot: map[int]int{},
		slotExpert: make([]int, nSlots),
		freq:       make([]int64, nSlots),
		lastUse:    make([]int64, nSlots),
	}
	for i := range ls.slotExpert {
		ls.slotExpert[i] = -1
	}
	for pi := range moeProjs {
		ts := [3]*engine.Tensor{&ls.projs[pi].W, &ls.projs[pi].S, &ls.projs[pi].B}
		for ti, dst := range ts {
			r := reads[pi][ti]
			edt, err := stDType(r.dt)
			if err != nil {
				panic(err) // dtype already validated by the read
			}
			rowBytes := len(r.rows[0])
			*dst = s.b.FromRaw(edt, make([]byte, nSlots*rowBytes), append([]int{nSlots}, r.rowShape...)...)
			if s.sw != nil {
				s.sw.Pin(*dst)
			}
		}
	}
	return ls
}

// pickVictim returns a free slot if any, else the least frequently used slot
// (least recently used on ties), skipping slots in use by the current token.
func (ls *layerSlots) pickVictim(inUse map[int]bool) int {
	best := -1
	for i, e := range ls.slotExpert {
		if inUse[i] {
			continue
		}
		if e < 0 {
			return i
		}
		if best < 0 || ls.freq[i] < ls.freq[best] ||
			(ls.freq[i] == ls.freq[best] && ls.lastUse[i] < ls.lastUse[best]) {
			best = i
		}
	}
	return best
}
