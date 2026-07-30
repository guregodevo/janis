package qwen

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"memdoor/llm/cpu"
	"memdoor/llm/safetensors"
)

type synTensor struct {
	dt    string
	shape []int
	data  []byte
}

func writeSyntheticST(t *testing.T, path string, ts map[string]synTensor) {
	t.Helper()
	type hdrE struct {
		Dtype       string   `json:"dtype"`
		Shape       []int    `json:"shape"`
		DataOffsets [2]int64 `json:"data_offsets"`
	}
	hdr := map[string]hdrE{}
	var data bytes.Buffer
	for name, td := range ts {
		start := int64(data.Len())
		data.Write(td.data)
		hdr[name] = hdrE{td.dt, td.shape, [2]int64{start, int64(data.Len())}}
	}
	hj, _ := json.Marshal(hdr)
	out := new(bytes.Buffer)
	_ = binary.Write(out, binary.LittleEndian, uint64(len(hj)))
	out.Write(hj)
	out.Write(data.Bytes())
	if err := os.WriteFile(path, out.Bytes(), 0644); err != nil {
		t.Fatal(err)
	}
}

// TestExpertStore checks the offload store's new logic: a miss materializes and
// caches, a repeat is a hit, and the LRU evicts once the byte budget is exceeded
// (keeping a minimum floor). Uses the pure-Go CPU backend so it runs anywhere.
func TestExpertStore(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.safetensors")
	const N = 8 // experts
	// gate_proj for layer 0: weight U32 [N,2,1], scales/biases F16 [N,2,1]
	mk := func(n, sz int) []byte { return make([]byte, n*sz) }
	writeSyntheticST(t, path, map[string]synTensor{
		"model.layers.0.mlp.switch_mlp.gate_proj.weight": {"U32", []int{N, 2, 1}, mk(N*2, 4)},
		"model.layers.0.mlp.switch_mlp.gate_proj.scales": {"F16", []int{N, 2, 1}, mk(N*2, 2)},
		"model.layers.0.mlp.switch_mlp.gate_proj.biases": {"F16", []int{N, 2, 1}, mk(N*2, 2)},
	})

	st, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	b := cpu.New()

	// per expert: weight 8B + scales 4B + biases 4B = 16B. Budget ~50B -> ~3 fit.
	store := NewExpertStore(b, st, 64, 4, 50)

	// miss then hit on the same expert
	if _, err := store.Expert(0, "gate_proj", 2); err != nil {
		t.Fatalf("Expert: %v", err)
	}
	if _, err := store.Expert(0, "gate_proj", 2); err != nil {
		t.Fatalf("Expert(repeat): %v", err)
	}
	if store.Misses != 1 || store.Hits != 1 {
		t.Errorf("after miss+repeat: hits=%d misses=%d, want 1/1", store.Hits, store.Misses)
	}

	// load all N distinct experts -> cache must stay bounded (LRU eviction)
	for e := 0; e < N; e++ {
		if _, err := store.Expert(0, "gate_proj", e); err != nil {
			t.Fatalf("Expert(%d): %v", e, err)
		}
	}
	if store.lru.Len() > 4 {
		t.Errorf("cache not bounded: lru.Len()=%d (budget should evict to ~3)", store.lru.Len())
	}
	if store.Resident() > 64 {
		t.Errorf("resident %d B exceeds budget+slack", store.Resident())
	}
	if len(store.cache) != store.lru.Len() {
		t.Errorf("cache map (%d) and lru (%d) out of sync", len(store.cache), store.lru.Len())
	}
}

// TestReadRowsParallel checks that the bounded-parallel read path preserves
// expert order in the stacked rows, counts bytes, and surfaces read errors.
func TestReadRowsParallel(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.safetensors")
	const N = 32
	// Each row of "stack" is 4 bytes; row e is filled with byte(e) so order
	// scrambling by the goroutines would be visible in the assembled rows.
	data := make([]byte, N*4)
	for e := 0; e < N; e++ {
		for j := 0; j < 4; j++ {
			data[e*4+j] = byte(e)
		}
	}
	writeSyntheticST(t, path, map[string]synTensor{
		"stack": {"U8", []int{N, 4}, data},
	})

	st, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	store := NewExpertStore(cpu.New(), st, 64, 4, 1<<20)

	experts := []int{7, 0, 31, 15, 3, 22, 9, 1, 28, 4, 18, 11}
	rows, err := store.readRows("stack", experts)
	if err != nil {
		t.Fatalf("readRows: %v", err)
	}
	if len(rows.rows) != len(experts) {
		t.Fatalf("got %d rows, want %d", len(rows.rows), len(experts))
	}
	for i, e := range experts {
		for _, b := range rows.rows[i] {
			if b != byte(e) {
				t.Fatalf("row %d (expert %d): got byte %d, want %d — order not preserved", i, e, b, e)
			}
		}
	}
	if got, _ := store.IOStats(); got != int64(len(experts)*4) {
		t.Errorf("IOStats bytes = %d, want %d", got, len(experts)*4)
	}

	if _, err := store.readRows("stack", []int{0, N + 5}); err == nil {
		t.Error("out-of-range expert should surface a read error")
	}
}

// TestEnsureCached checks that one call materializes all three projections of
// the requested experts into the cache, that a following Stack is pure hits,
// and that a second EnsureCached for the same experts reads nothing new.
func TestEnsureCached(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.safetensors")
	const N = 4
	ts := map[string]synTensor{}
	for _, p := range []string{"gate_proj", "up_proj", "down_proj"} {
		base := "model.layers.0.mlp.switch_mlp." + p
		ts[base+".weight"] = synTensor{"U32", []int{N, 2, 1}, make([]byte, N*2*4)}
		ts[base+".scales"] = synTensor{"F16", []int{N, 2, 1}, make([]byte, N*2*2)}
		ts[base+".biases"] = synTensor{"F16", []int{N, 2, 1}, make([]byte, N*2*2)}
	}
	writeSyntheticST(t, path, ts)

	st, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	store := NewExpertStore(cpu.New(), st, 64, 4, 1<<20)

	experts := []int{2, 0}
	if err := store.EnsureCached(0, experts); err != nil {
		t.Fatalf("EnsureCached: %v", err)
	}
	if store.Misses != len(experts)*3 {
		t.Errorf("misses = %d, want %d (3 projections per expert)", store.Misses, len(experts)*3)
	}
	bytesAfterFill, _ := store.IOStats()

	if _, _, _, err := store.Stack(0, "gate_proj", experts); err != nil {
		t.Fatalf("Stack: %v", err)
	}
	if store.Hits != len(experts) {
		t.Errorf("hits after Stack = %d, want %d", store.Hits, len(experts))
	}

	if err := store.EnsureCached(0, experts); err != nil {
		t.Fatalf("EnsureCached(repeat): %v", err)
	}
	if b, _ := store.IOStats(); b != bytesAfterFill {
		t.Errorf("repeat EnsureCached read %d new bytes, want 0", b-bytesAfterFill)
	}
}

// TestSlotStack checks the persistent slot cache: first touch misses and
// scatters rows in, a repeat is all hits with stable slot indices, a new
// expert evicts the least-frequently-used victim, and the slot tensors carry
// the right rows (verified through the CPU backend's exact U32 words).
func TestSlotStack(t *testing.T) {
	t.Setenv("MEMDOOR_MOE_SLOTS", "3")
	dir := t.TempDir()
	path := filepath.Join(dir, "model.safetensors")
	const N = 8
	ts := map[string]synTensor{}
	for _, p := range []string{"gate_proj", "up_proj", "down_proj"} {
		base := "model.layers.0.mlp.switch_mlp." + p
		// weight row e = two U32 words of value e, so slot contents are checkable
		wdata := make([]byte, N*2*4)
		for e := 0; e < N; e++ {
			for w := 0; w < 2; w++ {
				wdata[e*8+w*4] = byte(e)
			}
		}
		ts[base+".weight"] = synTensor{"U32", []int{N, 2, 1}, wdata}
		ts[base+".scales"] = synTensor{"F16", []int{N, 2, 1}, make([]byte, N*2*2)}
		ts[base+".biases"] = synTensor{"F16", []int{N, 2, 1}, make([]byte, N*2*2)}
	}
	writeSyntheticST(t, path, ts)

	st, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	b := cpu.New()
	store := NewExpertStore(b, st, 64, 4, 1<<20)

	// First touch: both experts miss and land in slots.
	projs, idx, ok, err := store.SlotStack(0, []int{5, 2})
	if err != nil || !ok {
		t.Fatalf("SlotStack: ok=%v err=%v", ok, err)
	}
	if store.Misses != 2 || store.Hits != 0 {
		t.Errorf("first touch: hits=%d misses=%d, want 0/2", store.Hits, store.Misses)
	}

	// Repeat: pure hits, same slots.
	_, idx2, ok, err := store.SlotStack(0, []int{5, 2})
	if err != nil || !ok {
		t.Fatalf("SlotStack(repeat): ok=%v err=%v", ok, err)
	}
	if store.Hits != 2 {
		t.Errorf("repeat: hits=%d, want 2", store.Hits)
	}
	if idx2[0] != idx[0] || idx2[1] != idx[1] {
		t.Errorf("slot indices changed on repeat: %v -> %v", idx, idx2)
	}

	// Slot contents: expert 5's weight row (words == 5) sits at slot idx[0].
	w := projs[0].W
	raw := b.Floats(w) // CPU backend mirrors U32 words into f32
	rowLen := 2
	if got := raw[int(idx[0])*rowLen]; got != 5 {
		t.Errorf("slot %d word = %v, want 5", idx[0], got)
	}
	if got := raw[int(idx2[1])*rowLen]; got != 2 {
		t.Errorf("slot %d word = %v, want 2", idx2[1], got)
	}

	// Third expert fills the last free slot; fourth evicts the LFU victim.
	if _, _, _, err := store.SlotStack(0, []int{7}); err != nil {
		t.Fatal(err)
	}
	if _, idx4, _, err := store.SlotStack(0, []int{6}); err != nil {
		t.Fatal(err)
	} else if idx4[0] != 2 {
		// slots 0/1 hold experts 5/2 (freq 2 each); slot 2 holds 7 (freq 1)
		t.Errorf("eviction picked slot %d, want 2 (LFU)", idx4[0])
	}
}

// TestSlotHits checks the read-only cache probe prefill uses: hits map to
// their slots, misses are -1, and probing mutates nothing — no insertions,
// no evictions, no reads.
func TestSlotHits(t *testing.T) {
	t.Setenv("MEMDOOR_MOE_SLOTS", "4")
	dir := t.TempDir()
	path := filepath.Join(dir, "model.safetensors")
	const N = 8
	ts := map[string]synTensor{}
	for _, p := range []string{"gate_proj", "up_proj", "down_proj"} {
		base := "model.layers.0.mlp.switch_mlp." + p
		ts[base+".weight"] = synTensor{"U32", []int{N, 2, 1}, make([]byte, N*2*4)}
		ts[base+".scales"] = synTensor{"F16", []int{N, 2, 1}, make([]byte, N*2*2)}
		ts[base+".biases"] = synTensor{"F16", []int{N, 2, 1}, make([]byte, N*2*2)}
	}
	writeSyntheticST(t, path, ts)
	st, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	store := NewExpertStore(cpu.New(), st, 64, 4, 1<<20)

	// No cache yet: not ok.
	if _, _, ok := store.SlotHits(0, []int{1, 2}); ok {
		t.Fatal("SlotHits should report ok=false before any SlotStack")
	}

	// Warm experts 5 and 2 via the decode path, then probe a mixed set.
	if _, _, _, err := store.SlotStack(0, []int{5, 2}); err != nil {
		t.Fatal(err)
	}
	bytesBefore, _ := store.IOStats()
	_, idx, ok := store.SlotHits(0, []int{5, 7, 2})
	if !ok {
		t.Fatal("SlotHits should be ok after SlotStack warmed the layer")
	}
	if idx[0] < 0 || idx[2] < 0 {
		t.Errorf("experts 5 and 2 should be hits, got %v", idx)
	}
	if idx[1] != -1 {
		t.Errorf("expert 7 should be a miss (-1), got %d", idx[1])
	}
	if b, _ := store.IOStats(); b != bytesBefore {
		t.Errorf("SlotHits read %d bytes; must be read-free", b-bytesBefore)
	}
	// The miss must NOT have been inserted.
	if _, idx2, _ := store.SlotHits(0, []int{7}); idx2[0] != -1 {
		t.Error("SlotHits must not insert misses into the cache")
	}
}

// TestStackProjs checks the fused multi-projection fetch: one call yields
// stacked W/S/B tensors per projection with the leading axis sized to the
// active expert count.
func TestStackProjs(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.safetensors")
	const N = 4
	ts := map[string]synTensor{}
	for _, p := range []string{"gate_proj", "up_proj", "down_proj"} {
		base := "model.layers.0.mlp.switch_mlp." + p
		ts[base+".weight"] = synTensor{"U32", []int{N, 2, 1}, make([]byte, N*2*4)}
		ts[base+".scales"] = synTensor{"F16", []int{N, 2, 1}, make([]byte, N*2*2)}
		ts[base+".biases"] = synTensor{"F16", []int{N, 2, 1}, make([]byte, N*2*2)}
	}
	writeSyntheticST(t, path, ts)

	st, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	store := NewExpertStore(cpu.New(), st, 64, 4, 1<<20)

	experts := []int{3, 1}
	ps, err := store.StackProjs(0, experts, "gate_proj", "up_proj", "down_proj")
	if err != nil {
		t.Fatalf("StackProjs: %v", err)
	}
	if len(ps) != 3 {
		t.Fatalf("got %d projections, want 3", len(ps))
	}
	for i, p := range ps {
		if p.W == nil || p.S == nil || p.B == nil {
			t.Fatalf("projection %d has nil tensors", i)
		}
		if shp := p.W.Shape(); shp[0] != len(experts) {
			t.Errorf("projection %d leading axis = %d, want %d", i, shp[0], len(experts))
		}
	}
	if bytes, _ := store.IOStats(); bytes == 0 {
		t.Error("IOStats should count the fused fetch")
	}
}
