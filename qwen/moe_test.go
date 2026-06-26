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
