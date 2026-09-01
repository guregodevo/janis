package qwen

import (
	"path/filepath"
	"testing"

	"memdoor/llm/cpu"
	"memdoor/llm/engine"
)

// TestSessionSaveRestoreRoundtrip verifies that a session's KV prefix survives
// a disk round trip byte-for-byte on the CPU backend: same IDs, same shapes,
// same values, and that model-identity / layer-count mismatches refuse to load.
func TestSessionSaveRestoreRoundtrip(t *testing.T) {
	b := cpu.New()
	const nLayers, T, heads, hd = 2, 5, 3, 4

	mk := func(seed float32) engine.Tensor {
		data := make([]float32, 1*heads*T*hd)
		for i := range data {
			data[i] = seed + float32(i)*0.5
		}
		return b.FromFloats(data, 1, heads, T, hd)
	}
	src := &Session{NLayers: nLayers, IDs: []int32{11, 22, 33, 44, 55}}
	for i := 0; i < nLayers; i++ {
		src.Caches = append(src.Caches, &KVCache{K: mk(float32(i)), V: mk(float32(i) + 100)})
	}

	path := filepath.Join(t.TempDir(), "kv.safetensors")
	if err := src.Save(b, path, "model-A"); err != nil {
		t.Fatalf("save: %v", err)
	}

	dst := &Session{NLayers: nLayers}
	n, err := dst.Restore(b, path, "model-A")
	if err != nil {
		t.Fatalf("restore: %v", err)
	}
	if n != T {
		t.Fatalf("restored %d tokens, want %d", n, T)
	}
	for i, want := range src.IDs {
		if dst.IDs[i] != want {
			t.Fatalf("ids[%d] = %d, want %d", i, dst.IDs[i], want)
		}
	}
	for l := 0; l < nLayers; l++ {
		srcKV, dstKV := src.Caches[l].(*KVCache), dst.Caches[l].(*KVCache)
		for name, pair := range map[string][2]engine.Tensor{
			"k": {srcKV.K, dstKV.K},
			"v": {srcKV.V, dstKV.V},
		} {
			want, got := b.Floats(pair[0]), b.Floats(pair[1])
			if len(want) != len(got) {
				t.Fatalf("layer %d %s: size %d, want %d", l, name, len(got), len(want))
			}
			for i := range want {
				if want[i] != got[i] {
					t.Fatalf("layer %d %s[%d] = %v, want %v", l, name, i, got[i], want[i])
				}
			}
			if gs, ws := pair[1].Shape(), pair[0].Shape(); len(gs) != len(ws) {
				t.Fatalf("layer %d %s shape %v, want %v", l, name, gs, ws)
			}
		}
	}

	// Wrong model identity must refuse.
	if _, err := (&Session{NLayers: nLayers}).Restore(b, path, "model-B"); err == nil {
		t.Fatal("restore with wrong model id should fail")
	}
	// Wrong layer count must refuse.
	if _, err := (&Session{NLayers: nLayers + 1}).Restore(b, path, "model-A"); err == nil {
		t.Fatal("restore with wrong layer count should fail")
	}
	// Restoring into a non-fresh session must refuse.
	if _, err := dst.Restore(b, path, "model-A"); err == nil {
		t.Fatal("restore into a warm session should fail")
	}
}

// TestSessionSaveTruncatesUnforwardedID: IDs may be one longer than the
// materialized cache (the last sampled token is never forwarded); Save must
// persist only the materialized prefix.
func TestSessionSaveTruncatesUnforwardedID(t *testing.T) {
	b := cpu.New()
	const T = 4
	data := make([]float32, 1*2*T*3)
	s := &Session{
		NLayers: 1,
		IDs:     []int32{1, 2, 3, 4, 5}, // one more than the cache holds
		Caches:  []LayerCache{&KVCache{K: b.FromFloats(data, 1, 2, T, 3), V: b.FromFloats(data, 1, 2, T, 3)}},
	}
	path := filepath.Join(t.TempDir(), "kv.safetensors")
	if err := s.Save(b, path, "m"); err != nil {
		t.Fatalf("save: %v", err)
	}
	dst := &Session{NLayers: 1}
	n, err := dst.Restore(b, path, "m")
	if err != nil {
		t.Fatalf("restore: %v", err)
	}
	if n != T || len(dst.IDs) != T {
		t.Fatalf("restored %d ids, want %d", len(dst.IDs), T)
	}
}
