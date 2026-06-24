package safetensors

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// tensorData is a synthetic tensor for the test fixtures.
type tensorData struct {
	dtype string
	shape []int
	data  []byte
}

// writeST writes a minimal .safetensors file: 8-byte LE header length, JSON
// header (name -> {dtype, shape, data_offsets}), then concatenated tensor bytes.
func writeST(t *testing.T, path string, tensors map[string]tensorData) {
	t.Helper()
	type hdrEntry struct {
		Dtype       string   `json:"dtype"`
		Shape       []int    `json:"shape"`
		DataOffsets [2]int64 `json:"data_offsets"`
	}
	hdr := map[string]hdrEntry{}
	var data bytes.Buffer
	// Deterministic offsets regardless of map order: sort handled implicitly by
	// writing in a fixed iteration via a slice would be ideal, but offsets only
	// need to be internally consistent, so accumulate as we go.
	for name, td := range tensors {
		start := int64(data.Len())
		data.Write(td.data)
		hdr[name] = hdrEntry{Dtype: td.dtype, Shape: td.shape, DataOffsets: [2]int64{start, int64(data.Len())}}
	}
	hdrJSON, err := json.Marshal(hdr)
	if err != nil {
		t.Fatalf("marshal header: %v", err)
	}
	var buf bytes.Buffer
	if err := binary.Write(&buf, binary.LittleEndian, uint64(len(hdrJSON))); err != nil {
		t.Fatalf("write hdr len: %v", err)
	}
	buf.Write(hdrJSON)
	buf.Write(data.Bytes())
	if err := os.WriteFile(path, buf.Bytes(), 0644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func mustGet(t *testing.T, f *File, name string) []byte {
	t.Helper()
	_, _, raw, err := f.Get(name)
	if err != nil {
		t.Fatalf("Get(%q): %v", name, err)
	}
	return raw
}

func TestSingleFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.safetensors")
	want := map[string]tensorData{
		"a": {dtype: "F32", shape: []int{2}, data: []byte{1, 2, 3, 4}},
		"b": {dtype: "F16", shape: []int{3}, data: []byte{9, 8, 7, 6, 5, 4}},
	}
	writeST(t, path, want)

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer f.Close()
	for name, td := range want {
		if got := mustGet(t, f, name); !bytes.Equal(got, td.data) {
			t.Errorf("%s = %v, want %v", name, got, td.data)
		}
	}
	if _, _, _, err := f.Get("missing"); err == nil {
		t.Errorf("Get(missing) should error")
	}
}

func TestOpenModelSingleFile(t *testing.T) {
	dir := t.TempDir()
	writeST(t, filepath.Join(dir, "model.safetensors"), map[string]tensorData{
		"w": {dtype: "F32", shape: []int{1}, data: []byte{42, 0, 0, 0}},
	})
	f, err := OpenModel(dir)
	if err != nil {
		t.Fatalf("OpenModel: %v", err)
	}
	defer f.Close()
	if got := mustGet(t, f, "w"); !bytes.Equal(got, []byte{42, 0, 0, 0}) {
		t.Errorf("w = %v", got)
	}
}

func TestOpenModelSharded(t *testing.T) {
	dir := t.TempDir()
	// Two shards with disjoint tensors, each with its own data section so a naive
	// single-file offset would read the wrong bytes — this is what exercises the
	// per-shard routing.
	writeST(t, filepath.Join(dir, "model-00001-of-00002.safetensors"), map[string]tensorData{
		"layer.0.weight": {dtype: "F32", shape: []int{2}, data: []byte{10, 11, 12, 13}},
		"layer.1.weight": {dtype: "F32", shape: []int{1}, data: []byte{20, 21}},
	})
	writeST(t, filepath.Join(dir, "model-00002-of-00002.safetensors"), map[string]tensorData{
		"layer.2.weight": {dtype: "F16", shape: []int{3}, data: []byte{30, 31, 32, 33, 34, 35}},
	})
	index := map[string]any{
		"metadata": map[string]any{"total_size": 12},
		"weight_map": map[string]string{
			"layer.0.weight": "model-00001-of-00002.safetensors",
			"layer.1.weight": "model-00001-of-00002.safetensors",
			"layer.2.weight": "model-00002-of-00002.safetensors",
		},
	}
	idxJSON, _ := json.Marshal(index)
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), idxJSON, 0644); err != nil {
		t.Fatal(err)
	}

	f, err := OpenModel(dir)
	if err != nil {
		t.Fatalf("OpenModel sharded: %v", err)
	}
	defer f.Close()
	if len(f.shards) != 2 {
		t.Errorf("got %d shards, want 2", len(f.shards))
	}
	cases := map[string][]byte{
		"layer.0.weight": {10, 11, 12, 13},
		"layer.1.weight": {20, 21},
		"layer.2.weight": {30, 31, 32, 33, 34, 35}, // lives in the 2nd shard
	}
	for name, want := range cases {
		if got := mustGet(t, f, name); !bytes.Equal(got, want) {
			t.Errorf("%s = %v, want %v", name, got, want)
		}
	}
}

func TestShardedDuplicateTensorErrors(t *testing.T) {
	dir := t.TempDir()
	writeST(t, filepath.Join(dir, "a.safetensors"), map[string]tensorData{
		"dup": {dtype: "F32", shape: []int{1}, data: []byte{1, 2, 3, 4}},
	})
	writeST(t, filepath.Join(dir, "b.safetensors"), map[string]tensorData{
		"dup": {dtype: "F32", shape: []int{1}, data: []byte{5, 6, 7, 8}},
	})
	idxJSON, _ := json.Marshal(map[string]any{
		"weight_map": map[string]string{"dup": "a.safetensors", "dup2": "b.safetensors"},
	})
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), idxJSON, 0644); err != nil {
		t.Fatal(err)
	}
	if f, err := OpenModel(dir); err == nil {
		f.Close()
		t.Errorf("expected duplicate-tensor error, got nil")
	}
}

func TestStackedRow(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.safetensors")
	// 4 rows x 6 float32, values 0..23
	vals := make([]float32, 24)
	for i := range vals {
		vals[i] = float32(i)
	}
	buf := new(bytes.Buffer)
	_ = binary.Write(buf, binary.LittleEndian, vals)
	writeST(t, path, map[string]tensorData{
		"stack": {dtype: "F32", shape: []int{4, 6}, data: buf.Bytes()},
	})
	f, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	for row := 0; row < 4; row++ {
		dt, shape, raw, err := f.StackedRow("stack", row)
		if err != nil {
			t.Fatalf("StackedRow(%d): %v", row, err)
		}
		if dt != "F32" || len(shape) != 1 || shape[0] != 6 {
			t.Fatalf("row %d: dt=%s shape=%v", row, dt, shape)
		}
		got := make([]float32, 6)
		_ = binary.Read(bytes.NewReader(raw), binary.LittleEndian, got)
		for j := 0; j < 6; j++ {
			if want := float32(row*6 + j); got[j] != want {
				t.Errorf("row %d elem %d = %v, want %v", row, j, got[j], want)
			}
		}
	}
	if _, _, _, err := f.StackedRow("stack", 4); err == nil {
		t.Error("expected out-of-range error for row 4")
	}
	if _, _, _, err := f.StackedRow("stack", -1); err == nil {
		t.Error("expected out-of-range error for row -1")
	}
}

func TestShardedMissingShardFile(t *testing.T) {
	dir := t.TempDir()
	// index references a shard file that doesn't exist on disk.
	idxJSON, _ := json.Marshal(map[string]any{
		"weight_map": map[string]string{"w": "model-00001-of-00001.safetensors"},
	})
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), idxJSON, 0644); err != nil {
		t.Fatal(err)
	}
	if f, err := OpenModel(dir); err == nil {
		f.Close()
		t.Errorf("expected error for missing shard file, got nil")
	}
}

func TestShardedMalformedIndex(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), []byte("{not json"), 0644); err != nil {
		t.Fatal(err)
	}
	if f, err := OpenModel(dir); err == nil {
		f.Close()
		t.Errorf("expected error for malformed index.json, got nil")
	}
}

func TestShardedEmptyWeightMap(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), []byte(`{"weight_map":{}}`), 0644); err != nil {
		t.Fatal(err)
	}
	if f, err := OpenModel(dir); err == nil {
		f.Close()
		t.Errorf("expected error for empty weight_map, got nil")
	}
}

// TestOpenModelMissingEntirely: no index.json and no model.safetensors → error,
// not a panic (a misconfigured/empty model dir).
func TestOpenModelMissingEntirely(t *testing.T) {
	if f, err := OpenModel(t.TempDir()); err == nil {
		f.Close()
		t.Errorf("expected error for a dir with no model files, got nil")
	}
}

// TestThreeShards exercises >2 shards and confirms tensors route to the correct
// (third) shard, not just the first two.
func TestThreeShards(t *testing.T) {
	dir := t.TempDir()
	writeST(t, filepath.Join(dir, "model-00001-of-00003.safetensors"), map[string]tensorData{
		"a": {dtype: "F32", shape: []int{1}, data: []byte{1, 1, 1, 1}},
	})
	writeST(t, filepath.Join(dir, "model-00002-of-00003.safetensors"), map[string]tensorData{
		"b": {dtype: "F32", shape: []int{1}, data: []byte{2, 2, 2, 2}},
	})
	writeST(t, filepath.Join(dir, "model-00003-of-00003.safetensors"), map[string]tensorData{
		"c": {dtype: "F32", shape: []int{1}, data: []byte{3, 3, 3, 3}},
	})
	idxJSON, _ := json.Marshal(map[string]any{"weight_map": map[string]string{
		"a": "model-00001-of-00003.safetensors",
		"b": "model-00002-of-00003.safetensors",
		"c": "model-00003-of-00003.safetensors",
	}})
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), idxJSON, 0644); err != nil {
		t.Fatal(err)
	}
	f, err := OpenModel(dir)
	if err != nil {
		t.Fatalf("OpenModel 3-shard: %v", err)
	}
	defer f.Close()
	if len(f.shards) != 3 {
		t.Errorf("got %d shards, want 3", len(f.shards))
	}
	for name, want := range map[string][]byte{"a": {1, 1, 1, 1}, "b": {2, 2, 2, 2}, "c": {3, 3, 3, 3}} {
		if got := mustGet(t, f, name); !bytes.Equal(got, want) {
			t.Errorf("%s = %v, want %v", name, got, want)
		}
	}
}
