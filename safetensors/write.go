package safetensors

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

// Entry is one tensor to serialize: the same (dtype, shape, raw) triple Get
// returns, so a Write/Get round trip is symmetric by construction.
type Entry struct {
	Dtype string
	Shape []int
	Raw   []byte
}

// Write serializes tensors (plus an optional __metadata__ string map) as a
// single-shard .safetensors file. Tensor bytes are laid out in sorted-name
// order for deterministic output. The write is atomic: staged to a temp file
// in the same directory, then renamed — a crash mid-write never leaves a
// truncated file where a reader (or the next boot) expects a valid one.
func Write(path string, tensors map[string]Entry, metadata map[string]string) error {
	type hdrEntry struct {
		Dtype       string   `json:"dtype"`
		Shape       []int    `json:"shape"`
		DataOffsets [2]int64 `json:"data_offsets"`
	}
	names := make([]string, 0, len(tensors))
	for n := range tensors {
		if n == "__metadata__" {
			return fmt.Errorf("safetensors: %q is a reserved name", n)
		}
		names = append(names, n)
	}
	sort.Strings(names)

	hdr := map[string]any{}
	if len(metadata) > 0 {
		hdr["__metadata__"] = metadata
	}
	var data bytes.Buffer
	for _, n := range names {
		e := tensors[n]
		if sz := dtypeSize(e.Dtype); sz == 0 {
			return fmt.Errorf("safetensors: tensor %q has unsupported dtype %q", n, e.Dtype)
		}
		start := int64(data.Len())
		data.Write(e.Raw)
		hdr[n] = hdrEntry{Dtype: e.Dtype, Shape: e.Shape, DataOffsets: [2]int64{start, int64(data.Len())}}
	}
	hdrJSON, err := json.Marshal(hdr)
	if err != nil {
		return fmt.Errorf("safetensors: marshal header: %w", err)
	}

	tmp, err := os.CreateTemp(filepath.Dir(path), ".st-write-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	if err := binary.Write(tmp, binary.LittleEndian, uint64(len(hdrJSON))); err == nil {
		if _, err = tmp.Write(hdrJSON); err == nil {
			_, err = tmp.Write(data.Bytes())
		}
	}
	if cerr := tmp.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return fmt.Errorf("safetensors: write %s: %w", path, err)
	}
	return os.Rename(tmp.Name(), path)
}
