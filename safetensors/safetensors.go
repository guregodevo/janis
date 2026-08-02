// Package safetensors is a minimal lazy reader for the .safetensors format:
// an 8-byte little-endian header length, a JSON header mapping tensor name ->
// {dtype, shape, data_offsets}, then the raw tensor bytes. Tensors are read on
// demand by byte range so we never load the whole multi-GB file.
//
// It also reads SHARDED models — the convention Hugging Face uses for large
// checkpoints: a model.safetensors.index.json whose "weight_map" names which
// model-XXXXX-of-XXXXX.safetensors file holds each tensor. Each shard is itself
// a normal safetensors file with its own header, so routing is by shard: open
// every shard once, merge their headers, and remember which shard each tensor
// lives in. Get/Close behave identically for single-file and sharded models.
package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

type entry struct {
	Dtype       string   `json:"dtype"`
	Shape       []int    `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
	shard       int      // index into File.shards holding this tensor's bytes
}

// shard is one backing .safetensors file: its handle plus the byte offset where
// the tensor data section begins (after the 8-byte length + JSON header).
type shard struct {
	f         *os.File
	dataStart int64
}

// File is an open safetensors model (one or more shards) with a parsed,
// merged index.
type File struct {
	shards   []shard
	index    map[string]entry
	metadata map[string]string
}

// Metadata returns the __metadata__ string map merged across shards (nil when
// none was present).
func (s *File) Metadata() map[string]string { return s.metadata }

// Names returns the indexed tensor names in sorted order.
func (s *File) Names() []string {
	names := make([]string, 0, len(s.index))
	for n := range s.index {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// Open parses a single .safetensors file's header and indexes its tensors
// without reading their data.
func Open(path string) (*File, error) {
	f := &File{index: map[string]entry{}}
	if err := f.addShard(path); err != nil {
		return nil, err
	}
	return f, nil
}

// OpenModel opens a model directory, transparently handling both a single
// model.safetensors and a sharded model (model.safetensors.index.json +
// model-XXXXX-of-XXXXX.safetensors). Callers use the same Get/Close API either
// way — this is the entry point model loaders should prefer over Open so large
// (sharded) checkpoints just work.
func OpenModel(dir string) (*File, error) {
	idxPath := filepath.Join(dir, "model.safetensors.index.json")
	if _, err := os.Stat(idxPath); err == nil {
		return openSharded(dir, idxPath)
	}
	return Open(filepath.Join(dir, "model.safetensors"))
}

// openSharded reads the index's weight_map to learn the set of shard files,
// then opens and merges each. The per-shard header (not the weight_map) is the
// authority on tensor offsets, so the map is used only to enumerate files.
func openSharded(dir, idxPath string) (*File, error) {
	raw, err := os.ReadFile(idxPath)
	if err != nil {
		return nil, err
	}
	var idx struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(raw, &idx); err != nil {
		return nil, fmt.Errorf("parse %s: %w", filepath.Base(idxPath), err)
	}
	if len(idx.WeightMap) == 0 {
		return nil, fmt.Errorf("%s: empty weight_map", filepath.Base(idxPath))
	}
	f := &File{index: map[string]entry{}}
	for _, name := range uniqueShardFiles(idx.WeightMap) {
		if err := f.addShard(filepath.Join(dir, name)); err != nil {
			f.Close()
			return nil, err
		}
	}
	return f, nil
}

// uniqueShardFiles returns the distinct shard filenames in deterministic order.
func uniqueShardFiles(weightMap map[string]string) []string {
	seen := map[string]bool{}
	var files []string
	for _, file := range weightMap {
		if !seen[file] {
			seen[file] = true
			files = append(files, file)
		}
	}
	sort.Strings(files)
	return files
}

// addShard opens one .safetensors file, parses its header, and merges its
// tensors into the index tagged with this shard's position.
func (f *File) addShard(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	var hdrLen uint64
	if err := binary.Read(file, binary.LittleEndian, &hdrLen); err != nil {
		file.Close()
		return fmt.Errorf("read header length: %w", err)
	}
	hdr := make([]byte, hdrLen)
	if _, err := file.ReadAt(hdr, 8); err != nil {
		file.Close()
		return fmt.Errorf("read header: %w", err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(hdr, &raw); err != nil {
		file.Close()
		return fmt.Errorf("parse header: %w", err)
	}
	si := len(f.shards)
	for name, msg := range raw {
		if name == "__metadata__" {
			// Retain the string map so callers can read invalidation headers
			// (e.g. the KV-snapshot's model identity). Values that aren't
			// plain strings are ignored, matching the safetensors spec.
			if f.metadata == nil {
				f.metadata = map[string]string{}
			}
			var md map[string]string
			if json.Unmarshal(msg, &md) == nil {
				for k, v := range md {
					f.metadata[k] = v
				}
			}
			continue
		}
		var e entry
		if err := json.Unmarshal(msg, &e); err != nil {
			file.Close()
			return fmt.Errorf("parse entry %q: %w", name, err)
		}
		if _, dup := f.index[name]; dup {
			file.Close()
			return fmt.Errorf("duplicate tensor %q across shards", name)
		}
		e.shard = si
		f.index[name] = e
	}
	f.shards = append(f.shards, shard{f: file, dataStart: int64(8 + hdrLen)})
	return nil
}

// Get returns the dtype, shape, and raw bytes of a named tensor, read on demand
// from whichever shard holds it.
func (s *File) Get(name string) (dtype string, shape []int, raw []byte, err error) {
	e, ok := s.index[name]
	if !ok {
		return "", nil, nil, fmt.Errorf("tensor %q not found", name)
	}
	n := e.DataOffsets[1] - e.DataOffsets[0]
	buf := make([]byte, n)
	sh := s.shards[e.shard]
	if _, err := sh.f.ReadAt(buf, sh.dataStart+e.DataOffsets[0]); err != nil {
		return "", nil, nil, fmt.Errorf("read tensor %q: %w", name, err)
	}
	return e.Dtype, e.Shape, buf, nil
}

// dtypeSize returns the byte width of a safetensors dtype.
func dtypeSize(dt string) int {
	switch dt {
	case "F32", "I32", "U32":
		return 4
	case "F16", "BF16", "I16", "U16":
		return 2
	case "I8", "U8":
		return 1
	case "F64", "I64", "U64":
		return 8
	}
	return 0
}

// StackedRow reads a single row (e.g. one MoE expert) out of a stacked tensor
// [nrows, ...] WITHOUT materializing the whole stack — it seeks to that row's
// contiguous byte range and reads only it. This is the primitive that lets the
// expert-offload engine touch ~8 of 128 experts per token instead of all of
// them. Returns the row's dtype, its shape (the stacked shape minus the leading
// axis), and the raw bytes.
func (s *File) StackedRow(name string, row int) (dtype string, rowShape []int, raw []byte, err error) {
	e, ok := s.index[name]
	if !ok {
		return "", nil, nil, fmt.Errorf("tensor %q not found", name)
	}
	if len(e.Shape) < 2 {
		return "", nil, nil, fmt.Errorf("tensor %q is not stacked (shape %v)", name, e.Shape)
	}
	if row < 0 || row >= e.Shape[0] {
		return "", nil, nil, fmt.Errorf("row %d out of range [0,%d) for %q", row, e.Shape[0], name)
	}
	esz := dtypeSize(e.Dtype)
	if esz == 0 {
		return "", nil, nil, fmt.Errorf("unsupported dtype %q for %q", e.Dtype, name)
	}
	per := 1
	for _, d := range e.Shape[1:] {
		per *= d
	}
	rowBytes := int64(per * esz)
	buf := make([]byte, rowBytes)
	sh := s.shards[e.shard]
	off := sh.dataStart + e.DataOffsets[0] + int64(row)*rowBytes
	if _, err := sh.f.ReadAt(buf, off); err != nil {
		return "", nil, nil, fmt.Errorf("read row %d of %q: %w", row, name, err)
	}
	return e.Dtype, append([]int(nil), e.Shape[1:]...), buf, nil
}

// Close releases every backing shard, returning the first close error.
func (s *File) Close() error {
	var firstErr error
	for _, sh := range s.shards {
		if sh.f == nil {
			continue
		}
		if err := sh.f.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}
