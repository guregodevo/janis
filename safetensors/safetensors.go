// Package safetensors is a minimal lazy reader for the .safetensors format:
// an 8-byte little-endian header length, a JSON header mapping tensor name ->
// {dtype, shape, data_offsets}, then the raw tensor bytes. Tensors are read on
// demand by byte range so we never load the whole multi-GB file.
package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
)

type entry struct {
	Dtype       string   `json:"dtype"`
	Shape       []int    `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

// File is an open safetensors file with a parsed index.
type File struct {
	f         *os.File
	dataStart int64
	index     map[string]entry
}

// Open parses the header and indexes the tensors without reading their data.
func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	var hdrLen uint64
	if err := binary.Read(f, binary.LittleEndian, &hdrLen); err != nil {
		f.Close()
		return nil, fmt.Errorf("read header length: %w", err)
	}
	hdr := make([]byte, hdrLen)
	if _, err := f.ReadAt(hdr, 8); err != nil {
		f.Close()
		return nil, fmt.Errorf("read header: %w", err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(hdr, &raw); err != nil {
		f.Close()
		return nil, fmt.Errorf("parse header: %w", err)
	}
	index := make(map[string]entry, len(raw))
	for name, msg := range raw {
		if name == "__metadata__" {
			continue
		}
		var e entry
		if err := json.Unmarshal(msg, &e); err != nil {
			f.Close()
			return nil, fmt.Errorf("parse entry %q: %w", name, err)
		}
		index[name] = e
	}
	return &File{f: f, dataStart: int64(8 + hdrLen), index: index}, nil
}

// Get returns the dtype, shape, and raw bytes of a named tensor.
func (s *File) Get(name string) (dtype string, shape []int, raw []byte, err error) {
	e, ok := s.index[name]
	if !ok {
		return "", nil, nil, fmt.Errorf("tensor %q not found", name)
	}
	n := e.DataOffsets[1] - e.DataOffsets[0]
	buf := make([]byte, n)
	if _, err := s.f.ReadAt(buf, s.dataStart+e.DataOffsets[0]); err != nil {
		return "", nil, nil, fmt.Errorf("read tensor %q: %w", name, err)
	}
	return e.Dtype, e.Shape, buf, nil
}

// Close releases the underlying file.
func (s *File) Close() error { return s.f.Close() }
