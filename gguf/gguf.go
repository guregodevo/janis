// Package gguf is a minimal lazy reader for the .gguf format — the single-file
// container llama.cpp and the GGML ecosystem use: a header (magic, version,
// tensor count, key/value count), a key/value metadata block, a tensor table
// (name, dimensions, type, offset), then the tensor data at an aligned offset.
//
// It is deliberately format-only: it knows the container and the GGML
// quantization types, and nothing about any model architecture. That is what
// makes it language-neutral — the caller decides what the tensor names mean.
//
// Tensors are read on demand by byte range, so a multi-GB file costs one
// open + a header parse, the same bargain the safetensors package makes
// (see ../safetensors). A caller that needs values rather than bytes uses
// F32/F32Tensor, which dequantizes the types GGML writes for weights.
package gguf

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"sort"
)

// Magic is the four bytes every GGUF file starts with, read as a little-endian
// uint32 ("GGUF").
const Magic uint32 = 0x46554747

// Version is the container version this reader understands. GGUF has been
// version 3 since 2024; the field is checked rather than assumed.
const Version = 3

// Range of the GGUFValueType enum, for validating metadata types.
const (
	kvUint8   = 0
	kvInt8    = 1
	kvUint16  = 2
	kvInt16   = 3
	kvUint32  = 4
	kvInt32   = 5
	kvFloat32 = 6
	kvBool    = 7
	kvString  = 8
	kvArray   = 9
	kvUint64  = 10
	kvInt64   = 11
	kvFloat64 = 12
)

// TensorType is the GGML quantization/format of a tensor's stored data.
type TensorType uint32

const (
	TensorF32  TensorType = 0
	TensorF16  TensorType = 1
	TensorQ4_0 TensorType = 2
	TensorQ4_1 TensorType = 3
	TensorQ5_0 TensorType = 6
	TensorQ5_1 TensorType = 7
	TensorQ8_0 TensorType = 8
	TensorQ8_1 TensorType = 9
	TensorBF16 TensorType = 30
)

// block describes how a type packs values: one block holds `values` numbers in
// `bytes` bytes, alongside whatever scales the type carries.
type block struct {
	values int
	bytes  int
}

// typeBlock returns the block layout of a type, or ok=false when the type is
// not one this reader can dequantize. K-quants and IQ-quants are intentionally
// absent: naming them unsupported is honest, whereas mishandling them would
// produce plausible numbers that are wrong.
func typeBlock(t TensorType) (block, bool) {
	switch t {
	case TensorF32:
		return block{1, 4}, true
	case TensorF16:
		return block{1, 2}, true
	case TensorBF16:
		return block{1, 2}, true
	case TensorQ8_0:
		return block{32, 2 + 32}, true // f16 scale + 32 int8
	case TensorQ8_1:
		return block{32, 2 + 2 + 32}, true // scale, sum, 32 int8
	case TensorQ5_0:
		return block{32, 2 + 4 + 16}, true // scale, 32x5bits packed
	case TensorQ5_1:
		return block{32, 2 + 2 + 4 + 16}, true
	case TensorQ4_0:
		return block{32, 2 + 16}, true
	case TensorQ4_1:
		return block{32, 2 + 2 + 16}, true
	}
	return block{}, false
}

// TypeName is the GGML name of a type, for messages and tests.
func TypeName(t TensorType) string {
	switch t {
	case TensorF32:
		return "F32"
	case TensorF16:
		return "F16"
	case TensorBF16:
		return "BF16"
	case TensorQ8_0:
		return "Q8_0"
	case TensorQ8_1:
		return "Q8_1"
	case TensorQ5_0:
		return "Q5_0"
	case TensorQ5_1:
		return "Q5_1"
	case TensorQ4_0:
		return "Q4_0"
	case TensorQ4_1:
		return "Q4_1"
	}
	// Types this reader recognizes but does not dequantize. Naming them keeps an
	// error message useful ("Q4_K is not supported") instead of numeric noise.
	switch t {
	case 10:
		return "Q2_K"
	case 11:
		return "Q3_K"
	case 12:
		return "Q4_K"
	case 13:
		return "Q5_K"
	case 14:
		return "Q6_K"
	case 15:
		return "Q8_K"
	case 16, 17, 18, 19, 20, 21, 22, 23:
		return "IQ" + fmt.Sprint(int(t)-15) + "_unsupported"
	}
	return fmt.Sprintf("GGML_TYPE_%d", uint32(t))
}

type entry struct {
	Type   TensorType
	Shape  []int
	Offset int64 // relative to the data section
}

// File is an open GGUF file: its metadata, its tensor table, and a handle for
// on-demand reads.
type File struct {
	f          *os.File
	index      map[string]entry
	meta       map[string]any
	dataStart  int64
	alignment  int64
	tensorText []string // names in file order, for stable output
}

// Metadata returns the file's key/value metadata. Values are Go types: uint64
// family for the integer kinds, float32/float64, bool, string, and []any for
// arrays.
func (g *File) Metadata() map[string]any { return g.meta }

// Names returns the tensor names sorted; TensorNames returns them in file order.
func (g *File) Names() []string {
	names := make([]string, 0, len(g.index))
	for n := range g.index {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

func (g *File) TensorNames() []string { return append([]string(nil), g.tensorText...) }

// Alignment is the data-section alignment the file declares (general.alignment,
// 32 by default).
func (g *File) Alignment() int64 { return g.alignment }

// Open parses path's header, metadata and tensor table without reading the
// tensor data.
func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	g := &File{f: f, index: map[string]entry{}, meta: map[string]any{}, alignment: 32}
	if err := g.parse(); err != nil {
		f.Close()
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return g, nil
}

// reader walks the file's front matter; it is the only place that reads
// sequentially, so every field decode happens through it.
type reader struct {
	f   *os.File
	pos int64
}

func (r *reader) read(n int) ([]byte, error) {
	buf := make([]byte, n)
	if _, err := r.f.ReadAt(buf, r.pos); err != nil {
		return nil, fmt.Errorf("read %d bytes at %d: %w", n, r.pos, err)
	}
	r.pos += int64(n)
	return buf, nil
}

func (r *reader) u32() (uint32, error) {
	b, err := r.read(4)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(b), nil
}

func (r *reader) u64() (uint64, error) {
	b, err := r.read(8)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(b), nil
}

func (r *reader) str() (string, error) {
	n, err := r.u64()
	if err != nil {
		return "", err
	}
	if n > 1<<24 { // a metadata string is never megabytes; refuse rather than allocate
		return "", fmt.Errorf("string length %d is implausible", n)
	}
	b, err := r.read(int(n))
	if err != nil {
		return "", err
	}
	return string(b), nil
}

func (r *reader) value(t uint32) (any, error) {
	switch t {
	case kvUint8:
		b, err := r.read(1)
		if err != nil {
			return nil, err
		}
		return uint64(b[0]), nil
	case kvInt8:
		b, err := r.read(1)
		if err != nil {
			return nil, err
		}
		return int64(int8(b[0])), nil
	case kvUint16:
		b, err := r.read(2)
		if err != nil {
			return nil, err
		}
		return uint64(binary.LittleEndian.Uint16(b)), nil
	case kvInt16:
		b, err := r.read(2)
		if err != nil {
			return nil, err
		}
		return int64(int16(binary.LittleEndian.Uint16(b))), nil
	case kvUint32:
		v, err := r.u32()
		return uint64(v), err
	case kvInt32:
		v, err := r.u32()
		return int64(int32(v)), err
	case kvFloat32:
		v, err := r.u32()
		return math.Float32frombits(v), err
	case kvBool:
		b, err := r.read(1)
		if err != nil {
			return nil, err
		}
		return b[0] != 0, nil
	case kvString:
		return r.str()
	case kvUint64:
		return r.u64()
	case kvInt64:
		v, err := r.u64()
		return int64(v), err
	case kvFloat64:
		v, err := r.u64()
		return math.Float64frombits(v), err
	case kvArray:
		et, err := r.u32()
		if err != nil {
			return nil, err
		}
		n, err := r.u64()
		if err != nil {
			return nil, err
		}
		if n > 1<<22 {
			return nil, fmt.Errorf("array of %d elements is implausible", n)
		}
		out := make([]any, 0, n)
		for i := uint64(0); i < n; i++ {
			v, err := r.value(et)
			if err != nil {
				return nil, err
			}
			out = append(out, v)
		}
		return out, nil
	}
	return nil, fmt.Errorf("unknown metadata value type %d", t)
}

func (g *File) parse() error {
	r := &reader{f: g.f}
	magic, err := r.u32()
	if err != nil {
		return err
	}
	if magic != Magic {
		return fmt.Errorf("not a GGUF file: magic %08x", magic)
	}
	version, err := r.u32()
	if err != nil {
		return err
	}
	if version != Version {
		return fmt.Errorf("unsupported GGUF version %d (want %d)", version, Version)
	}
	nTensors, err := r.u64()
	if err != nil {
		return err
	}
	nKV, err := r.u64()
	if err != nil {
		return err
	}
	if nTensors > 1<<22 || nKV > 1<<22 {
		return fmt.Errorf("implausible counts: %d tensors, %d metadata entries", nTensors, nKV)
	}
	for i := uint64(0); i < nKV; i++ {
		key, err := r.str()
		if err != nil {
			return err
		}
		t, err := r.u32()
		if err != nil {
			return err
		}
		v, err := r.value(t)
		if err != nil {
			return fmt.Errorf("metadata %q: %w", key, err)
		}
		g.meta[key] = v
	}
	// general.alignment, when present, governs where the data section starts.
	if a, ok := g.meta["general.alignment"].(uint64); ok && a > 0 && a <= 1<<20 {
		g.alignment = int64(a)
	}
	for i := uint64(0); i < nTensors; i++ {
		name, err := r.str()
		if err != nil {
			return err
		}
		nd, err := r.u32()
		if err != nil {
			return err
		}
		if nd > 8 {
			return fmt.Errorf("tensor %q: implausible rank %d", name, nd)
		}
		shape := make([]int, 0, nd)
		for d := uint32(0); d < nd; d++ {
			v, err := r.u64()
			if err != nil {
				return err
			}
			shape = append(shape, int(v))
		}
		tt, err := r.u32()
		if err != nil {
			return err
		}
		off, err := r.u64()
		if err != nil {
			return err
		}
		if _, dup := g.index[name]; dup {
			return fmt.Errorf("tensor %q appears twice", name)
		}
		g.index[name] = entry{Type: TensorType(tt), Shape: shape, Offset: int64(off)}
		g.tensorText = append(g.tensorText, name)
	}
	// Data begins after the (aligned) end of the tensor table.
	g.dataStart = align(r.pos, g.alignment)
	return nil
}

func align(v, a int64) int64 {
	if a <= 0 {
		return v
	}
	return (v + a - 1) / a * a
}

// Has reports whether a tensor is in the file.
func (g *File) Has(name string) bool {
	_, ok := g.index[name]
	return ok
}

// Info returns a tensor's type and shape.
//
// The shape is reported exactly as GGUF stores it, in GGML's dimension order:
// Shape[0] is the FASTEST-VARYING (contiguous) dimension — the reverse of the
// row-major convention safetensors and numpy use. A weight GGML calls
// [in, out] is therefore [out, in] in row-major terms, over the very same
// contiguous bytes, so the VALUES need no transpose: reverse the dims with
// RowMajorShape and the data is already in the right order. Reading it the other
// way is easy and silent — the numbers still look like plausible weights — which
// is why this is spelled out and covered by a test against a second container.
func (g *File) Info(name string) (typeName string, t TensorType, shape []int, err error) {
	e, ok := g.index[name]
	if !ok {
		return "", 0, nil, fmt.Errorf("tensor %q not found", name)
	}
	return TypeName(e.Type), e.Type, append([]int(nil), e.Shape...), nil
}

// Raw returns a tensor's stored bytes exactly as they are in the file, with its
// type and shape. For quantized types these are the packed blocks, not floats.
func (g *File) Raw(name string) (t TensorType, shape []int, raw []byte, err error) {
	e, ok := g.index[name]
	if !ok {
		return 0, nil, nil, fmt.Errorf("tensor %q not found", name)
	}
	blk, ok := typeBlock(e.Type)
	if !ok {
		return 0, nil, nil, fmt.Errorf("tensor %q: %s is not a type this reader dequantizes",
			name, TypeName(e.Type))
	}
	n := 1
	for _, d := range e.Shape {
		n *= d
	}
	if n%blk.values != 0 {
		return 0, nil, nil, fmt.Errorf("tensor %q: %d values is not a multiple of the %s block (%d)",
			name, n, TypeName(e.Type), blk.values)
	}
	size := int64(n/blk.values) * int64(blk.bytes)
	buf := make([]byte, size)
	if _, err := g.f.ReadAt(buf, g.dataStart+e.Offset); err != nil {
		return 0, nil, nil, fmt.Errorf("read tensor %q: %w", name, err)
	}
	return e.Type, append([]int(nil), e.Shape...), buf, nil
}

// F32 dequantizes a tensor to float32 values in the tensor's own logical order
// (row-major), which is what a Go learner/forward pass wants.
func (g *File) F32(name string) ([]float32, []int, error) {
	t, shape, raw, err := g.Raw(name)
	if err != nil {
		return nil, nil, err
	}
	out, err := Dequantize(t, raw)
	if err != nil {
		return nil, nil, fmt.Errorf("tensor %q: %w", name, err)
	}
	return out, shape, nil
}

// Close releases the file handle.
func (g *File) Close() error { return g.f.Close() }

// RowMajorShape reverses a GGML shape into the row-major convention (numpy,
// safetensors): the returned slice describes the same contiguous data with its
// dimensions in the opposite order.
func RowMajorShape(shape []int) []int {
	out := make([]int, len(shape))
	for i, d := range shape {
		out[len(shape)-1-i] = d
	}
	return out
}
