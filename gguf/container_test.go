package gguf

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// synTensor is a tensor for the synthetic container below: its GGML type, its
// GGML dims, and the payload bytes exactly as they will appear in the file.
type synTensor struct {
	name string
	t    TensorType
	dims []int
	data []byte
}

// writeGGUF builds a minimal but real GGUF container by hand, so the reader can
// be tested on malformed and unusual files that no published model would use.
func writeGGUF(t *testing.T, path string, meta [][2]any, tensors []synTensor, alignment int) {
	t.Helper()
	var b []byte
	u32 := func(v uint32) { b = binary.LittleEndian.AppendUint32(b, v) }
	u64 := func(v uint64) { b = binary.LittleEndian.AppendUint64(b, v) }
	str := func(s string) { u64(uint64(len(s))); b = append(b, s...) }

	u32(Magic)
	u32(Version)
	u64(uint64(len(tensors)))
	u64(uint64(len(meta)))
	for _, kv := range meta {
		str(kv[0].(string))
		switch v := kv[1].(type) {
		case string:
			u32(kvString)
			str(v)
		case uint64:
			u32(kvUint64)
			u64(v)
		case uint32:
			u32(kvUint32)
			u32(v)
		case bool:
			u32(kvBool)
			if v {
				b = append(b, 1)
			} else {
				b = append(b, 0)
			}
		case float32:
			u32(kvFloat32)
			u32(math.Float32bits(v))
		case []string:
			u32(kvArray)
			u32(kvString)
			u64(uint64(len(v)))
			for _, s := range v {
				str(s)
			}
		default:
			t.Fatalf("test bug: unsupported metadata type %T", kv[1])
		}
	}
	// Tensor table. Offsets are filled in after the header layout is known.
	dataStart := 0
	offsets := make([]int, len(tensors))
	{
		// simulate the table size to compute where data begins
		probe := len(b)
		_ = probe
	}
	type pending struct {
		off int
	}
	var positions []pending
	for i, tn := range tensors {
		_ = i
		str(tn.name)
		u32(uint32(len(tn.dims)))
		for _, d := range tn.dims {
			u64(uint64(d))
		}
		u32(uint32(tn.t))
		positions = append(positions, pending{off: len(b)})
		u64(0) // placeholder
	}
	// data begins at the aligned end of the table
	dataStart = (len(b) + alignment - 1) / alignment * alignment
	// Tensor offsets in GGUF are relative to the START OF THE DATA SECTION, not
	// the file — the reader adds dataStart. Writing absolute offsets here made
	// every read land past EOF, which is how this was pinned down.
	off := 0
	for i, tn := range tensors {
		offsets[i] = off
		binary.LittleEndian.PutUint64(b[positions[i].off:], uint64(off))
		off += len(tn.data)
	}
	for len(b) < dataStart {
		b = append(b, 0) // alignment padding
	}
	for _, tn := range tensors {
		b = append(b, tn.data...)
	}
	_ = offsets
	if err := os.WriteFile(path, b, 0644); err != nil {
		t.Fatal(err)
	}
}

func f32Bytes(vals ...float32) []byte {
	var out []byte
	for _, v := range vals {
		out = binary.LittleEndian.AppendUint32(out, math.Float32bits(v))
	}
	return out
}

// TestOpenSyntheticContainer builds a container by hand and reads it back: this
// pins the header layout, the key/value decoding, the alignment arithmetic that
// locates the data section, and the tensor-table offsets — the parts a bug in
// would silently misread a real file rather than fail loudly.
func TestOpenSyntheticContainer(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "synthetic.gguf")
	tensors := []synTensor{
		{name: "a.weight", t: TensorF32, dims: []int{3, 2}, data: f32Bytes(1, 2, 3, 4, 5, 6)},
		{name: "b.weight", t: TensorF32, dims: []int{2}, data: f32Bytes(-1, 0.5)},
	}
	// alignment 64, not the default 32: the reader must honour what the file says
	writeGGUF(t, path, [][2]any{
		{"general.architecture", "synthetic"},
		{"general.alignment", uint64(64)},
		{"synth.block_count", uint32(2)},
		{"synth.scale", float32(0.5)},
		{"synth.flag", true},
		{"synth.names", []string{"x", "y", "z"}},
		{"synth.big", uint64(1 << 40)},
	}, tensors, 64)

	g, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer g.Close()
	if got := g.Alignment(); got != 64 {
		t.Errorf("alignment = %d, want 64 from general.alignment", got)
	}
	m := g.Metadata()
	if m["general.architecture"] != "synthetic" {
		t.Errorf("architecture = %v", m["general.architecture"])
	}
	if v, ok := m["synth.block_count"].(uint64); !ok || v != 2 {
		t.Errorf("uint32 metadata = %v (%T), want uint64 2", m["synth.block_count"], m["synth.block_count"])
	}
	if v, ok := m["synth.scale"].(float32); !ok || v != 0.5 {
		t.Errorf("float metadata = %v, want 0.5", m["synth.scale"])
	}
	if v, ok := m["synth.flag"].(bool); !ok || !v {
		t.Errorf("bool metadata = %v, want true", m["synth.flag"])
	}
	if v, ok := m["synth.names"].([]any); !ok || len(v) != 3 || v[2] != "z" {
		t.Errorf("string array metadata = %v, want [x y z]", m["synth.names"])
	}
	if v, ok := m["synth.big"].(uint64); !ok || v != 1<<40 {
		t.Errorf("uint64 metadata = %v, want 2^40", m["synth.big"])
	}

	if names := g.Names(); len(names) != 2 || names[0] != "a.weight" || names[1] != "b.weight" {
		t.Errorf("Names = %v", names)
	}
	if inOrder := g.TensorNames(); inOrder[0] != "a.weight" {
		t.Errorf("TensorNames = %v", inOrder)
	}
	_, tt, shape, err := g.Info("a.weight")
	if err != nil {
		t.Fatal(err)
	}
	if tt != TensorF32 || len(shape) != 2 || shape[0] != 3 || shape[1] != 2 {
		t.Errorf("Info(a.weight) = %s %v", TypeName(tt), shape)
	}
	vals, _, err := g.F32("a.weight")
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{1, 2, 3, 4, 5, 6}
	for i := range want {
		if vals[i] != want[i] {
			t.Errorf("a.weight[%d] = %v, want %v — the data offset or alignment is wrong",
				i, vals[i], want[i])
		}
	}
	// b follows a directly; an off-by-one in the offset walk would show here.
	vals, _, err = g.F32("b.weight")
	if err != nil {
		t.Fatal(err)
	}
	if vals[0] != -1 || vals[1] != 0.5 {
		t.Errorf("b.weight = %v, want [-1 0.5]", vals)
	}
}

// TestOpenRejectsMalformed checks the reader fails cleanly (no panic, no silent
// acceptance) on the ways a file can be wrong.
func TestOpenRejectsMalformed(t *testing.T) {
	dir := t.TempDir()
	good := []synTensor{{name: "t", t: TensorF32, dims: []int{2}, data: f32Bytes(1, 2)}}
	valid := filepath.Join(dir, "valid.gguf")
	writeGGUF(t, valid, nil, good, 32)
	raw, err := os.ReadFile(valid)
	if err != nil {
		t.Fatal(err)
	}

	cases := []struct {
		name    string
		bytes   []byte
		wantSub string
	}{
		{"bad magic", append([]byte("XXXX"), raw[4:]...), "not a GGUF file"},
		{"bad version", func() []byte {
			b := append([]byte(nil), raw...)
			binary.LittleEndian.PutUint32(b[4:], 1)
			return b
		}(), "unsupported GGUF version"},
		{"truncated header", raw[:12], "read"},
		{"empty file", nil, "read"},
		{"implausible tensor count", func() []byte {
			b := append([]byte(nil), raw...)
			binary.LittleEndian.PutUint64(b[8:], 1<<40)
			return b
		}(), "implausible"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := filepath.Join(dir, strings.ReplaceAll(tc.name, " ", "_")+".gguf")
			if err := os.WriteFile(p, tc.bytes, 0644); err != nil {
				t.Fatal(err)
			}
			g, err := Open(p)
			if err == nil {
				g.Close()
				t.Fatalf("Open accepted %s", tc.name)
			}
			if !strings.Contains(err.Error(), tc.wantSub) {
				t.Errorf("error %q does not mention %q", err, tc.wantSub)
			}
		})
	}
}

// TestTensorLookupErrors covers the lookup and capability failures a caller
// meets: a missing name, and a type this reader refuses to dequantize.
func TestTensorLookupErrors(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "types.gguf")
	// Q4_K (type 12) is a real GGML type this reader deliberately does not
	// support: claiming bytes are values would be worse than refusing.
	writeGGUF(t, path, nil, []synTensor{
		{name: "f32", t: TensorF32, dims: []int{1}, data: f32Bytes(1)},
		{name: "q4k", t: TensorType(12), dims: []int{256}, data: make([]byte, 144)},
		{name: "q4_0", t: TensorQ4_0, dims: []int{31}, data: make([]byte, 18)},
	}, 32)
	g, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()

	if g.Has("nope") {
		t.Error("Has(nope) = true")
	}
	if _, _, _, err := g.Info("nope"); err == nil {
		t.Error("Info(nope) succeeded")
	}
	if _, _, _, err := g.Raw("nope"); err == nil {
		t.Error("Raw(nope) succeeded")
	}
	if _, _, err := g.F32("nope"); err == nil {
		t.Error("F32(nope) succeeded")
	}
	// Q4_K must be refused by name, so a caller knows why.
	if _, _, _, err := g.Raw("q4k"); err == nil || !strings.Contains(err.Error(), "Q4_K") {
		t.Errorf("Raw(q4k) error = %v, want it to name the unsupported type", err)
	}
	// 31 values is not a whole Q4_0 block of 32: refuse rather than truncate.
	if _, _, _, err := g.Raw("q4_0"); err == nil {
		t.Error("Raw accepted a tensor that is not a whole block")
	}
	if got := TypeName(TensorType(12)); got != "GGML_TYPE_12" {
		// Q4_K is not in this reader's name table on purpose; the numeric form is
		// what a reader that does not support it should print.
		t.Logf("TypeName(12) = %q (unnamed type prints numerically)", got)
	}
}
