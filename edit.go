package llm

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/guregodevo/janis/engine"
	"github.com/guregodevo/janis/qwen"
	"github.com/guregodevo/janis/safetensors"
)

// editsFormat names a file SaveEdits writes, in its metadata.
const editsFormat = "janis-edits"

// paramLister is a model whose weights an edit may replace in place.
type paramLister interface {
	Params() []qwen.Param
}

// Weight is one weight an edit may replace.
type Weight struct {
	Name string
	// Shape is what ReadWeight returns and EditWeight takes: [out, in] for a
	// quantized matrix, the stored shape for a plain tensor.
	Shape []int
	// Dtype is the checkpoint's dtype of the tensor, or of a quantized
	// matrix's scales and biases.
	Dtype string
	// GroupSize and Bits are the matrix's quantization now, 0 for a plain
	// tensor.
	GroupSize, Bits int
	Edited          bool
}

// weightEdit is one replaced weight: what the checkpoint loaded, to revert
// to, and the tensors that replaced it, as written to an edits file.
type weightEdit struct {
	quant   qwen.QuantLinear
	plain   engine.Tensor
	entries map[string]safetensors.Entry
}

type editState struct {
	st     *safetensors.File
	params map[string]qwen.Param
	order  []string
	edits  map[string]*weightEdit
	editID string
}

func (e *Engine) indexParams(ps []qwen.Param) {
	e.params = make(map[string]qwen.Param, len(ps))
	for _, p := range ps {
		e.params[p.Name] = p
		e.order = append(e.order, p.Name)
	}
	e.edits = map[string]*weightEdit{}
}

// editable fails for a model that lists no weights to edit.
func (e *Engine) editable() error {
	if e.params == nil {
		return fmt.Errorf("weight edits: model_type %q lists no weights", e.mtype)
	}
	return nil
}

func (e *Engine) param(name string) (qwen.Param, error) {
	if err := e.editable(); err != nil {
		return qwen.Param{}, err
	}
	p, ok := e.params[name]
	if !ok {
		return qwen.Param{}, fmt.Errorf("weight edits: %q is not a weight of this model (Weights lists them)", name)
	}
	return p, nil
}

func quantPrefix(name string) string { return strings.TrimSuffix(name, ".weight") }

// checkpointDtype is the dtype the checkpoint stores name in.
func (e *Engine) checkpointDtype(name string) (string, error) {
	dt, _, err := e.st.Info(name)
	if err != nil {
		return "", fmt.Errorf("weight edits: %w", err)
	}
	return dt, nil
}

func (e *Engine) describe(name string) (Weight, error) {
	p := e.params[name]
	w := Weight{Name: name}
	_, w.Edited = e.edits[name]
	var err error
	if l := p.Quant; l != nil {
		ws := l.Weight.Shape()
		w.Shape = []int{ws[0], ws[1] * 32 / l.Bits}
		w.GroupSize, w.Bits = l.GroupSize, l.Bits
		w.Dtype, err = e.checkpointDtype(quantPrefix(name) + ".scales")
	} else {
		w.Shape = append([]int(nil), (*p.Plain).Shape()...)
		w.Dtype, err = e.checkpointDtype(name)
	}
	return w, err
}

// Weights lists the weights an edit may replace, in the model's order.
func (e *Engine) Weights() ([]Weight, error) {
	if err := e.editable(); err != nil {
		return nil, err
	}
	mlxComputeMu.Lock()
	defer mlxComputeMu.Unlock()
	out := make([]Weight, 0, len(e.order))
	for _, n := range e.order {
		w, err := e.describe(n)
		if err != nil {
			return nil, err
		}
		out = append(out, w)
	}
	return out, nil
}

// ReadWeight returns the weight's values as the model uses them now, row
// major, with its shape: a quantized matrix dequantized to [out, in].
func (e *Engine) ReadWeight(name string) ([]float32, []int, error) {
	p, err := e.param(name)
	if err != nil {
		return nil, nil, err
	}
	mlxComputeMu.Lock()
	defer mlxComputeMu.Unlock()
	var t engine.Tensor
	if l := p.Quant; l != nil {
		t = e.bk.Dequantize(l.Weight, l.Scales, l.Biases, l.GroupSize, l.Bits)
	} else {
		t = e.bk.Cast(*p.Plain, engine.F32)
	}
	v, shape := e.bk.Floats(t), append([]int(nil), t.Shape()...)
	sweep(e.bk)
	return v, shape, nil
}

// EditWeight replaces the weight with v, row major in the shape ReadWeight
// returns. A quantized matrix is quantized again in groups of its own size,
// at bits (2, 4 or 8; 0 keeps its bits now); a plain tensor takes bits 0 and
// is stored in the checkpoint's dtype. Every session is reset: its caches
// were computed by the weights before.
func (e *Engine) EditWeight(name string, v []float32, bits int) error {
	p, err := e.param(name)
	if err != nil {
		return err
	}
	for i, x := range v {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return fmt.Errorf("weight edits: %s[%d] is %v", name, i, x)
		}
	}
	mlxComputeMu.Lock()
	defer mlxComputeMu.Unlock()
	entries := map[string]safetensors.Entry{}
	if l := p.Quant; l != nil {
		if bits == 0 {
			bits = l.Bits
		}
		pre := quantPrefix(name)
		dt, err := e.checkpointDtype(pre + ".scales")
		if err != nil {
			return err
		}
		rows, cols := l.Weight.Shape()[0], l.Weight.Shape()[1]*32/l.Bits
		packed, scales, biases, err := qwen.QuantizeAffine(v, rows, cols, l.GroupSize, bits, dt)
		if err != nil {
			return fmt.Errorf("weight edits: %s: %w", name, err)
		}
		entries[pre+".weight"] = safetensors.Entry{Dtype: "U32", Shape: []int{rows, cols * bits / 32}, Raw: packed}
		entries[pre+".scales"] = safetensors.Entry{Dtype: dt, Shape: []int{rows, cols / l.GroupSize}, Raw: scales}
		entries[pre+".biases"] = safetensors.Entry{Dtype: dt, Shape: []int{rows, cols / l.GroupSize}, Raw: biases}
		return e.applyAll(map[string]pendingEdit{name: {p, entries, l.GroupSize, bits}})
	}
	if bits != 0 {
		return fmt.Errorf("weight edits: %s is not quantized, bits must be 0", name)
	}
	shape := (*p.Plain).Shape()
	if n := numel(shape); len(v) != n {
		return fmt.Errorf("weight edits: %d values for %s %v (%d)", len(v), name, shape, n)
	}
	dt, err := e.checkpointDtype(name)
	if err != nil {
		return err
	}
	raw, err := safetensors.EncodeFloats(dt, v)
	if err != nil {
		return fmt.Errorf("weight edits: %s: %w", name, err)
	}
	entries[name] = safetensors.Entry{Dtype: dt, Shape: append([]int(nil), shape...), Raw: raw}
	return e.applyAll(map[string]pendingEdit{name: {p, entries, 0, 0}})
}

// pendingEdit is a weight's replacement, checked against the model.
type pendingEdit struct {
	p               qwen.Param
	entries         map[string]safetensors.Entry
	groupSize, bits int
}

// applyAll applies edits, then resets the sessions once.
func (e *Engine) applyAll(edits map[string]pendingEdit) error {
	defer e.weightsChanged()
	for name, pe := range edits {
		if err := e.apply(name, pe.p, pe.entries, pe.groupSize, pe.bits); err != nil {
			return err
		}
	}
	return nil
}

// apply puts entries in the model in place of the weight name. The caller
// holds mlxComputeMu and has checked entries against the weight.
func (e *Engine) apply(name string, p qwen.Param, entries map[string]safetensors.Entry, groupSize, bits int) error {
	ts := map[string]engine.Tensor{}
	for n, en := range entries {
		t, err := qwen.TensorFrom(e.bk, en)
		if err != nil {
			return fmt.Errorf("weight edits: %s: %w", n, err)
		}
		ts[n] = t
	}
	rec, edited := e.edits[name]
	if !edited {
		rec = &weightEdit{}
		if p.Quant != nil {
			rec.quant = *p.Quant
		} else {
			rec.plain = *p.Plain
		}
	}
	replaced := e.current(p)
	if l := p.Quant; l != nil {
		pre := quantPrefix(name)
		l.Weight, l.Scales, l.Biases = ts[pre+".weight"], ts[pre+".scales"], ts[pre+".biases"]
		l.GroupSize, l.Bits = groupSize, bits
	} else {
		*p.Plain = ts[name]
	}
	if sw, ok := e.bk.(engine.Sweeper); ok {
		sw.Pin(e.current(p)...)
		if edited {
			sw.Unpin(replaced...)
		}
	}
	rec.entries = entries
	e.edits[name] = rec
	return nil
}

// current is the tensors the model uses for p now.
func (e *Engine) current(p qwen.Param) []engine.Tensor {
	if l := p.Quant; l != nil {
		return []engine.Tensor{l.Weight, l.Scales, l.Biases}
	}
	return []engine.Tensor{*p.Plain}
}

// RevertWeights puts back every weight the checkpoint loaded.
func (e *Engine) RevertWeights() {
	mlxComputeMu.Lock()
	defer mlxComputeMu.Unlock()
	if len(e.edits) == 0 {
		return
	}
	sw, _ := e.bk.(engine.Sweeper)
	for name, rec := range e.edits {
		p := e.params[name]
		if sw != nil {
			sw.Unpin(e.current(p)...)
		}
		if p.Quant != nil {
			*p.Quant = rec.quant
		} else {
			*p.Plain = rec.plain
		}
	}
	e.edits = map[string]*weightEdit{}
	e.weightsChanged()
}

// Edits lists the weights replaced now, sorted.
func (e *Engine) Edits() []string {
	mlxComputeMu.Lock()
	defer mlxComputeMu.Unlock()
	names := make([]string, 0, len(e.edits))
	for n := range e.edits {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// weightsChanged resets every session and names the weights now in use, so
// a KV snapshot of other weights is never restored.
func (e *Engine) weightsChanged() {
	for _, s := range []*qwen.Session{e.session, e.utilSession, e.decideSession} {
		s.Reset(e.bk)
	}
	sweep(e.bk)
	e.editID = ""
	if len(e.edits) == 0 {
		return
	}
	h := sha256.New()
	for _, name := range e.sortedEntryNames() {
		en := e.entry(name)
		fmt.Fprintf(h, "%s %s %v %d\n", name, en.Dtype, en.Shape, len(en.Raw))
		h.Write(en.Raw)
	}
	e.editID = hex.EncodeToString(h.Sum(nil))[:16]
}

func (e *Engine) sortedEntryNames() []string {
	var names []string
	for _, rec := range e.edits {
		for n := range rec.entries {
			names = append(names, n)
		}
	}
	sort.Strings(names)
	return names
}

func (e *Engine) entry(name string) safetensors.Entry {
	for _, rec := range e.edits {
		if en, ok := rec.entries[name]; ok {
			return en
		}
	}
	return safetensors.Entry{}
}

// modelID is what a KV snapshot is keyed by: the snapshot directory, and
// the edits when there are any.
func (e *Engine) modelID() string {
	if e.editID == "" {
		return e.modelDir
	}
	return e.modelDir + "#edits:" + e.editID
}

type quantMeta struct {
	GroupSize int `json:"group_size"`
	Bits      int `json:"bits"`
}

// SaveEdits writes the replaced weights to path, a safetensors file in the
// checkpoint's names and dtypes that LoadEdits reads back exactly.
func (e *Engine) SaveEdits(path string) error {
	mlxComputeMu.Lock()
	defer mlxComputeMu.Unlock()
	if len(e.edits) == 0 {
		return fmt.Errorf("weight edits: none to save")
	}
	tensors := map[string]safetensors.Entry{}
	meta := map[string]string{"format": editsFormat}
	for name, rec := range e.edits {
		for n, en := range rec.entries {
			tensors[n] = en
		}
		if l := e.params[name].Quant; l != nil {
			b, _ := json.Marshal(quantMeta{GroupSize: l.GroupSize, Bits: l.Bits})
			meta[name] = string(b)
		}
	}
	return safetensors.Write(path, tensors, meta)
}

// LoadEdits replaces the weights an edits file holds, all of them or none:
// every tensor is checked against the model before any is applied.
func (e *Engine) LoadEdits(path string) error {
	if err := e.editable(); err != nil {
		return err
	}
	f, err := safetensors.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	if f.Metadata()["format"] != editsFormat {
		return fmt.Errorf("weight edits: %s is not an edits file (no format %q)", path, editsFormat)
	}
	mlxComputeMu.Lock()
	defer mlxComputeMu.Unlock()

	todo := map[string]pendingEdit{}
	used := map[string]bool{}
	get := func(n string) (safetensors.Entry, error) {
		dt, shape, raw, err := f.Get(n)
		used[n] = true
		return safetensors.Entry{Dtype: dt, Shape: shape, Raw: raw}, err
	}
	for _, n := range f.Names() {
		p, ok := e.params[n]
		if !ok {
			continue
		}
		w, err := e.describe(n)
		if err != nil {
			return err
		}
		pd := pendingEdit{p: p, entries: map[string]safetensors.Entry{}}
		if p.Quant != nil {
			var q quantMeta
			if err := json.Unmarshal([]byte(f.Metadata()[n]), &q); err != nil || q.Bits == 0 {
				return fmt.Errorf("weight edits: %s: no quantization in the metadata", n)
			}
			if q.GroupSize != w.GroupSize {
				return fmt.Errorf("weight edits: %s: groups of %d, the model's are %d", n, q.GroupSize, w.GroupSize)
			}
			rows, cols := w.Shape[0], w.Shape[1]
			pre := quantPrefix(n)
			want := map[string]safetensors.Entry{
				pre + ".weight": {Dtype: "U32", Shape: []int{rows, cols * q.Bits / 32}},
				pre + ".scales": {Dtype: w.Dtype, Shape: []int{rows, cols / q.GroupSize}},
				pre + ".biases": {Dtype: w.Dtype, Shape: []int{rows, cols / q.GroupSize}},
			}
			for tn, wt := range want {
				en, err := get(tn)
				if err != nil {
					return fmt.Errorf("weight edits: %s: %w", n, err)
				}
				if en.Dtype != wt.Dtype || !sameShape(en.Shape, wt.Shape) {
					return fmt.Errorf("weight edits: %s is %s %v, the model takes %s %v", tn, en.Dtype, en.Shape, wt.Dtype, wt.Shape)
				}
				pd.entries[tn] = en
			}
			pd.groupSize, pd.bits = q.GroupSize, q.Bits
		} else {
			en, err := get(n)
			if err != nil {
				return err
			}
			if en.Dtype != w.Dtype || !sameShape(en.Shape, w.Shape) {
				return fmt.Errorf("weight edits: %s is %s %v, the model takes %s %v", n, en.Dtype, en.Shape, w.Dtype, w.Shape)
			}
			pd.entries[n] = en
		}
		todo[n] = pd
	}
	for _, n := range f.Names() {
		if !used[n] {
			return fmt.Errorf("weight edits: %s in %s is not a weight of this model", n, path)
		}
	}
	if len(todo) == 0 {
		return fmt.Errorf("weight edits: %s holds no weight of this model", path)
	}
	return e.applyAll(todo)
}

func sameShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func numel(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}
