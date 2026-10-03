package qwen

import (
	"encoding/binary"
	"fmt"
	"strconv"

	"github.com/guregodevo/janis/engine"
	"github.com/guregodevo/janis/safetensors"
)

// Snapshot metadata keys. model_id must match exactly on restore — KV is a
// deterministic function of (weights, prompt bytes), so a snapshot from a
// different model is garbage even when every shape lines up.
const (
	snapMetaModel  = "model_id"
	snapMetaLayers = "n_layers"
)

// dtString is the inverse of stDType, for serializing.
func dtString(dt engine.DType) (string, error) {
	switch dt {
	case engine.F32:
		return "F32", nil
	case engine.F16:
		return "F16", nil
	case engine.BF16:
		return "BF16", nil
	case engine.I32:
		return "I32", nil
	case engine.U32:
		return "U32", nil
	default:
		return "", fmt.Errorf("unsupported engine dtype %d", dt)
	}
}

// Save serializes the session's materialized KV prefix (token ids + per-layer
// K/V at native precision) as a single safetensors file. The point: a warmed
// prefix is a deterministic function of (model, prompt bytes), and on a
// streamed-MoE model recomputing it costs 20-60 minutes per gateway restart —
// while loading ~2 GB from disk costs seconds.
func (s *Session) Save(b engine.Backend, path, modelID string) error {
	rr, ok := b.(engine.RawReader)
	if !ok {
		return fmt.Errorf("kv snapshot: backend does not implement RawReader")
	}
	kv := make([]*KVCache, len(s.Caches))
	for i, c := range s.Caches {
		k, isKV := c.(*KVCache)
		if !isKV {
			// A recurrent state is not a token-addressable prefix; snapshotting
			// it would break the LCP contract Restore relies on.
			return fmt.Errorf("kv snapshot: unsupported for hybrid linear-attention sessions")
		}
		kv[i] = k
	}
	if len(kv) == 0 || kv[0] == nil || kv[0].K == nil {
		return fmt.Errorf("kv snapshot: session has no materialized cache")
	}
	// The last sampled token of a turn is recorded in IDs but never forwarded,
	// so the cache is legitimately one shorter. Persist only the materialized
	// prefix — claiming un-forwarded tokens would poison the next LCP.
	t := kv[0].Len()
	if t > len(s.IDs) {
		return fmt.Errorf("kv snapshot: cache holds %d tokens but session ids only %d", t, len(s.IDs))
	}
	ids := s.IDs[:t]

	tensors := make(map[string]safetensors.Entry, 2*len(kv)+1)
	idRaw := make([]byte, len(ids)*4)
	for i, id := range ids {
		binary.LittleEndian.PutUint32(idRaw[i*4:], uint32(id))
	}
	tensors["ids"] = safetensors.Entry{Dtype: "I32", Shape: []int{len(ids)}, Raw: idRaw}

	for i, c := range kv {
		if c == nil || c.K == nil || c.V == nil {
			return fmt.Errorf("kv snapshot: layer %d cache empty", i)
		}
		if c.Len() != t {
			return fmt.Errorf("kv snapshot: layer %d holds %d tokens, layer 0 holds %d", i, c.Len(), t)
		}
		for name, tt := range map[string]engine.Tensor{"k": c.K, "v": c.V} {
			dt, err := dtString(rr.DTypeOf(tt))
			if err != nil {
				return fmt.Errorf("kv snapshot: layer %d %s: %w", i, name, err)
			}
			tensors[fmt.Sprintf("%s.%d", name, i)] = safetensors.Entry{
				Dtype: dt, Shape: tt.Shape(), Raw: rr.Bytes(tt),
			}
		}
	}
	meta := map[string]string{
		snapMetaModel:  modelID,
		snapMetaLayers: strconv.Itoa(len(kv)),
	}
	return safetensors.Write(path, tensors, meta)
}

// Restore loads a Save'd snapshot into this (fresh) session: uploads every
// layer's K/V, pins them against the sweeper, and sets IDs so the next
// Generate's longest-common-prefix logic treats the snapshot exactly like a
// previous turn's prefill. Returns the number of restored prefix tokens.
// A snapshot whose prompt has since changed is NOT an error — LCP simply
// reuses the part that still matches.
func (s *Session) Restore(b engine.Backend, path, modelID string) (int, error) {
	if len(s.Caches) != 0 {
		return 0, fmt.Errorf("kv snapshot: session already has caches — restore only into a fresh session")
	}
	if s.NewCache != nil {
		return 0, fmt.Errorf("kv snapshot: unsupported for hybrid linear-attention sessions")
	}
	f, err := safetensors.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	md := f.Metadata()
	if got := md[snapMetaModel]; got != modelID {
		return 0, fmt.Errorf("kv snapshot: model mismatch (snapshot %q, loaded %q)", got, modelID)
	}
	if got, _ := strconv.Atoi(md[snapMetaLayers]); got != s.NLayers {
		return 0, fmt.Errorf("kv snapshot: layer count mismatch (snapshot %s, model %d)", md[snapMetaLayers], s.NLayers)
	}

	_, idShape, idRaw, err := f.Get("ids")
	if err != nil {
		return 0, err
	}
	ids := make([]int32, idShape[0])
	for i := range ids {
		ids[i] = int32(binary.LittleEndian.Uint32(idRaw[i*4:]))
	}

	caches := make([]LayerCache, s.NLayers)
	for i := 0; i < s.NLayers; i++ {
		c := &KVCache{}
		for name, dst := range map[string]*engine.Tensor{"k": &c.K, "v": &c.V} {
			dt, shape, raw, err := f.Get(fmt.Sprintf("%s.%d", name, i))
			if err != nil {
				return 0, err
			}
			edt, err := stDType(dt)
			if err != nil {
				return 0, err
			}
			*dst = b.FromRaw(edt, raw, shape...)
		}
		caches[i] = c
	}
	// Pin before anything can sweep, or the first Generate frees the restored
	// tensors out from under the forward pass.
	if sw, ok := b.(engine.Sweeper); ok {
		pinCaches(sw, caches)
	}
	s.Caches = caches
	s.IDs = ids
	return len(ids), nil
}
