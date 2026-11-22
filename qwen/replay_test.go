package qwen

import (
	"testing"

	"github.com/guregodevo/janis/engine"
)

// pinTensor and pinBackend are just enough of a backend to watch settle
// pin the caches: a sweep frees every tracked tensor nobody pinned.
type pinTensor struct{ shape []int }

func (t *pinTensor) Shape() []int { return t.shape }

type pinBackend struct{ pinned map[engine.Tensor]bool }

func (b *pinBackend) BeginStep(int, []engine.Tensor) []engine.Tensor { return nil }
func (b *pinBackend) EndStep(_, _ engine.Tensor, _ []engine.Tensor) (engine.Step, []float32) {
	return nil, nil
}
func (b *pinBackend) CacheView(c engine.Tensor, n int) engine.Tensor {
	s := append([]int(nil), c.Shape()...)
	s[2] = n
	return &pinTensor{shape: s}
}
func (b *pinBackend) PinAll()                   {}
func (b *pinBackend) Pin(ts ...engine.Tensor)   { setPins(b.pinned, ts, true) }
func (b *pinBackend) Unpin(ts ...engine.Tensor) { setPins(b.pinned, ts, false) }
func (b *pinBackend) Sweep()                    {}

func setPins(m map[engine.Tensor]bool, ts []engine.Tensor, on bool) {
	for _, t := range ts {
		m[t] = on
	}
}

type noStep struct{}

func (noStep) Fits(int) bool            { return true }
func (noStep) Run(int32, int) []float32 { return nil }
func (noStep) Close()                   {}

// The caches a recorded step's runs leave are new views: settle must pin
// them and let the ones they replace go, or the next request's sweep frees
// what the caches still hold (a segfault in Truncate on an H100, 2026-11-22).
func TestSettlePinsTheCachesItLeaves(t *testing.T) {
	b := &pinBackend{pinned: map[engine.Tensor]bool{}}
	bucket := []int{1, 2, 256, 8}
	kv := &KVCache{K: &pinTensor{shape: bucket}, V: &pinTensor{shape: bucket}}
	b.Pin(kv.K, kv.V)
	oldK, oldV := kv.K, kv.V
	s := &Session{Caches: []LayerCache{kv}}
	r := &replaying{step: noStep{}, rp: b}
	r.settle(s, 40)
	if kv.K == oldK || kv.K.Shape()[2] != 40 {
		t.Fatalf("the keys were not viewed at 40 tokens: %v", kv.K.Shape())
	}
	for _, x := range cacheTensors(s.Caches) {
		if !b.pinned[x] {
			t.Errorf("a cache tensor of %v is not pinned after settle", x.Shape())
		}
	}
	if b.pinned[oldK] || b.pinned[oldV] {
		t.Error("the padded views settle replaced are still pinned")
	}
}
