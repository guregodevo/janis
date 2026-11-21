//go:build darwin && arm64 && cgo

package inkling2

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/guregodevo/janis/engine"
	"github.com/guregodevo/janis/qwen"
)

// counting wraps the draft head to see what it proposed.
type counting struct {
	*Draft
	runs, tokens int
}

func (c *counting) Propose(b engine.Backend, tok int32, caches []qwen.LayerCache) []int32 {
	out := c.Draft.Propose(b, tok, caches)
	c.runs++
	c.tokens += len(out)
	return out
}

// THE DRAFT HEAD CHANGES NOTHING OF THE REPLY.
//
// tiny-draft is a draft head of random weights, in the names and shapes of
// the published one (2 layers, one sliding over 4 tokens, blocks of 4). Its
// proposals mean nothing. With it the session generates the reference's 16
// tokens, the head proposes 3 tokens each time it is asked, its cache
// follows the session's, and a second turn reads on from both.
func TestTheDraftHeadChangesNothingOfTheReply(t *testing.T) {
	b, m, want := tiny(t, "f32")
	raw, err := os.ReadFile(filepath.Join("testdata", "tiny-draft", "mtp_config.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cfg DraftConfig
	if err := json.Unmarshal(raw, &cfg); err != nil {
		t.Fatal(err)
	}
	if err := m.LoadDraft(filepath.Join("testdata", "tiny-draft"), cfg); err != nil {
		t.Fatal(err)
	}
	b.PinAll()

	ids := wantTokens(t, want)[:8]
	_, _, graw, err := want.Get("greedy")
	if err != nil {
		t.Fatal(err)
	}
	wantGen := make([]int32, len(graw)/4)
	for i := range wantGen {
		wantGen[i] = int32(binary.LittleEndian.Uint32(graw[i*4:]))
	}
	s := m.NewSession()
	if s.NLayers != m.Cfg.Visits()+1 {
		t.Fatalf("the session has %d caches, want one more than the %d visits", s.NLayers, m.Cfg.Visits())
	}
	seen := &counting{Draft: m.draft}
	s.Speculator = seen
	gen := s.Generate(b, ids, len(wantGen), qwen.SampleParams{})
	if fmt.Sprint(gen) != fmt.Sprint(wantGen) {
		t.Errorf("generated %v, the reference %v", gen, wantGen)
	}
	if seen.runs == 0 || seen.tokens != seen.runs*(cfg.Block-1) {
		t.Errorf("the head proposed %d tokens in %d runs, want %d a run", seen.tokens, seen.runs, cfg.Block-1)
	}
	read := len(ids) + len(wantGen) - 1
	for i, c := range s.Caches {
		if c.Len() != read {
			t.Errorf("cache %d holds %d tokens, %d were read", i, c.Len(), read)
		}
	}
	dc := s.Caches[len(s.Caches)-1].(*DraftCache)
	if k := dc.K[0]; k == nil || k.Shape()[2] > cfg.Window {
		t.Errorf("the sliding draft layer keeps %v keys, want at most %d", k, cfg.Window)
	}

	more := append(append(append([]int32{}, ids...), gen...), 7, 9)
	got := s.Prefill(b, more, nil)
	fresh := m.NewSession().Prefill(b, more, nil)
	_, scale := maxDiff(fresh, fresh)
	if d, _ := maxDiff(got, fresh); d > 2e-4*scale {
		t.Errorf("the next turn's logits are off by %.3g (largest value %.3g)", d, scale)
	}
}
