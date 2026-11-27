//go:build darwin && arm64 && cgo

package inkling2

import (
	"encoding/binary"
	"fmt"
	"math"
	"path/filepath"
	"testing"

	"github.com/guregodevo/janis/engine"
	"github.com/guregodevo/janis/mlxc"
	"github.com/guregodevo/janis/qwen"
	"github.com/guregodevo/janis/safetensors"
)

// tiny loads one of the models testdata/gen_fixture.py made with the
// reference implementation, and what the reference computed on it.
func tiny(t *testing.T, name string) (*mlxc.Backend, *Model, *safetensors.File) {
	return tinyWith(t, name, false)
}

func tinyWith(t *testing.T, name string, stream bool) (*mlxc.Backend, *Model, *safetensors.File) {
	t.Helper()
	b := mlxc.New()
	t.Cleanup(b.Close)
	m, want := loadTiny(t, b, name, stream)
	b.PinAll()
	return b, m, want
}

// EVERY LAYER VISIT GIVES WHAT THE REFERENCE GIVES.
//
// The tiny models have every part of the published one: sliding and full
// layers, a loop of three passes over a core that holds a full layer, nvfp4
// experts with a global scale and mxfp8 ones in the last layer.
//
// The f32 model checks the arithmetic, visit by visit and on the logits.
// The bf16 one checks the dtype each step runs in, on the first six visits
// only: with random weights its rounding flips an expert choice in the
// second loop pass, and from there the reference's two own code paths
// (Metal kernels, plain operations) differ by as much as the values
// (3.0 on visits of at most 9.8, 1.5 on the logits, measured).
func TestEveryVisitMatchesTheReference(t *testing.T) {
	for _, tc := range []struct {
		name   string
		tol    float64 // of the largest expected value
		visits int     // how many are compared, 0 for all and the logits
	}{{"f32", 2e-4, 0}, {"bf16", 5e-2, 6}} {
		t.Run(tc.name, func(t *testing.T) {
			b, m, want := tiny(t, tc.name)
			ids := wantTokens(t, want)
			check := func(name string, got engine.Tensor) {
				t.Helper()
				g, w := b.Floats(got), wantFloats(t, want, name)
				if len(g) != len(w) {
					t.Fatalf("%s: %d values, the reference has %d", name, len(g), len(w))
				}
				if d, s := maxDiff(g, w); d > tc.tol*s || math.IsNaN(d) {
					t.Errorf("%s: off by %.3g (largest value %.3g)", name, d, s)
				}
			}
			var visits []engine.Tensor
			h := m.hidden(b.FromInt32(ids, len(ids)), len(ids), 0, m.newCaches(), func(i int, h engine.Tensor) {
				if got := b.DTypeOf(h); got != m.c.dtype {
					t.Errorf("after visit %d the stream is dtype %v, the model's is %v", i, got, m.c.dtype)
				}
				if i < 0 {
					check("embed", h)
					return
				}
				visits = append(visits, h)
			})
			if len(visits) != m.Cfg.Visits() {
				t.Fatalf("%d visits, want %d", len(visits), m.Cfg.Visits())
			}
			for i, v := range visits {
				if tc.visits == 0 || i < tc.visits {
					check(fmt.Sprintf("visit.%02d", i), v)
				}
			}
			logits := m.logits(h)
			if tc.visits == 0 {
				check("logits", logits)
			}
			for _, v := range b.Floats(logits) {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Fatalf("a logit is %v", v)
				}
			}
		})
	}
}

// TOKENS READ IN STEPS GIVE THE LOGITS OF ONE PASS.
//
// Two chunks of six, then one token at a time: the caches carry the keys
// and values (the sliding window of 8 is passed at token 9), the tails of
// the four convolutions of each visit, and the position the bias and the
// query scale are computed from.
func TestStepsGiveTheLogitsOfOnePass(t *testing.T) {
	b, m, want := tiny(t, "f32")
	ids := wantTokens(t, want)
	full := wantFloats(t, want, "logits")
	vocab := m.Cfg.UnpaddedVocab
	_, scale := maxDiff(full, full)

	caches := m.newCaches()
	for pos := 0; pos < len(ids); {
		n := 1
		if pos < 12 {
			n = 6
		}
		chunk := ids[pos : pos+n]
		got := b.Floats(m.logits(m.hidden(b.FromInt32(chunk, n), n, pos, caches, nil)))
		if d, _ := maxDiff(got, full[pos*vocab:(pos+n)*vocab]); d > 2e-4*scale {
			t.Errorf("tokens %d..%d: off by %.3g from one pass (largest value %.3g)", pos, pos+n-1, d, scale)
		}
		pos += n
		if got := caches[0].Len(); got != pos {
			t.Fatalf("the cache holds %d tokens after %d", got, pos)
		}
	}
	// A sliding layer keeps the keys the next token may see and no more.
	local := caches[0].(*VisitCache)
	if got := local.K.Shape()[2]; got != m.Cfg.Window-1 {
		t.Errorf("a sliding layer keeps %d keys, want window-1 = %d", got, m.Cfg.Window-1)
	}
}

// THE SESSION GENERATES WHAT THE REFERENCE GENERATES.
//
// Greedy, from the first 8 tokens, through janis's own generate loop: the
// prefill, the sweep of each step's intermediates and the caches it pins.
func TestTheSessionGeneratesTheReferenceTokens(t *testing.T) {
	b, m, want := tiny(t, "f32")
	ids := wantTokens(t, want)[:8]
	_, _, raw, err := want.Get("greedy")
	if err != nil {
		t.Fatal(err)
	}
	wantGen := make([]int32, len(raw)/4)
	for i := range wantGen {
		wantGen[i] = int32(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	got := m.NewSession().Generate(b, ids, len(wantGen), qwen.SampleParams{})
	if fmt.Sprint(got) != fmt.Sprint(wantGen) {
		t.Errorf("generated %v, the reference %v", got, wantGen)
	}
}

// A PROMPT THAT PARTS FROM THE LAST ONE RETURNS TO THE MARK.
//
// The caches hold no history to cut back to (convolution tails, a window
// of keys). The session marks where two prompts parted; the next prompt
// that shares that prefix starts from the mark, and its logits are those
// of a session that read it from the first token.
func TestTheSessionReturnsToAMark(t *testing.T) {
	b, m, want := tiny(t, "f32")
	ids := wantTokens(t, want)
	shared := ids[:11]
	first := append(append([]int32{}, shared...), 4, 8, 15)
	second := append(append([]int32{}, shared...), 16, 23, 42, 9)
	third := append(append([]int32{}, shared...), 77, 78)

	s := m.NewSession()
	// Read four tokens a pass: a cache can be cut back inside its last pass,
	// and the prompts part before it.
	s.PrefillChunk = 4
	s.Prefill(b, first, nil)
	s.Prefill(b, second, nil) // parts at 11: read again from 0, marked at 11
	if got := s.Marks(); len(got) != 1 || got[0] != len(shared) {
		t.Fatalf("marks %v, want one at %d", got, len(shared))
	}
	got := s.Prefill(b, third, nil)
	if n := s.Caches[0].Len(); n != len(third) {
		t.Fatalf("the cache holds %d tokens, want %d", n, len(third))
	}
	fresh := m.NewSession().Prefill(b, third, nil)
	_, scale := maxDiff(fresh, fresh)
	if d, _ := maxDiff(got, fresh); d > 2e-4*scale {
		t.Errorf("from the mark the logits are off by %.3g from a fresh read (largest value %.3g)", d, scale)
	}
}

// EXPERTS READ FROM DISK GIVE WHAT EXPERTS ON THE DEVICE GIVE.
//
// With the experts left in the file, each step reads the rows of the
// experts its tokens were routed to. One pass over the 20 tokens gives the
// reference's logits, and the session, which then reads the prompt a token
// at a time, generates the reference's tokens. No expert stack is loaded.
func TestExpertsReadFromDiskGiveTheSameLogits(t *testing.T) {
	b, m, want := tinyWith(t, "f32", true)
	for i, l := range m.Layers {
		if l.MLP.GateUp.W != nil || l.MLP.Down.W != nil {
			t.Fatalf("layer %d holds its experts on the device", i)
		}
	}
	ids := wantTokens(t, want)
	full := wantFloats(t, want, "logits")
	got := b.Floats(m.logits(m.hidden(b.FromInt32(ids, len(ids)), len(ids), 0, m.newCaches(), nil)))
	_, scale := maxDiff(full, full)
	if d, _ := maxDiff(got, full); d > 2e-4*scale {
		t.Errorf("logits off by %.3g (largest value %.3g)", d, scale)
	}
	if l := m.Layers[0].MLP.GateUp; l.reads == 0 || l.reads > int64(m.Cfg.Experts) {
		t.Errorf("layer 0 read %d expert rows for one pass, want 1 to %d", l.reads, m.Cfg.Experts)
	}

	_, _, raw, err := want.Get("greedy")
	if err != nil {
		t.Fatal(err)
	}
	wantGen := make([]int32, len(raw)/4)
	for i := range wantGen {
		wantGen[i] = int32(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	s := m.NewSession()
	if s.PrefillChunk != 1 {
		t.Errorf("the session reads the prompt %d tokens at a time, want 1", s.PrefillChunk)
	}
	before := m.Layers[0].MLP.GateUp.reads
	gen := s.Generate(b, ids[:8], len(wantGen), qwen.SampleParams{})
	if fmt.Sprint(gen) != fmt.Sprint(wantGen) {
		t.Errorf("generated %v, the reference %v", gen, wantGen)
	}
	// 8 prompt tokens and 15 more steps, 2 experts each.
	steps := int64(8 + len(wantGen) - 1)
	if got := m.Layers[0].MLP.GateUp.reads - before; got != steps*int64(m.Cfg.TopK) {
		t.Errorf("layer 0 read %d expert rows for %d steps, want %d", got, steps, steps*int64(m.Cfg.TopK))
	}
}

// A HOT EXPERT IS READ FROM DISK ONCE.
//
// With room in the cache the session generates the same tokens and reads
// each expert it uses once; with a budget of a few experts the cache stays
// within it, less the step in hand.
func TestAHotExpertIsReadOnce(t *testing.T) {
	dir := filepath.Join("testdata", "tiny-f32")
	want, err := safetensors.Open(filepath.Join(dir, "expected.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	ids := wantTokens(t, want)[:8]
	_, _, raw, err := want.Get("greedy")
	if err != nil {
		t.Fatal(err)
	}
	wantGen := make([]int32, len(raw)/4)
	for i := range wantGen {
		wantGen[i] = int32(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	for _, budget := range []int64{1 << 30, 40 << 10} {
		cfg, err := LoadConfig(dir)
		if err != nil {
			t.Fatal(err)
		}
		cfg.StreamExperts, cfg.ExpertCacheBytes = true, budget
		st, err := safetensors.OpenModel(dir)
		if err != nil {
			t.Fatal(err)
		}
		b := mlxc.New()
		m, err := LoadModel(b, st, cfg)
		if err != nil {
			t.Fatal(err)
		}
		b.PinAll()
		gen := m.NewSession().Generate(b, ids, len(wantGen), qwen.SampleParams{})
		if fmt.Sprint(gen) != fmt.Sprint(wantGen) {
			t.Errorf("budget %d: generated %v, the reference %v", budget, gen, wantGen)
		}
		hits, misses, held := m.ExpertCache()
		// 8 layers, 2 projections, 4 experts: 64 rows in all.
		if budget == 1<<30 && (misses > 64 || hits == 0) {
			t.Errorf("with room for all, %d reads and %d hits; want at most 64 reads", misses, hits)
		}
		if one := int64(m.Layers[0].MLP.GateUp.wBytes + m.Layers[0].MLP.GateUp.sBytes); held > budget+int64(m.Cfg.TopK)*one {
			t.Errorf("budget %d: the cache holds %d bytes", budget, held)
		}
		b.Close()
	}
}

// THE MLX-VLM LAYOUT GIVES THE SAME MODEL.
//
// tiny-alt is the f32 model as the netprojects conversions store it: the
// tensors renamed, gate and up projections apart, experts affine at 2 and
// 3 bits, sink_proj and weight_proj as floats. The expected logits are the
// reference's with those requantized experts put back as floats. Read with
// the experts on the device, then from disk.
func TestTheOtherLayoutGivesTheReferenceLogits(t *testing.T) {
	for _, stream := range []bool{false, true} {
		b, m, want := tinyWith(t, "alt", stream)
		if !m.Cfg.AltLayout {
			t.Fatal("the layout was not recognized")
		}
		if l := m.Layers[0].MLP; l.GateUp != nil || l.Gate == nil || l.Gate.Mode != "affine" || l.Gate.Bits != 3 || m.Layers[1].MLP.Down.Bits != 2 {
			t.Fatalf("layer 0 experts: fused %v, gate %+v", l.GateUp != nil, l.Gate)
		}
		ids := wantTokens(t, want)
		full := wantFloats(t, want, "logits")
		got := b.Floats(m.logits(m.hidden(b.FromInt32(ids, len(ids)), len(ids), 0, m.newCaches(), nil)))
		_, scale := maxDiff(full, full)
		if d, _ := maxDiff(got, full); d > 5e-4*scale {
			t.Errorf("experts on disk %v: logits off by %.3g (largest value %.3g)", stream, d, scale)
		}
	}
}

// TOKENS READ IN ONE PASS CAN BE TAKEN BACK.
//
// Eight tokens are read at once, as a run of drafted tokens is, and the
// caches cut back to the first three of them. Reading on from there gives
// the logits of one pass over the true tokens: the keys, the sliding
// window and the four convolution tails are those of token 9.
func TestTokensReadInOnePassCanBeTakenBack(t *testing.T) {
	b, m, want := tiny(t, "f32")
	ids := wantTokens(t, want)
	full := wantFloats(t, want, "logits")
	vocab := m.Cfg.UnpaddedVocab
	_, scale := maxDiff(full, full)

	caches := m.newCaches()
	m.hidden(b.FromInt32(ids[:6], 6), 6, 0, caches, nil)
	wrong := append(append([]int32{}, ids[6:9]...), 1, 2, 3, 4, 5) // three true, five not
	m.hidden(b.FromInt32(wrong, 8), 8, 6, caches, nil)
	for _, c := range caches {
		if !c.CanTruncate(9) {
			t.Fatal("a cache cannot be cut back inside the last pass")
		}
		c.Truncate(b, 9)
		if c.Len() != 9 {
			t.Fatalf("a cache holds %d tokens after the cut, want 9", c.Len())
		}
	}
	rest := ids[9:]
	got := b.Floats(m.logits(m.hidden(b.FromInt32(rest, len(rest)), len(rest), 9, caches, nil)))
	if d, _ := maxDiff(got, full[9*vocab:]); d > 2e-4*scale {
		t.Errorf("after the cut the logits are off by %.3g (largest value %.3g)", d, scale)
	}
	if caches[0].CanTruncate(5) {
		t.Error("a cache claims it can be cut back inside an older pass")
	}
}

// oracle drafts from a list of what comes next, right or wrong.
type oracle struct {
	next  []int32 // the reference's tokens, in order
	wrong func(i int) bool
	at    int
	runs  []int
}

func (o *oracle) Propose(_ engine.Backend, tok int32, _ []qwen.LayerCache) []int32 {
	// tok is next[at-1], or the first token when at is 0.
	for o.at < len(o.next) && o.next[o.at] != tok {
		o.at++
	}
	o.at++
	var out []int32
	for i := o.at; i < len(o.next) && len(out) < 5; i++ {
		t := o.next[i]
		if o.wrong != nil && o.wrong(i) {
			t = (t + 1) % 100
		}
		out = append(out, t)
	}
	o.runs = append(o.runs, len(out))
	return out
}

// DRAFTED TOKENS, RIGHT OR WRONG, LEAVE THE REPLY AS IT WAS.
//
// The session reads a run of drafts in one pass and keeps those the model
// would have sampled. With every draft right, some wrong, or all wrong,
// it generates the reference's 16 tokens, and its caches end holding
// exactly the tokens read.
func TestDraftsLeaveTheReplyAsItWas(t *testing.T) {
	for name, wrong := range map[string]func(int) bool{
		"all right":         nil,
		"every third wrong": func(i int) bool { return i%3 == 2 },
		"all wrong":         func(int) bool { return true },
	} {
		b, m, want := tiny(t, "f32")
		ids := wantTokens(t, want)[:8]
		_, _, raw, err := want.Get("greedy")
		if err != nil {
			t.Fatal(err)
		}
		wantGen := make([]int32, len(raw)/4)
		for i := range wantGen {
			wantGen[i] = int32(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		s := m.NewSession()
		o := &oracle{next: wantGen, wrong: wrong}
		s.Speculator, s.ForwardAll = o, m.forwardAllT
		gen := s.Generate(b, ids, len(wantGen), qwen.SampleParams{})
		if fmt.Sprint(gen) != fmt.Sprint(wantGen) {
			t.Errorf("%s: generated %v, the reference %v", name, gen, wantGen)
		}
		// The last token sampled is never read.
		if got, read := s.Caches[0].Len(), len(ids)+len(wantGen)-1; got != read {
			t.Errorf("%s: the caches hold %d tokens, %d were read", name, got, read)
		}
		if name == "all right" && len(o.runs) > 4 {
			t.Errorf("with every draft right the model was asked %d times for 16 tokens", len(o.runs))
		}
		// A second turn on the same session reads on from these caches.
		more := append(append(append([]int32{}, ids...), gen...), 7, 9)
		got := s.Prefill(b, more, nil)
		fresh := m.NewSession().Prefill(b, more, nil)
		_, scale := maxDiff(fresh, fresh)
		if d, _ := maxDiff(got, fresh); d > 2e-4*scale {
			t.Errorf("%s: the next turn's logits are off by %.3g (largest value %.3g)", name, d, scale)
		}
	}
}
