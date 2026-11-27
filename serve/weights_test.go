package serve

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"testing"

	llm "github.com/guregodevo/janis"
)

// fakeEditor is a model with one 2x2 matrix, edited in memory.
type fakeEditor struct {
	fakeEngine
	w      []float32
	edited map[string]bool
	saved  string
}

const fakeWeight = "layers.0.mlp.down_proj.weight"

func newFakeEditor() *fakeEditor {
	return &fakeEditor{w: []float32{1, 2, 3, 4}, edited: map[string]bool{}}
}

func (f *fakeEditor) Weights() ([]llm.Weight, error) {
	return []llm.Weight{{Name: fakeWeight, Shape: []int{2, 2}, Dtype: "BF16", GroupSize: 64, Bits: 4, Edited: f.edited[fakeWeight]}}, nil
}
func (f *fakeEditor) Edits() []string {
	var out []string
	for n := range f.edited {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}
func (f *fakeEditor) check(name string) error {
	if name != fakeWeight {
		return fmt.Errorf("weight edits: %q is not a weight of this model", name)
	}
	return nil
}
func (f *fakeEditor) ReadWeight(name string) ([]float32, []int, error) {
	if err := f.check(name); err != nil {
		return nil, nil, err
	}
	return f.w, []int{2, 2}, nil
}
func (f *fakeEditor) EditWeight(name string, v []float32, bits int) error {
	if err := f.check(name); err != nil {
		return err
	}
	if len(v) != 4 {
		return fmt.Errorf("weight edits: %d values for %s [2 2] (4)", len(v), name)
	}
	f.w, f.edited[name] = v, true
	return nil
}
func (f *fakeEditor) ProjectWeight(name string, u []float32, alpha float64) error {
	if err := f.check(name); err != nil {
		return err
	}
	if len(u) != 2 {
		return fmt.Errorf("u has %d values, the matrix writes 2", len(u))
	}
	f.w = []float32{f.w[0] * float32(1-alpha), f.w[1] * float32(1-alpha), f.w[2], f.w[3]}
	f.edited[name] = true
	return nil
}
func (f *fakeEditor) RevertWeights() {
	f.w, f.edited = []float32{1, 2, 3, 4}, map[string]bool{}
}
func (f *fakeEditor) SaveEdits(path string) error {
	if len(f.edited) == 0 {
		return fmt.Errorf("weight edits: none to save")
	}
	f.saved = path
	return nil
}
func (f *fakeEditor) LoadEdits(path string) error {
	if path != f.saved {
		return fmt.Errorf("open %s: no such file", path)
	}
	f.edited[fakeWeight] = true
	return nil
}

func newEditServer(t *testing.T, g Generator, edit bool) string {
	t.Helper()
	installFake(t, "Qwen3.5-0.8B-MLX-4bit", 262144)
	s := NewWithLoader(func(string) (Generator, error) { return g, nil })
	s.logf, s.Edit = t.Logf, edit
	srv := httptest.NewServer(s.Handler())
	t.Cleanup(srv.Close)
	return srv.URL
}

func get(t *testing.T, url string) (*http.Response, map[string]any) {
	t.Helper()
	resp, err := http.Get(url)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var v map[string]any
	_ = jsonDecode(resp, &v)
	return resp, v
}

func TestWeightRoutesAreOffWithoutEdit(t *testing.T) {
	url := newEditServer(t, newFakeEditor(), false)
	if resp, _ := get(t, url+"/v1/weights"); resp.StatusCode != http.StatusNotFound {
		t.Fatalf("GET /v1/weights without --edit: %d", resp.StatusCode)
	}
	if resp, _ := post(t, url+"/v1/weights/revert", `{}`); resp.StatusCode == http.StatusOK {
		t.Fatal("POST /v1/weights/revert answered without --edit")
	}
}

func TestWeightRoutesEditProjectSaveRevertLoad(t *testing.T) {
	ed := newFakeEditor()
	url := newEditServer(t, ed, true)

	resp, v := get(t, url+"/v1/weights")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /v1/weights: %d %v", resp.StatusCode, v)
	}
	data := v["data"].([]any)
	card := data[0].(map[string]any)
	if card["name"] != fakeWeight || card["bits"] != 4.0 || card["edited"] != false {
		t.Fatalf("card %v", card)
	}

	resp, v = get(t, url+"/v1/weights/"+fakeWeight)
	if resp.StatusCode != http.StatusOK || fmt.Sprint(v["values"]) != "[1 2 3 4]" || fmt.Sprint(v["shape"]) != "[2 2]" {
		t.Fatalf("read: %d %v", resp.StatusCode, v)
	}

	resp, v = post(t, url+"/v1/weights/"+fakeWeight, `{"values":[5,6,7,8]}`)
	if resp.StatusCode != http.StatusOK || fmt.Sprint(v["edits"]) != "["+fakeWeight+"]" {
		t.Fatalf("edit: %d %v", resp.StatusCode, v)
	}
	if fmt.Sprint(ed.w) != "[5 6 7 8]" {
		t.Fatalf("weight after edit %v", ed.w)
	}

	// alpha defaults to 1: the direction is removed.
	if resp, v = post(t, url+"/v1/weights/"+fakeWeight+"/project", `{"u":[1,0]}`); resp.StatusCode != http.StatusOK {
		t.Fatalf("project: %d %v", resp.StatusCode, v)
	}
	if fmt.Sprint(ed.w) != "[0 0 7 8]" {
		t.Fatalf("weight after project %v", ed.w)
	}

	if resp, v = post(t, url+"/v1/weights/save", `{"path":"/tmp/x.safetensors"}`); resp.StatusCode != http.StatusOK {
		t.Fatalf("save: %d %v", resp.StatusCode, v)
	}
	resp, v = post(t, url+"/v1/weights/revert", `{}`)
	if resp.StatusCode != http.StatusOK || fmt.Sprint(v["edits"]) != "[]" || fmt.Sprint(ed.w) != "[1 2 3 4]" {
		t.Fatalf("revert: %d %v, weight %v", resp.StatusCode, v, ed.w)
	}
	resp, v = post(t, url+"/v1/weights/load", `{"path":"/tmp/x.safetensors"}`)
	if resp.StatusCode != http.StatusOK || fmt.Sprint(v["edits"]) != "["+fakeWeight+"]" {
		t.Fatalf("load: %d %v", resp.StatusCode, v)
	}
}

func TestWeightRoutesSayWhatIsWrong(t *testing.T) {
	url := newEditServer(t, newFakeEditor(), true)
	for _, c := range []struct {
		method, path, body string
		code               int
		says               string
	}{
		{"GET", "/v1/weights/nope.weight", "", 422, "not a weight"},
		{"POST", "/v1/weights/" + fakeWeight, `{"values":[1]}`, 422, "1 values"},
		{"POST", "/v1/weights/" + fakeWeight, `not json`, 400, "JSON body"},
		{"POST", "/v1/weights/" + fakeWeight + "/project", `{"u":[1,2,3]}`, 422, "3 values"},
		{"POST", "/v1/weights/save", `{"path":"/tmp/y"}`, 422, "none to save"},
		{"POST", "/v1/weights/save", `{}`, 400, "path is empty"},
		{"POST", "/v1/weights/load", `{"path":"/tmp/missing"}`, 422, "no such file"},
	} {
		req, _ := http.NewRequest(c.method, url+c.path, strings.NewReader(c.body))
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		var v map[string]any
		_ = jsonDecode(resp, &v)
		resp.Body.Close()
		msg := fmt.Sprint(v["error"])
		if resp.StatusCode != c.code || !strings.Contains(msg, c.says) {
			t.Errorf("%s %s: %d %s, want %d saying %q", c.method, c.path, resp.StatusCode, msg, c.code, c.says)
		}
	}
}

func TestWeightRoutesNeedAnEditableModel(t *testing.T) {
	url := newEditServer(t, &fakeEngine{}, true)
	resp, v := get(t, url+"/v1/weights")
	if resp.StatusCode != http.StatusNotImplemented || !strings.Contains(fmt.Sprint(v["error"]), "takes no weight edits") {
		t.Fatalf("GET /v1/weights on a model with no edits: %d %v", resp.StatusCode, v)
	}
}

func jsonDecode(resp *http.Response, v any) error { return json.NewDecoder(resp.Body).Decode(v) }

func TestAnEditWaitsForTheRepliesDecoding(t *testing.T) {
	installFake(t, "Qwen3.5-0.8B-MLX-4bit", 262144)
	ed := newFakeEditor()
	s := NewWithLoader(func(string) (Generator, error) { return ed, nil })
	s.logf, s.Edit = t.Logf, true
	srv := httptest.NewServer(s.Handler())
	t.Cleanup(srv.Close)
	s.onEngine(func() { _ = s.ensure(""); s.batch.decoding = 2 })
	resp, v := post(t, srv.URL+"/v1/weights/revert", `{}`)
	if resp.StatusCode != http.StatusConflict || !strings.Contains(fmt.Sprint(v["error"]), "2 replies are decoding") {
		t.Fatalf("revert while decoding: %d %v", resp.StatusCode, v)
	}
	s.onEngine(func() { s.batch.decoding = 0 })
	if resp, _ := post(t, srv.URL+"/v1/weights/revert", `{}`); resp.StatusCode != http.StatusOK {
		t.Fatalf("revert after: %d", resp.StatusCode)
	}
}
