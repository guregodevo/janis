package serve

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	llm "github.com/guregodevo/janis"
)

// Editor is a Generator whose weights a request may read and replace
// (llm.Engine on the Qwen3.5 family, edit.go).
type Editor interface {
	Weights() ([]llm.Weight, error)
	Edits() []string
	ReadWeight(name string) ([]float32, []int, error)
	EditWeight(name string, v []float32, bits int) error
	ProjectWeight(name string, u []float32, alpha float64) error
	RevertWeights()
	SaveEdits(path string) error
	LoadEdits(path string) error
}

// maxWeightBody bounds a request carrying a matrix's values: a 27B MLP
// matrix is 89M floats, about 1.5 GB of JSON.
const maxWeightBody = 4 << 30

type weightCard struct {
	Name      string `json:"name"`
	Shape     []int  `json:"shape"`
	Dtype     string `json:"dtype"`
	GroupSize int    `json:"group_size,omitempty"`
	Bits      int    `json:"bits,omitempty"`
	Edited    bool   `json:"edited"`
}

// editsAnswer is what a call that changes the weights answers: the weights
// replaced now.
type editsAnswer struct {
	Edits []string `json:"edits"`
}

func edits(ed Editor) editsAnswer {
	return editsAnswer{Edits: append([]string{}, ed.Edits()...)}
}

// weightRoutes adds the weight-edit endpoints (janis serve --edit).
func (s *Server) weightRoutes(mux *http.ServeMux) {
	mux.HandleFunc("GET /v1/weights", s.handleWeights)
	mux.HandleFunc("GET /v1/weights/{name}", s.handleReadWeight)
	mux.HandleFunc("POST /v1/weights/{name}", s.handleEditWeight)
	mux.HandleFunc("POST /v1/weights/{name}/project", s.handleProjectWeight)
	mux.HandleFunc("POST /v1/weights/revert", s.handleRevert)
	mux.HandleFunc("POST /v1/weights/save", s.handleSaveEdits)
	mux.HandleFunc("POST /v1/weights/load", s.handleLoadEdits)
}

// onEditor runs f on the engine thread with the loaded model as an Editor
// and answers what f returns. An edit resets every session, so none is
// made while the batch loop decodes replies: those would continue on caches
// the old weights computed.
func (s *Server) onEditor(w http.ResponseWriter, r *http.Request, f func(Editor) (any, error)) {
	var out any
	var err error
	code := http.StatusUnprocessableEntity
	cerr := s.onEngineCtx(r.Context(), func() {
		if err = s.ensure(""); err != nil {
			code = http.StatusNotFound
			return
		}
		ed, ok := s.current.(Editor)
		if !ok {
			code, err = http.StatusNotImplemented, fmt.Errorf("model %s takes no weight edits", s.name)
			return
		}
		if s.batch.decoding > 0 {
			code, err = http.StatusConflict, fmt.Errorf("%d replies are decoding: send the weight call once they are done", s.batch.decoding)
			return
		}
		out, err = f(ed)
	})
	if cerr != nil {
		writeCancelled(w)
		return
	}
	if err != nil {
		writeError(w, code, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, out)
}

// decodeBody reads a JSON body into v, answering 400 when it cannot.
func decodeBody(w http.ResponseWriter, r *http.Request, limit int64, v any) bool {
	if err := json.NewDecoder(io.LimitReader(r.Body, limit)).Decode(v); err != nil {
		writeError(w, http.StatusBadRequest, "a JSON body: "+err.Error())
		return false
	}
	return true
}

func (s *Server) handleWeights(w http.ResponseWriter, r *http.Request) {
	s.onEditor(w, r, func(ed Editor) (any, error) {
		ws, err := ed.Weights()
		if err != nil {
			return nil, err
		}
		cards := make([]weightCard, len(ws))
		for i, x := range ws {
			cards[i] = weightCard{Name: x.Name, Shape: x.Shape, Dtype: x.Dtype, GroupSize: x.GroupSize, Bits: x.Bits, Edited: x.Edited}
		}
		return map[string]any{"object": "list", "data": cards, "edits": edits(ed).Edits}, nil
	})
}

func (s *Server) handleReadWeight(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	s.onEditor(w, r, func(ed Editor) (any, error) {
		v, shape, err := ed.ReadWeight(name)
		if err != nil {
			return nil, err
		}
		return map[string]any{"name": name, "shape": shape, "values": v}, nil
	})
}

func (s *Server) handleEditWeight(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Values []float32 `json:"values"`
		Bits   int       `json:"bits"`
	}
	if !decodeBody(w, r, maxWeightBody, &req) {
		return
	}
	name := r.PathValue("name")
	s.onEditor(w, r, func(ed Editor) (any, error) {
		if err := ed.EditWeight(name, req.Values, req.Bits); err != nil {
			return nil, err
		}
		return edits(ed), nil
	})
}

func (s *Server) handleProjectWeight(w http.ResponseWriter, r *http.Request) {
	var req struct {
		U     []float32 `json:"u"`
		Alpha *float64  `json:"alpha"`
	}
	if !decodeBody(w, r, maxWeightBody, &req) {
		return
	}
	alpha := 1.0
	if req.Alpha != nil {
		alpha = *req.Alpha
	}
	name := r.PathValue("name")
	s.onEditor(w, r, func(ed Editor) (any, error) {
		if err := ed.ProjectWeight(name, req.U, alpha); err != nil {
			return nil, err
		}
		return edits(ed), nil
	})
}

func (s *Server) handleRevert(w http.ResponseWriter, r *http.Request) {
	s.onEditor(w, r, func(ed Editor) (any, error) {
		ed.RevertWeights()
		return edits(ed), nil
	})
}

type editsFile struct {
	Path string `json:"path"`
}

func (s *Server) editsFileCall(w http.ResponseWriter, r *http.Request, f func(Editor, string) error) {
	var req editsFile
	if !decodeBody(w, r, 1<<20, &req) {
		return
	}
	if req.Path == "" {
		writeError(w, http.StatusBadRequest, "path is empty")
		return
	}
	s.onEditor(w, r, func(ed Editor) (any, error) {
		if err := f(ed, req.Path); err != nil {
			return nil, err
		}
		return edits(ed), nil
	})
}

func (s *Server) handleSaveEdits(w http.ResponseWriter, r *http.Request) {
	s.editsFileCall(w, r, Editor.SaveEdits)
}

func (s *Server) handleLoadEdits(w http.ResponseWriter, r *http.Request) {
	s.editsFileCall(w, r, Editor.LoadEdits)
}
