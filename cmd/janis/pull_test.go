package main

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/guregodevo/janis/hostfit"
)

// pull writes the Hub's files into the cache layout hostfit reads, through a
// temporary name, and a second pull finds the model installed.
func TestPullInstallsIntoTheCacheLayout(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("HF_TOKEN", "")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/api/models/mlx-community/Tiny-1B-4bit":
			_, _ = w.Write([]byte(`{"sha":"abc123","siblings":[{"rfilename":"config.json","size":2},{"rfilename":"model.safetensors","size":5},{"rfilename":".gitattributes","size":1}]}`))
		case strings.HasPrefix(r.URL.Path, "/mlx-community/Tiny-1B-4bit/resolve/abc123/config.json"):
			_, _ = w.Write([]byte("{}"))
		case strings.HasPrefix(r.URL.Path, "/mlx-community/Tiny-1B-4bit/resolve/abc123/model.safetensors"):
			_, _ = w.Write([]byte("12345"))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()
	saved := hubBase
	hubBase = srv.URL
	defer func() { hubBase = saved }()

	pull([]string{"mlx-community/Tiny-1B-4bit"})
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, ".cache/huggingface/hub/models--mlx-community--Tiny-1B-4bit/snapshots/abc123")
	if b, err := os.ReadFile(filepath.Join(dir, "model.safetensors")); err != nil || string(b) != "12345" {
		t.Fatalf("weights in the snapshot: %q %v", b, err)
	}
	if _, err := os.Stat(filepath.Join(dir, ".gitattributes")); err == nil {
		t.Fatal("dot files are not fetched")
	}
	if entries, _ := filepath.Glob(filepath.Join(dir, "*.part")); len(entries) != 0 {
		t.Fatalf("no temporary file left: %v", entries)
	}
	list, _ := hostfit.ListInstalled()
	if !hasModel(list, "Tiny-1B-4bit") {
		t.Fatalf("hostfit lists it: %+v", list)
	}
}
