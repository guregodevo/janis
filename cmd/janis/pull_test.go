package main

import (
	"fmt"
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

	if err := pull([]string{"mlx-community/Tiny-1B-4bit"}); err != nil {
		t.Fatal(err)
	}
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

// fakeHub answers the model API with one weights file of the given size per
// model, and counts the downloads.
func fakeHub(t *testing.T, sizes map[string]int64) *int {
	t.Helper()
	fetched := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		for name, size := range sizes {
			if r.URL.Path == "/api/models/mlx-community/"+name {
				fmt.Fprintf(w, `{"sha":"abc","siblings":[{"rfilename":"model.safetensors","size":%d}]}`, size)
				return
			}
			if strings.HasPrefix(r.URL.Path, "/mlx-community/"+name+"/resolve/") {
				fetched++
				_, _ = w.Write(make([]byte, size))
				return
			}
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	t.Cleanup(srv.Close)
	saved := hubBase
	hubBase = srv.URL
	t.Cleanup(func() { hubBase = saved })
	return &fetched
}

// smallDisk makes the cache's disk report free bytes.
func smallDisk(t *testing.T, free uint64) {
	saved := freeBytes
	freeBytes = func(string) (uint64, error) { return free, nil }
	t.Cleanup(func() { freeBytes = saved })
}

// A named model the disk has no room for is refused before a byte is fetched.
func TestPullRefusesAModelTheDiskCannotHold(t *testing.T) {
	t.Setenv("HF_HOME", t.TempDir())
	fetched := fakeHub(t, map[string]int64{"Big-9B-4bit": 2 << 30})
	smallDisk(t, 1<<30)
	err := pull([]string{"Big-9B-4bit"})
	if err == nil || !strings.Contains(err.Error(), "HF_HOME") {
		t.Fatalf("pull = %v, want a refusal naming HF_HOME", err)
	}
	if *fetched != 0 {
		t.Fatalf("%d downloads started", *fetched)
	}
}

// With no name, a disk without room for the RAM tier's model gets the
// largest smaller tier that fits.
func TestPickModelStepsDownToWhatTheDiskHolds(t *testing.T) {
	t.Setenv("HF_HOME", t.TempDir())
	fakeHub(t, map[string]int64{
		"Qwen3.8-27B-4bit":            15 << 30,
		"Qwen3.5-9B-MLX-4bit":         5 << 30,
		"Qwen3-4B-Instruct-2507-4bit": 2 << 30,
	})
	smallDisk(t, 3<<30)
	got, err := pickModel(&http.Client{}, hostfit.Host{RAMGiB: 32})
	if err != nil || got != "Qwen3-4B-Instruct-2507-4bit" {
		t.Fatalf("pickModel = %q, %v; want the 4B", got, err)
	}
	smallDisk(t, 1<<30)
	if _, err := pickModel(&http.Client{}, hostfit.Host{RAMGiB: 32}); err == nil {
		t.Fatal("no tier fits 1 GiB, want an error")
	}
}

// A partial file from a stopped pull is started over, so its room counts as
// free: the disk needs only the rest.
func TestMissingBytesCountsAPartialFileAsFree(t *testing.T) {
	dir := t.TempDir()
	meta := repoMeta{SHA: "abc", Files: []struct {
		Name string
		Size int64
	}{{"model.safetensors", 100}, {"config.json", 10}}}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.part"), make([]byte, 30), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), make([]byte, 10), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := missingBytes(meta, dir); got != 70 {
		t.Fatalf("missingBytes = %d, want 70", got)
	}
}

// Each file ends on one line at 100%: a small file done within one redraw
// interval printed an empty line, and the read that returns EOF carries no
// bytes, so the last redraw stopped short (99.9% of 5.0 GB, 2026-11-12).
func TestEachFileEndsOnOneLineAt100Percent(t *testing.T) {
	body := strings.Repeat("x", 3<<20)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "small.json") {
			w.Write([]byte("{}"))
			return
		}
		w.Write([]byte(body))
	}))
	defer srv.Close()
	var out strings.Builder
	saved := progressOut
	progressOut = &out
	t.Cleanup(func() { progressOut = saved })
	dir := t.TempDir()
	for _, f := range []struct {
		name string
		size int64
	}{{"small.json", 2}, {"model.safetensors", int64(len(body))}} {
		out.Reset()
		if err := download(srv.Client(), srv.URL+"/"+f.name, filepath.Join(dir, f.name), f.size); err != nil {
			t.Fatal(err)
		}
		got := out.String()
		if !strings.HasSuffix(got, "\n") || strings.Count(got, "\n") != 1 {
			t.Fatalf("%s: one line: %q", f.name, got)
		}
		draws := strings.Split(strings.TrimSuffix(got, "\n"), "\r")
		last := draws[len(draws)-1]
		if !strings.Contains(last, f.name) || !strings.Contains(last, "100.0%") {
			t.Fatalf("%s: the last draw: %q", f.name, last)
		}
	}
}
