package llm

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestEmbedMemoryGrowth probes whether MLX's buffer-reuse cache grows unbounded
// across many ops of varying size (the suspected "leak"). Gated on MLX_LEAK=1.
//
//	MLX_LEAK=1 go test ./llm/ -run TestEmbedMemoryGrowth -v
func TestEmbedMemoryGrowth(t *testing.T) {
	if os.Getenv("MLX_LEAK") != "1" {
		t.Skip("set MLX_LEAK=1 to run")
	}
	dir := bgeDir(t)
	e, err := OpenEmbedder(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()

	clear := os.Getenv("MLX_CLEAR") == "1"
	if os.Getenv("MLX_CACHE_LIMIT") != "" {
		e.bk.SetCacheLimit(0) // disable cache entirely
		t.Log("cache limit set to 0 (caching disabled)")
	}

	base := "data protection law GDPR article 33 breach notification supervisory authority "
	for i := 0; i < 300; i++ {
		// Vary length 1..40× so every iteration asks MLX for differently-sized
		// buffers — the worst case for a size-keyed reuse cache.
		txt := strings.Repeat(base, 1+(i%40))
		_ = e.Embed(txt)
		if clear {
			e.bk.ClearCache()
		}
		if i%30 == 0 {
			t.Logf("iter=%3d  active=%6.0fMB  cache=%6.0fMB  peak=%6.0fMB",
				i, e.bk.ActiveMemoryMB(), e.bk.CacheMemoryMB(), e.bk.PeakMemoryMB())
		}
	}
	t.Logf("FINAL     active=%6.0fMB  cache=%6.0fMB  peak=%6.0fMB",
		e.bk.ActiveMemoryMB(), e.bk.CacheMemoryMB(), e.bk.PeakMemoryMB())
}

func bgeDir(t *testing.T) string {
	t.Helper()
	if d := os.Getenv("MLX_EMBED_DIR"); d != "" {
		return d
	}
	home, _ := os.UserHomeDir()
	m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--mlx-community--*bge*mlx*/snapshots/*/model.safetensors"))
	if len(m) == 0 {
		t.Skip("no bge-m3 MLX model cached")
	}
	return filepath.Dir(m[0])
}
