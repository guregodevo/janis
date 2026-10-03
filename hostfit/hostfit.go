// Package hostfit is the domain model for "which local model fits this machine,
// and how well". It answers two questions in one place — what `memdoor llm auto`
// should install, and the traffic-light the /model picker shows — so the RAM
// tiers live once instead of being re-derived per call site.
//
// It deliberately has NO dependency on the cgo inference engine: whether the
// host is accelerated (an MLX/GPU backend) is detected by the caller and passed
// in, keeping this package pure and testable.
package hostfit

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
)

// Host is the machine an LLM runs on — the constraint that decides which models
// fit. RAMGiB is the binding limit on the accelerated (MLX/GPU) path; the CPU
// path is compute-bound, modeled by Accelerated=false.
type Host struct {
	RAMGiB      int
	Accelerated bool
}

// Detect reads the machine's installed RAM (Linux /proc/meminfo MemTotal, macOS
// and other BSD via `sysctl hw.memsize`). `accelerated` reflects whether a
// GPU/MLX backend is in use — the caller knows this (it links the engine).
func Detect(accelerated bool) (Host, error) {
	gib, err := ramGiB()
	if err != nil {
		return Host{}, err
	}
	return Host{RAMGiB: gib, Accelerated: accelerated}, nil
}

// Recommend returns the mlx-community model name best suited to this host — the
// single source of truth for "what should I run here".
//
// On a GPU (MLX) backend the limit is RAM — bigger models run fast, so scale up
// with installed memory. On the CPU backend the limit is COMPUTE, not RAM: an 8B
// fits in 16 GB but decodes painfully slowly, so cap at a small model regardless
// of RAM.
func (h Host) Recommend() string {
	if !h.Accelerated {
		// CPU: compute- AND memory-bound. A 1.7B is the right default — measured
		// ~1.7 tok/s on the pure-Go backend (vs ~1.4 for a 4B), but the bigger win
		// for weak hardware is footprint: ~1 GB download and much less RAM than a
		// 4B (~2.5 GB), so it's likelier to run at all on an old box.
		return "Qwen3-1.7B-4bit"
	}
	switch {
	case h.RAMGiB >= 64:
		return "Qwen3-32B-4bit"
	case h.RAMGiB >= 32:
		return "Qwen3-14B-4bit"
	case h.RAMGiB >= 16:
		// Qwen3.5-9B replaced Qwen3-8B as the 16 GB default (2026-09-01): the
		// hybrid linear-attention stack decodes FASTER in this engine (11.5 vs
		// 8 tok/s measured) and only every 4th layer grows a KV cache, so long
		// contexts cost far less memory.
		return "Qwen3.5-9B-MLX-4bit"
	default:
		return "Qwen3-4B-Instruct-2507-4bit"
	}
}

// Fit rates how well a model runs on a host.
type Fit int

const (
	FitUnknown Fit = iota // size couldn't be determined (e.g. an embedding model)
	FitGood               // runs comfortably — at or below what this host should run
	FitTight              // fits but bandwidth-bound / little headroom
	FitTooBig             // won't fit this RAM well
)

// Flag is the traffic-light glyph (two-cell width preserved for unknown so a
// model list stays aligned).
func (f Fit) Flag() string {
	switch f {
	case FitGood:
		return "🟢"
	case FitTight:
		return "🟡"
	case FitTooBig:
		return "🔴"
	default:
		return "  "
	}
}

// Rate rates a model on this host by comparing its parameter size to the size
// this host is recommended to run (Recommend, which already bakes in headroom
// for the KV cache + OS): at or below → Good, up to ~2× → Tight, beyond →
// TooBig. Returns FitUnknown when the model name carries no parameter count
// (embedding/ASR models), so callers don't flag a non-chat model.
func (h Host) Rate(modelName string) Fit {
	p := ParamsB(modelName)
	if p <= 0 {
		return FitUnknown
	}
	rec := ParamsB(h.Recommend())
	if rec <= 0 {
		return FitUnknown
	}
	switch {
	case p <= rec:
		return FitGood
	case p <= rec*2:
		return FitTight
	default:
		return FitTooBig
	}
}

// paramsRe matches the first "<n>B" / "<n.n>B" parameter token in a model name,
// e.g. Qwen3-8B-4bit → 8, Qwen2.5-0.5B → 0.5, Qwen3-30B-A3B → 30 (total, not
// the active-expert count).
var paramsRe = regexp.MustCompile(`(\d+(?:\.\d+)?)[Bb](?:[^a-zA-Z]|$)`)

// ParamsB extracts a model's parameter count in billions from its name, or 0
// when the name carries none.
func ParamsB(name string) float64 {
	m := paramsRe.FindStringSubmatch(name)
	if m == nil {
		return 0
	}
	v, _ := strconv.ParseFloat(m[1], 64)
	return v
}

// InstalledModel is one locally-cached mlx-community model.
type InstalledModel struct {
	Name      string // bare model name, e.g. "Qwen3-8B-4bit"
	SizeBytes int64  // on-disk size of the snapshot
	Path      string // snapshot directory
}

// ListInstalled returns the mlx-community models in the local HuggingFace cache,
// de-duplicated by name (first snapshot wins). The single inventory source the
// CLI (`llm list`) and the TUI (/model) both read, so neither re-globs by hand.
func ListInstalled() ([]InstalledModel, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}
	matches, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--mlx-community--*/snapshots/*"))
	out := make([]InstalledModel, 0, len(matches))
	seen := map[string]bool{}
	for _, dir := range matches {
		name := nameFromPath(dir)
		if name == "" || seen[name] || IsEmbeddingModel(name) {
			continue
		}
		seen[name] = true
		out = append(out, InstalledModel{Name: name, SizeBytes: dirSize(dir), Path: dir})
	}
	return out, nil
}

// IsEmbeddingModel reports whether a cached model is an embedder rather than a
// chat model. They share the cache directory, so a naive listing offers them as
// brains — and picking one silently breaks every turn, because an embedder
// cannot answer. Matched by name: these families exist to produce vectors.
func IsEmbeddingModel(name string) bool {
	n := strings.ToLower(name)
	for _, marker := range []string{"bge-", "gte-", "e5-", "-embed", "embedding", "reranker", "rerank-"} {
		if strings.Contains(n, marker) {
			return true
		}
	}
	return false
}

// nameFromPath extracts the bare model name from a HuggingFace cache path by
// walking up to the models--mlx-community--<NAME> dir.
func nameFromPath(p string) string {
	for {
		base := filepath.Base(p)
		if strings.HasPrefix(base, "models--mlx-community--") {
			return strings.TrimPrefix(base, "models--mlx-community--")
		}
		next := filepath.Dir(p)
		if next == p {
			return p
		}
		p = next
	}
}

// dirSize sums the sizes of files under dir.
func dirSize(dir string) int64 {
	var total int64
	_ = filepath.Walk(dir, func(_ string, info os.FileInfo, err error) error {
		if err == nil && !info.IsDir() {
			total += info.Size()
		}
		return nil
	})
	return total
}

// HumanSize formats a byte count as B/KB/MB/GB.
func HumanSize(n int64) string {
	const k = 1024.0
	switch {
	case n < k:
		return fmt.Sprintf("%d B", n)
	case n < k*k:
		return fmt.Sprintf("%.1f KB", float64(n)/k)
	case n < k*k*k:
		return fmt.Sprintf("%.1f MB", float64(n)/(k*k))
	default:
		return fmt.Sprintf("%.1f GB", float64(n)/(k*k*k))
	}
}

// ramGiB reports installed RAM in whole GiB.
func ramGiB() (int, error) {
	if runtime.GOOS == "linux" {
		data, err := os.ReadFile("/proc/meminfo")
		if err != nil {
			return 0, err
		}
		for _, line := range strings.Split(string(data), "\n") {
			if !strings.HasPrefix(line, "MemTotal:") {
				continue
			}
			fields := strings.Fields(line) // "MemTotal: 16307812 kB"
			if len(fields) < 2 {
				break
			}
			kb, err := strconv.ParseInt(fields[1], 10, 64)
			if err != nil {
				return 0, err
			}
			return int(kb / (1024 * 1024)), nil
		}
		return 0, fmt.Errorf("MemTotal not found in /proc/meminfo")
	}
	out, err := exec.Command("sysctl", "-n", "hw.memsize").Output()
	if err != nil {
		return 0, err
	}
	bytes, err := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64)
	if err != nil {
		return 0, err
	}
	return int(bytes / (1024 * 1024 * 1024)), nil
}
