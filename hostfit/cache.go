package hostfit

import (
	"os"
	"path/filepath"
)

// CacheDir is the Hugging Face hub cache janis lists and pulls into, resolved
// the way the Hugging Face tools resolve it: HF_HUB_CACHE, else
// $HF_HOME/hub, else ~/.cache/huggingface/hub. A small home disk keeps its
// models elsewhere with HF_HOME.
func CacheDir() string {
	if d := os.Getenv("HF_HUB_CACHE"); d != "" {
		return d
	}
	if d := os.Getenv("HF_HOME"); d != "" {
		return filepath.Join(d, "hub")
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".cache", "huggingface", "hub")
}
