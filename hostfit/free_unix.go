//go:build unix

package hostfit

import (
	"path/filepath"
	"syscall"
)

// FreeBytes is the space an unprivileged user can still write on the disk
// holding dir, or its nearest existing parent (dir may not exist yet).
func FreeBytes(dir string) (uint64, error) {
	for {
		var st syscall.Statfs_t
		err := syscall.Statfs(dir, &st)
		if err == nil {
			return st.Bavail * uint64(st.Bsize), nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return 0, err
		}
		dir = parent
	}
}
