//go:build !unix

package hostfit

import "errors"

// FreeBytes is not known here; callers go ahead without the check.
func FreeBytes(dir string) (uint64, error) {
	return 0, errors.New("free disk space is not known on this OS")
}
