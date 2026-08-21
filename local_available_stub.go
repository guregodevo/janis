//go:build !(darwin && arm64 && cgo)

package llm

// HasLocalEngine is false in builds without the cgo inference engine — see
// local_available_cgo.go for why this is a build fact rather than a hardware
// one.
func HasLocalEngine() bool { return false }
