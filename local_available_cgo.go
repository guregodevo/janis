//go:build darwin && arm64 && cgo

package llm

// HasLocalEngine reports whether THIS BUILD can run a model on this machine.
//
// It is a build fact, not a hardware one: the inference engine and its
// tokenizer are cgo libraries, so a static build made for reach (windows, or
// any cgo-free target) carries the marketplace client without them. Callers
// that list or recommend local models must ask this first — otherwise they
// rate models against RAM the build could never use, and tell the user their
// hardware is too small when the truth is the binary cannot run models at all.
func HasLocalEngine() bool { return true }
