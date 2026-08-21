//go:build !cgo

// Local inference needs a tokenizer, and ours is a Rust library behind cgo
// with linker flags for darwin and linux only. Rather than let that make the
// whole binary unbuildable elsewhere, this fallback keeps everything that does
// NOT tokenize locally — the marketplace client, the TUI, renting a GPU and
// talking to it — working on any platform Go targets.
//
// A rented GPU tokenizes on the machine you rented, so the remote path is
// unaffected. What is unavailable here is running a model on THIS machine.
package tokenizer

import "fmt"

// Tokenizer is the no-cgo stand-in. It exists so callers compile; every method
// reports plainly that local inference is not available in this build rather
// than returning silently wrong token counts.
type Tokenizer struct{}

// ErrUnavailable is what every path returns here. It names the alternative,
// because "unsupported" without a next step is a dead end.
var ErrUnavailable = fmt.Errorf(
	"local inference is not available in this build (no cgo tokenizer) — rent a GPU with `memdoor llm deploy <model>`")

func New(modelDir string) (*Tokenizer, error) { return nil, ErrUnavailable }

// Encode returns nothing rather than guessing. A wrong token count is worse
// than an empty one: it would silently mis-size a context window.
func (t *Tokenizer) Encode(text string) []int32        { return nil }
func (t *Tokenizer) EncodeSpecial(text string) []int32 { return nil }
func (t *Tokenizer) Decode(ids []int32) string         { return "" }
func (t *Tokenizer) Close()                            {}
