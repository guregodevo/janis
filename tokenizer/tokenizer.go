// Package tokenizer wraps the HuggingFace tokenizers library (via cgo) so the
// engine can convert between text and token IDs exactly as the reference does.
package tokenizer

/*
#cgo darwin LDFLAGS: -L${SRCDIR}/../lib
*/
import "C"

import (
	"path/filepath"

	"github.com/daulet/tokenizers"
)

// Tokenizer encodes text to token IDs and back.
type Tokenizer struct {
	tk *tokenizers.Tokenizer
}

// New loads the tokenizer.json from a model directory.
func New(modelDir string) (*Tokenizer, error) {
	tk, err := tokenizers.FromFile(filepath.Join(modelDir, "tokenizer.json"))
	if err != nil {
		return nil, err
	}
	return &Tokenizer{tk: tk}, nil
}

// Encode returns the token IDs for text (special tokens in the text, e.g. chat
// markers, are mapped to their IDs; no extra BOS/EOS is added).
func (t *Tokenizer) Encode(text string) []int32 {
	ids, _ := t.tk.Encode(text, false)
	out := make([]int32, len(ids))
	for i, id := range ids {
		out[i] = int32(id)
	}
	return out
}

// EncodeSpecial returns the token IDs for text WITH the model's special tokens
// added (e.g. BOS/EOS) — used by encoder/embedding models like bge-m3 whose
// reference adds <s>…</s>.
func (t *Tokenizer) EncodeSpecial(text string) []int32 {
	ids, _ := t.tk.Encode(text, true)
	out := make([]int32, len(ids))
	for i, id := range ids {
		out[i] = int32(id)
	}
	return out
}

// Decode renders token IDs back to text, skipping special tokens (clean chat
// output).
func (t *Tokenizer) Decode(ids []int32) string {
	u := make([]uint32, len(ids))
	for i, id := range ids {
		u[i] = uint32(id)
	}
	return t.tk.Decode(u, true)
}

func (t *Tokenizer) Close() { t.tk.Close() }
