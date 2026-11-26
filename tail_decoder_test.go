package llm

import (
	"strings"
	"testing"
)

// A streamed reply's text, built from its last tokens, is the text of
// decoding all of it: with characters whose bytes lie over several tokens,
// and with a tokenizer that drops the space before the first token. Each
// decode reads a few tokens, however long the reply.
func TestAStreamedReplyIsTheWholeReplysText(t *testing.T) {
	pieces := []string{" the", " light", "house", " keeper", "\n", " é", "\xf0\x9f", "\x8c", "\x8a", " sea", "\xe6", "\xb5\xb7", ".", " ", " end"}
	bytesOf := func(ids []int32) string {
		var b strings.Builder
		for _, id := range ids {
			b.WriteString(pieces[id])
		}
		return strings.ToValidUTF8(b.String(), "\uFFFD")
	}
	for name, decode := range map[string]func([]int32) string{
		"bytes":         bytesOf,
		"leading space": func(ids []int32) string { return strings.TrimPrefix(bytesOf(ids), " ") },
	} {
		var ids []int32
		for i := 0; i < 400; i++ {
			ids = append(ids, int32((i*7+i/5)%len(pieces)))
		}
		for len(ids) > 0 && strings.HasSuffix(decode(ids), "\uFFFD") {
			ids = ids[:len(ids)-1] // a reply that ends inside a character has no last text to compare
		}
		longest := 0
		d := tailDecoder{decode: func(part []int32) string {
			longest = max(longest, len(part))
			return decode(part)
		}}
		want := decode(ids)
		var got string
		for i, id := range ids {
			got = d.add(id)
			if !strings.HasPrefix(want, got) {
				t.Fatalf("%s: after %d tokens the text %q is not the start of the reply", name, i+1, got[max(0, len(got)-30):])
			}
			if strings.HasSuffix(got, "\uFFFD") {
				t.Fatalf("%s: after %d tokens the text ends inside a character", name, i+1)
			}
		}
		if got != want {
			t.Errorf("%s: the stream's text is not the reply's:\n got %q\nwant %q", name, got[max(0, len(got)-60):], want[max(0, len(want)-60):])
		}
		if longest > 12 {
			t.Errorf("%s: one decode read %d tokens of a %d-token reply", name, longest, len(ids))
		}
	}
}
