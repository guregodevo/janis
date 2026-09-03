package whisper

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"fmt"
	"strconv"
	"strings"

	_ "embed"
)

// The multilingual tiktoken vocabulary whisper decodes with: 50,257 byte-
// level BPE ranks ("<base64 bytes> <rank>" per line) followed by the
// special tokens whisper defines by position. Decoding is a table lookup;
// encoding (only needed for an optional text prompt) is the byte-pair merge
// over the same ranks.

//go:embed assets/multilingual.tiktoken
var multilingualRanks []byte

// Whisper's special-token layout for the v3 vocabulary (51,866 entries).
const (
	TokEOT           = 50257
	TokSOT           = 50258
	tokLangBase      = 50259 // 100 languages follow, in whisperLanguages order
	numLanguages     = 100
	TokTranslate     = 50359
	TokTranscribe    = 50360
	TokStartOfLM     = 50361
	TokStartOfPrev   = 50362
	TokNoSpeech      = 50363
	TokNoTimestamps  = 50364
	TokTimestampBase = 50365 // <|0.00|>; each step is 0.02 s, 1501 of them
	VocabSize        = 51866
)

// whisperLanguages is the v3 order (99 from v2 plus "yue" last).
var whisperLanguages = strings.Fields(`en zh de es ru ko fr ja pt tr pl ca nl ar sv it id hi fi vi he uk el ms cs ro da hu ta no th ur hr bg lt la mi ml cy sk te fa lv bn sr az sl kn et mk br eu is hy ne mn bs kk sq sw gl mr pa si km sn yo so af oc ka be tg sd gu am yi lo uz fo ht ps tk nn mt sa lb my bo tl mg as tt haw ln ha ba jw su yue`)

// LangToken returns the language token for an ISO code, or 0 if unknown.
func LangToken(code string) int {
	for i, l := range whisperLanguages {
		if l == code {
			return tokLangBase + i
		}
	}
	return 0
}

// LangCode is the inverse of LangToken.
func LangCode(tok int) string {
	if i := tok - tokLangBase; i >= 0 && i < numLanguages {
		return whisperLanguages[i]
	}
	return ""
}

// IsTimestamp reports whether tok is a <|t|> token; Timestamp gives its
// seconds.
func IsTimestamp(tok int) bool  { return tok >= TokTimestampBase && tok < VocabSize }
func Timestamp(tok int) float64 { return float64(tok-TokTimestampBase) * 0.02 }

// Tokenizer decodes (and, for prompts, encodes) whisper's BPE vocabulary.
type Tokenizer struct {
	tokens [][]byte       // rank → bytes
	ranks  map[string]int // bytes → rank
}

// NewTokenizer parses the embedded multilingual ranks.
func NewTokenizer() (*Tokenizer, error) {
	t := &Tokenizer{ranks: make(map[string]int, 51000)}
	sc := bufio.NewScanner(bytes.NewReader(multilingualRanks))
	for sc.Scan() {
		line := sc.Text()
		if line == "" {
			continue
		}
		sp := strings.IndexByte(line, ' ')
		if sp < 0 {
			return nil, fmt.Errorf("tiktoken: bad line %q", line)
		}
		b, err := base64.StdEncoding.DecodeString(line[:sp])
		if err != nil {
			// The last rank (50256) is the empty token, written as a bare "=".
			if b, err = base64.RawStdEncoding.DecodeString(strings.TrimRight(line[:sp], "=")); err != nil {
				return nil, fmt.Errorf("tiktoken: line %q: %w", line, err)
			}
		}
		rank, err := strconv.Atoi(line[sp+1:])
		if err != nil {
			return nil, fmt.Errorf("tiktoken: %w", err)
		}
		for len(t.tokens) <= rank {
			t.tokens = append(t.tokens, nil)
		}
		t.tokens[rank] = b
		t.ranks[string(b)] = rank
	}
	if len(t.tokens) != TokEOT {
		return nil, fmt.Errorf("tiktoken: %d ranks, want %d", len(t.tokens), TokEOT)
	}
	return t, nil
}

// Decode turns text tokens into a string; special tokens are skipped.
func (t *Tokenizer) Decode(ids []int) string {
	var buf bytes.Buffer
	for _, id := range ids {
		if id >= 0 && id < len(t.tokens) {
			buf.Write(t.tokens[id])
		}
	}
	return buf.String()
}

// DecodeWithSpecial is Decode with special tokens rendered as <|name|>.
func (t *Tokenizer) DecodeWithSpecial(ids []int) string {
	var buf bytes.Buffer
	for _, id := range ids {
		switch {
		case id < len(t.tokens):
			buf.Write(t.tokens[id])
		case id == TokEOT:
			buf.WriteString("<|endoftext|>")
		case id == TokSOT:
			buf.WriteString("<|startoftranscript|>")
		case LangCode(id) != "":
			buf.WriteString("<|" + LangCode(id) + "|>")
		case id == TokTranscribe:
			buf.WriteString("<|transcribe|>")
		case id == TokTranslate:
			buf.WriteString("<|translate|>")
		case id == TokNoTimestamps:
			buf.WriteString("<|notimestamps|>")
		case id == TokNoSpeech:
			buf.WriteString("<|nospeech|>")
		case IsTimestamp(id):
			buf.WriteString(fmt.Sprintf("<|%.2f|>", Timestamp(id)))
		}
	}
	return buf.String()
}

// Encode is the byte-pair merge for plain text (prompts): the text is split
// with GPT-2's pre-tokenizer pattern, each piece merged greedily by rank.
func (t *Tokenizer) Encode(text string) []int {
	var out []int
	for _, piece := range gpt2Pretokenize(text) {
		out = append(out, t.bpe([]byte(piece))...)
	}
	return out
}

func (t *Tokenizer) bpe(piece []byte) []int {
	if r, ok := t.ranks[string(piece)]; ok {
		return []int{r}
	}
	parts := make([][]byte, len(piece))
	for i := range piece {
		parts[i] = piece[i : i+1]
	}
	for len(parts) > 1 {
		best, bestRank := -1, int(^uint(0)>>1)
		for i := 0; i+1 < len(parts); i++ {
			if r, ok := t.ranks[string(parts[i])+string(parts[i+1])]; ok && r < bestRank {
				best, bestRank = i, r
			}
		}
		if best < 0 {
			break
		}
		merged := append(append([]byte{}, parts[best]...), parts[best+1]...)
		parts = append(parts[:best], append([][]byte{merged}, parts[best+2:]...)...)
	}
	out := make([]int, 0, len(parts))
	for _, p := range parts {
		if r, ok := t.ranks[string(p)]; ok {
			out = append(out, r)
		}
	}
	return out
}

// gpt2Pretokenize approximates GPT-2's regex split: contractions, words
// with a leading space, numbers, punctuation runs, whitespace.
func gpt2Pretokenize(s string) []string {
	var pieces []string
	rs := []rune(s)
	for i := 0; i < len(rs); {
		start := i
		switch {
		case rs[i] == '\'' && i+1 < len(rs):
			// 's 't 're 've 'm 'll 'd
			for _, c := range []string{"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"} {
				if strings.HasPrefix(string(rs[i:]), c) {
					i += len([]rune(c))
					break
				}
			}
			if i == start {
				i++
			}
		case isLetter(rs[i]) || (rs[i] == ' ' && i+1 < len(rs) && isLetter(rs[i+1])):
			if rs[i] == ' ' {
				i++
			}
			for i < len(rs) && isLetter(rs[i]) {
				i++
			}
		case isDigit(rs[i]) || (rs[i] == ' ' && i+1 < len(rs) && isDigit(rs[i+1])):
			if rs[i] == ' ' {
				i++
			}
			for i < len(rs) && isDigit(rs[i]) {
				i++
			}
		case rs[i] == ' ' || rs[i] == '\n' || rs[i] == '\t':
			for i < len(rs) && (rs[i] == ' ' || rs[i] == '\n' || rs[i] == '\t') {
				i++
			}
		default:
			if rs[i] == ' ' {
				i++
			}
			for i < len(rs) && !isLetter(rs[i]) && !isDigit(rs[i]) && rs[i] != ' ' && rs[i] != '\n' {
				i++
			}
			if i == start {
				i++
			}
		}
		pieces = append(pieces, string(rs[start:i]))
	}
	return pieces
}

func isLetter(r rune) bool {
	return strings.ContainsRune("", r) || (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || r > 127 && !isDigit(r) && r != 0xA0
}
func isDigit(r rune) bool { return r >= '0' && r <= '9' }
