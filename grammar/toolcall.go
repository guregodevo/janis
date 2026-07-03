// Package grammar implements grammar-constrained decoding: a Grammar vets each
// candidate token during sampling and advances its state as tokens commit, so the
// model can only produce output matching a formal structure.
//
// ToolCall enforces the Hermes/Qwen tool-call envelope — <tool_call>{json}
// </tool_call> — which is the single canonical format the provider's ToolProtocol
// renders and parses. It makes every tool call the model DECIDES to emit
// well-formed (valid JSON, properly escaped, correctly closed) instead of relying
// on the parser to repair malformed output. Plain-text answers stay unconstrained:
// the grammar only engages AFTER the model itself opens a <tool_call>.
package grammar

// Decoder renders a single token id to its text — the tokenizer's Decode of [id].
type Decoder func(id int32) string

const (
	openLit  = "<tool_call>"
	closeLit = "</tool_call>"
)

type phase uint8

const (
	phText  phase = iota // free assistant text; the open literal is being watched
	phJSON               // inside the tool_call, consuming the JSON object
	phClose              // JSON done; the closing </tool_call> tag is required
)

// state is a value type (copyable) so Allows can simulate a candidate token's
// bytes on a COPY without disturbing the committed state.
type state struct {
	phase   phase
	open    int  // matched prefix length of openLit while in phText
	closed  int  // matched prefix length of closeLit while in phClose
	depth   int  // JSON container depth
	inStr   bool // inside a JSON string literal
	esc     bool // previous byte was a backslash escape (inside a string)
	started bool // the JSON object's first '{' has been seen
}

// step consumes one byte, returning the next state and whether the byte is legal.
// In phText every byte is legal (free text). In phJSON/phClose the byte is legal
// only if it keeps the envelope well-formed.
func step(s state, b byte) (state, bool) {
	switch s.phase {
	case phText:
		if b == openLit[s.open] {
			s.open++
			if s.open == len(openLit) {
				s = state{phase: phJSON} // enter the call; reset JSON tracking
			}
		} else if b == openLit[0] {
			s.open = 1
		} else {
			s.open = 0
		}
		return s, true

	case phJSON:
		return jsonStep(s, b)

	case phClose:
		// Allow whitespace between the JSON and the closing tag.
		if s.closed == 0 && isSpace(b) {
			return s, true
		}
		if b == closeLit[s.closed] {
			s.closed++
			if s.closed == len(closeLit) {
				s = state{phase: phText} // call complete; back to free text
			}
			return s, true
		}
		return s, false
	}
	return s, false
}

// jsonStep tracks a JSON value's string/escape/brace state and forbids the
// malformations local models actually produce — chiefly raw control characters
// inside strings (unescaped newlines in a shell command). When the object closes,
// it transitions to requiring the closing tag.
func jsonStep(s state, b byte) (state, bool) {
	if s.inStr {
		switch {
		case s.esc:
			s.esc = false
		case b == '\\':
			s.esc = true
		case b == '"':
			s.inStr = false
		case b < 0x20:
			return s, false // raw control char in a string must be escaped
		}
		return s, true
	}

	if !s.started {
		if isSpace(b) {
			return s, true // leading whitespace before the object
		}
		if b != '{' {
			return s, false // the tool call must be a JSON object
		}
		s.started = true
		s.depth = 1
		return s, true
	}

	switch {
	case b == '"':
		s.inStr = true
	case b == '{' || b == '[':
		s.depth++
	case b == '}' || b == ']':
		s.depth--
		if s.depth < 0 {
			return s, false
		}
		if s.depth == 0 {
			s.phase = phClose
			s.closed = 0
		}
	case b < 0x20 && !isSpace(b):
		return s, false // stray control char in structure
	}
	return s, true
}

func isSpace(b byte) bool { return b == ' ' || b == '\t' || b == '\n' || b == '\r' }

// ToolCall is a Grammar enforcing the tool-call envelope. Construct with
// NewToolCall; pass it to the sampler via SampleParams.Grammar.
type ToolCall struct {
	decode Decoder
	st     state
}

// NewToolCall returns a ToolCall grammar that decodes candidate tokens with the
// given decoder (the tokenizer's single-token Decode). Free text is unconstrained;
// only a tool call the model opens on its own is forced well-formed.
func NewToolCall(decode Decoder) *ToolCall {
	return &ToolCall{decode: decode}
}

// Active reports whether the grammar is currently constraining. It returns false
// in free-text mode, so the sampler skips per-token vetting on the hot path and
// only pays for it inside a tool call.
func (g *ToolCall) Active() bool { return g.st.phase != phText }

// Allows reports whether emitting token id keeps the output within the envelope.
// Consulted only when Active is true. decode(id) must be cheap — the sampler calls
// this across the candidate set each constrained step, so back it with an O(1)
// lookup table, not a live tokenizer call.
func (g *ToolCall) Allows(id int32) bool {
	s := g.st
	str := g.decode(id)
	for i := 0; i < len(str); i++ {
		var ok bool
		if s, ok = step(s, str[i]); !ok {
			return false
		}
	}
	return true
}

// Advance commits token id, moving the grammar state forward over its bytes.
func (g *ToolCall) Advance(id int32) {
	str := g.decode(id)
	for i := 0; i < len(str); i++ {
		g.st, _ = step(g.st, str[i])
	}
}
