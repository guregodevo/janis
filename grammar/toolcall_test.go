package grammar

import "testing"

// charDecoder treats each token id as a single byte, so a test can drive the
// grammar with an arbitrary byte stream (one id per byte).
func charDecoder(id int32) string { return string([]byte{byte(id)}) }

// feed pushes s through the grammar one byte at a time, checking Allows before
// each Advance while the grammar is Active. Returns the index of the first byte
// the grammar rejected, or -1 if the whole string was accepted.
func feed(g *ToolCall, s string) int {
	for i := 0; i < len(s); i++ {
		id := int32(s[i])
		if g.Active() && !g.Allows(id) {
			return i
		}
		g.Advance(id)
	}
	return -1
}

func TestFreeTextIsUnconstrained(t *testing.T) {
	g := NewToolCall(charDecoder)
	if g.Active() {
		t.Fatal("grammar should start inactive (free text)")
	}
	if i := feed(g, "here is a plain answer with {braces} and \"quotes\" and \n newlines"); i != -1 {
		t.Fatalf("free text must never be rejected, failed at byte %d", i)
	}
	if g.Active() {
		t.Fatal("still free text — should be inactive")
	}
}

func TestWellFormedCallAccepted(t *testing.T) {
	g := NewToolCall(charDecoder)
	call := `sure<tool_call>{"name":"bash","arguments":{"command":"ls -la"}}</tool_call>`
	if i := feed(g, call); i != -1 {
		t.Fatalf("well-formed call rejected at byte %d (%q)", i, call[i:i+1])
	}
	if g.Active() {
		t.Fatal("after the closing tag the grammar should be back in free text")
	}
}

func TestActivatesOnlyAfterOpenLiteral(t *testing.T) {
	g := NewToolCall(charDecoder)
	feed(g, "talking about <tool_call> in prose is fine until it is actually opened")
	// The literal above DID open a call (the exact "<tool_call>" appears), so the
	// grammar is now constraining. Confirm it engaged.
	if !g.Active() {
		t.Fatal("emitting the open literal should activate the grammar")
	}
}

func TestRawControlCharInStringRejected(t *testing.T) {
	g := NewToolCall(charDecoder)
	// Open a call and get into the "command" string value.
	prefix := `<tool_call>{"name":"bash","arguments":{"command":"`
	if i := feed(g, prefix); i != -1 {
		t.Fatalf("prefix rejected at %d", i)
	}
	if !g.Active() {
		t.Fatal("should be constraining inside the JSON string")
	}
	// A raw newline inside the string is illegal (must be \n); reject it.
	if g.Allows('\n') {
		t.Fatal("raw newline inside a JSON string must be rejected")
	}
	// The escaped form is fine: backslash then n.
	if !g.Allows('\\') {
		t.Fatal("backslash (starting an escape) must be allowed")
	}
	g.Advance('\\')
	if !g.Allows('n') {
		t.Fatal("escaped \\n must be allowed")
	}
}

func TestClosingTagRequiredAfterJSON(t *testing.T) {
	g := NewToolCall(charDecoder)
	if i := feed(g, `<tool_call>{"name":"bash","arguments":{}}`); i != -1 {
		t.Fatalf("valid JSON body rejected at %d", i)
	}
	if !g.Active() {
		t.Fatal("after the JSON closes the closing tag is still required — must stay active")
	}
	// Whitespace before the tag is allowed; arbitrary prose is not.
	if !g.Allows(' ') {
		t.Fatal("whitespace before the closing tag should be allowed")
	}
	if g.Allows('x') {
		t.Fatal("prose after the JSON (before the closing tag) must be rejected")
	}
	if i := feed(g, `</tool_call>`); i != -1 {
		t.Fatalf("closing tag rejected at %d", i)
	}
	if g.Active() {
		t.Fatal("call complete — grammar should be inactive again")
	}
}

func TestBodyMustBeObject(t *testing.T) {
	g := NewToolCall(charDecoder)
	feed(g, `<tool_call>`)
	if !g.Active() {
		t.Fatal("open literal should have activated the grammar")
	}
	// Leading whitespace is fine, but the first real char must be '{'.
	if !g.Allows(' ') {
		t.Fatal("leading whitespace before the object should be allowed")
	}
	if g.Allows('[') {
		t.Fatal("a tool call body must be a JSON object, not an array")
	}
	if !g.Allows('{') {
		t.Fatal("'{' must open the object")
	}
}

// mapDecoder returns registered multi-byte strings, to test that the open literal
// is recognized even when it arrives as a single (multi-byte) token.
func TestOpenLiteralAsSingleToken(t *testing.T) {
	vocab := map[int32]string{1: "<tool_call>", 2: `{"name":"x","arguments":{}}`, 3: "</tool_call>"}
	g := NewToolCall(func(id int32) string { return vocab[id] })
	g.Advance(1)
	if !g.Active() {
		t.Fatal("a single-token '<tool_call>' should activate the grammar")
	}
	if !g.Allows(2) {
		t.Fatal("a well-formed object token should be allowed")
	}
	g.Advance(2)
	if !g.Allows(3) {
		t.Fatal("the closing-tag token should be allowed")
	}
	g.Advance(3)
	if g.Active() {
		t.Fatal("after the closing token the grammar should be inactive")
	}
}

// The live 30B runaway: unbounded whitespace in the forced-open and close
// states let generation stall forever. Whitespace past the budget must be
// ILLEGAL so the only legal bytes make progress.
func TestForcedGrammarBoundsWhitespace(t *testing.T) {
	g := NewToolCallForced(charDecoder)
	ws := int32(' ')
	for i := 0; i < forceOpenWSBudget; i++ {
		if !g.Allows(ws) {
			t.Fatalf("whitespace byte %d within budget must be legal", i)
		}
		g.Advance(ws)
	}
	if g.Allows(ws) {
		t.Fatal("whitespace past the forced-open budget must be illegal")
	}
	if i := feed(g, "<tool_call>{\"name\":\"x\",\"arguments\":{}}"); i != -1 {
		t.Fatalf("tag+json must stay legal after bounded whitespace, failed at %d", i)
	}
	nl := int32('\n')
	for i := 0; i < closeWSBudget; i++ {
		if !g.Allows(nl) {
			t.Fatalf("close whitespace %d within budget must be legal", i)
		}
		g.Advance(nl)
	}
	if g.Allows(nl) {
		t.Fatal("whitespace past the close budget must be illegal")
	}
	if i := feed(g, "</tool_call>"); i != -1 {
		t.Fatalf("closing tag must stay legal, failed at %d", i)
	}
}
