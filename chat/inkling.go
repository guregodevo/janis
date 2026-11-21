package chat

import (
	"bytes"
	"encoding/json"
	"regexp"
	"sort"
	"strings"
)

// Inkling 2's markers. A turn is one or more messages, each a role (with a
// tool's name after it, for a call or a result), a content type and
// <|end_message|>; the model ends its turn with
// <|content_model_end_sampling|>.
const (
	InklingModel      = "<|message_model|>"
	InklingThinking   = "<|content_thinking|>"
	InklingText       = "<|content_text|>"
	InklingCall       = "<|content_invoke_tool_json|>"
	InklingEndMessage = "<|end_message|>"
	inklingEndTurn    = "<|content_model_end_sampling|>"
)

// inklingEfforts are the template's names for a thinking effort.
var inklingEfforts = map[string]string{
	"none": "0", "minimal": "0.1", "low": "0.2", "medium": "0.7",
	"high": "0.9", "xhigh": "0.99", "max": "0.99",
}

// The Hermes forms janis's server writes into the messages, which this
// template turns back into the model's own: the tools block of the system
// prompt, a call in an assistant turn, a result in a user turn.
var (
	hermesTools    = regexp.MustCompile(`(?s)\n*# Tools\n\nYou may call one or more functions.*?<tools>\n(.*?)</tools>.*</tool_call>`)
	hermesCall     = regexp.MustCompile(`(?s)<tool_call>\s*(\{.*?\})\s*</tool_call>`)
	hermesResponse = regexp.MustCompile(`(?s)<tool_response>\n(.*?)\n</tool_response>`)
)

// hermesFiller is the system prompt the server writes when a request has
// tools and no system message of its own.
const hermesFiller = "You are a helpful assistant."

// ApplyTemplateEffort is ApplyTemplate with a thinking effort for the
// families that take one (Inkling 2: none, minimal, low, medium, high,
// xhigh, max). "" is the family's default, and so is a name it does not
// know.
func ApplyTemplateEffort(modelType string, msgs []Message, effort string) string {
	if modelType == "inkling2_loop" {
		return inkling(msgs, effort)
	}
	return ApplyTemplate(modelType, msgs)
}

// inkling renders Inkling 2's chat template (chat_template.jinja of the
// published checkpoints). The template opens no assistant turn: the model
// writes its own <|message_model|> header. Before the first message that is
// not a system one it states the thinking effort: the one asked for, else 0
// unless thinking is enabled, then the template's default (0.9).
//
// Tools reach it in the Hermes text the server renders for every family,
// and leave in the model's own messages: the declarations as a tool_declare
// system message, a call as <|message_model|>name<|content_invoke_tool_json|>,
// a result as <|message_tool|>name.
func inkling(msgs []Message, effort string) string {
	level, ok := inklingEfforts[effort]
	if !ok {
		level = "0"
		if thinkingEnabled() {
			level = "0.9"
		}
	}
	stated := false
	var b strings.Builder
	message := func(role, kind, body string) {
		b.WriteString(role)
		b.WriteString(kind)
		b.WriteString(body)
		b.WriteString(InklingEndMessage)
	}
	state := func() {
		if !stated {
			message("<|message_system|>", InklingText, "Thinking effort level: "+level)
			stated = true
		}
	}

	// The declarations come first, whatever message carried them.
	for i, m := range msgs {
		if m.Role != "system" {
			continue
		}
		block := hermesTools.FindStringSubmatch(m.Content)
		if block == nil {
			continue
		}
		if specs := inklingToolSpecs(block[1]); specs != "" {
			message("<|message_system|>tool_declare", "<|content_xml|>", specs)
		}
		rest := strings.TrimSpace(hermesTools.ReplaceAllString(m.Content, ""))
		if rest == hermesFiller {
			rest = ""
		}
		msgs = append(append(append([]Message{}, msgs[:i]...), Message{Role: "system", Content: rest}), msgs[i+1:]...)
		break
	}

	var called []string // the names of the last assistant turn's calls
	for _, m := range msgs {
		switch m.Role {
		case "system":
			if m.Content != "" {
				message("<|message_system|>", InklingText, m.Content)
			}
		case "assistant":
			state()
			calls := hermesCall.FindAllStringSubmatch(m.Content, -1)
			text := strings.TrimSpace(hermesCall.ReplaceAllString(m.Content, ""))
			if text != "" || len(calls) == 0 {
				message(InklingModel, InklingText, text)
			}
			called = called[:0]
			for _, c := range calls {
				name, body := inklingCallBody(c[1])
				called = append(called, name)
				message(InklingModel+name, InklingCall, body)
			}
			b.WriteString(inklingEndTurn)
		default:
			state()
			results := hermesResponse.FindAllStringSubmatch(m.Content, -1)
			if m.Role == "user" && len(results) > 0 && strings.HasPrefix(m.Content, "<tool_response>") {
				for i, r := range results {
					name := ""
					if i < len(called) {
						name = called[i]
					}
					message("<|message_tool|>"+name, InklingText, r[1])
				}
				continue
			}
			role := "<|message_user|>"
			if m.Role == "tool" {
				role = "<|message_tool|>"
			}
			message(role, InklingText, m.Content)
		}
	}
	state()
	return b.String()
}

// compactJSON is v as the template's tojson writes it: keys sorted, no
// spaces, no HTML escapes.
func compactJSON(v any) string {
	var out bytes.Buffer
	enc := json.NewEncoder(&out)
	enc.SetEscapeHTML(false)
	if enc.Encode(v) != nil {
		return ""
	}
	return strings.TrimSuffix(out.String(), "\n")
}

// inklingToolSpecs turns the server's tools block, one
// {"type":"function","function":{…}} a line, into the template's list.
func inklingToolSpecs(lines string) string {
	var specs []map[string]any
	for _, line := range strings.Split(strings.TrimSpace(lines), "\n") {
		var t struct {
			Type     string `json:"type"`
			Function struct {
				Name        string `json:"name"`
				Description string `json:"description"`
				Parameters  any    `json:"parameters"`
			} `json:"function"`
		}
		if json.Unmarshal([]byte(line), &t) != nil || t.Function.Name == "" {
			continue
		}
		if t.Type == "" {
			t.Type = "function"
		}
		if t.Function.Parameters == nil {
			t.Function.Parameters = map[string]any{}
		}
		specs = append(specs, map[string]any{
			"description": t.Function.Description, "name": t.Function.Name,
			"parameters": t.Function.Parameters, "type": t.Type,
		})
	}
	if len(specs) == 0 {
		return ""
	}
	return compactJSON(specs)
}

// inklingCallBody turns a Hermes call {"name":…,"arguments":{…}} into the
// model's {"name":…,"args":{…}}: the arguments' keys sorted, their values
// as they were, compacted.
func inklingCallBody(hermes string) (name, body string) {
	var c struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	_ = json.Unmarshal([]byte(hermes), &c)
	var asString string
	if json.Unmarshal(c.Arguments, &asString) == nil { // arguments sent as a JSON string
		c.Arguments = json.RawMessage(asString)
	}
	var args map[string]json.RawMessage
	_ = json.Unmarshal(c.Arguments, &args)
	keys := make([]string, 0, len(args))
	for k := range args {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var b strings.Builder
	b.WriteString(`{"name":` + compactJSON(c.Name) + `,"args":{`)
	for i, k := range keys {
		if i > 0 {
			b.WriteByte(',')
		}
		var v bytes.Buffer
		if json.Compact(&v, args[k]) != nil {
			v.Write(args[k])
		}
		b.WriteString(compactJSON(k) + ":" + plainHTML(v.String()))
	}
	b.WriteString("}}")
	return c.Name, b.String()
}

// InklingMarks are the token ids of the markers a reply is read by.
type InklingMarks struct {
	Model, Thinking, Text, Call, End int32
}

// InklingReply reads the tokens Inkling 2 generated, a run of messages, as
// the reply janis's callers expect: the text messages as they are, a tool
// call as a Hermes <tool_call> block, and the thinking apart. The markers
// decode to nothing, so decoding the tokens as one text would give the
// thoughts as the start of the answer and a call as bare JSON.
//
// A message still open gives its text so far; a call still open gives
// nothing until it closes, so the reply only ever grows as tokens come.
func InklingReply(ids []int32, m InklingMarks, decode func([]int32) string) (reply, reasoning string) {
	var text, think strings.Builder
	kind := int32(-1) // the open message's content marker, -1 for none yet
	var header, body []int32
	flush := func(closed bool) {
		switch kind {
		case m.Thinking:
			think.WriteString(decode(body))
		case m.Call:
			if !closed {
				break
			}
			if block := hermesCallBlock(decode(header), decode(body)); block != "" {
				if text.Len() > 0 {
					text.WriteByte('\n')
				}
				text.WriteString(block)
			}
		default:
			text.WriteString(decode(body))
		}
		body = body[:0]
	}
	inHeader := false
	for _, id := range ids {
		switch {
		case id == m.Model:
			flush(false)
			kind, inHeader, header = -1, true, header[:0]
		case id == m.Thinking || id == m.Text || id == m.Call:
			kind, inHeader = id, false
		case id == m.End:
			flush(true)
			kind = -1
		case inHeader:
			header = append(header, id)
		default:
			body = append(body, id)
		}
	}
	flush(false)
	return text.String(), think.String()
}

// hermesCallBlock is the model's call {"name":…,"args":{…}} as the block
// the server parses. The name falls back to the message header's.
func hermesCallBlock(header, body string) string {
	var c struct {
		Name string          `json:"name"`
		Args json.RawMessage `json:"args"`
	}
	if json.Unmarshal([]byte(strings.TrimSpace(body)), &c) != nil {
		return ""
	}
	if c.Name == "" {
		c.Name = strings.TrimSpace(header)
	}
	if c.Name == "" {
		return ""
	}
	if len(c.Args) == 0 {
		c.Args = json.RawMessage("{}")
	}
	return "<tool_call>\n" + `{"name":` + compactJSON(c.Name) + `,"arguments":` + string(c.Args) + "}\n</tool_call>"
}

// plainHTML undoes the escapes Go's encoder writes for <, > and & in JSON
// strings, which the template writes as they are. A coder's arguments are
// full of them.
func plainHTML(raw string) string {
	if !strings.Contains(raw, `\u00`) {
		return raw
	}
	plain := map[string]byte{"003c": '<', "003e": '>', "0026": '&'}
	var b strings.Builder
	for i := 0; i < len(raw); i++ {
		if raw[i] != '\\' || i+1 >= len(raw) {
			b.WriteByte(raw[i])
			continue
		}
		if raw[i+1] == 'u' && i+6 <= len(raw) {
			if c, ok := plain[strings.ToLower(raw[i+2:i+6])]; ok {
				b.WriteByte(c)
				i += 5
				continue
			}
		}
		b.WriteString(raw[i : i+2]) // an escape of another kind, whole
		i++
	}
	return b.String()
}
