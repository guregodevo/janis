package serve

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// The Qwen3.5 family (Qwen3.5, Qwen3.8) is trained on tool calls in XML, its
// chat template's own form:
//
//	<tool_call>
//	<function=write_file>
//	<parameter=path>
//	hello.py
//	</parameter>
//	</function>
//	</tool_call>
//
// Asked in the Hermes JSON form, Qwen3.5-0.8B called a tool in 0 of 10
// replies and wrote the code out as prose; asked in its own form, in 10 of
// 10 (2026-11-22). Qwen3.8-27B under Memdoor answered "create a tetris game"
// with a plan and no tool call (memdoor#392). A model whose template writes
// <function= gets its tools, its past calls and its parsed calls in this
// form; every other model keeps the Hermes one.

// usesXMLTools reports whether the model in modelDir is asked for tools the
// XML way: its chat template (chat_template.jinja, else tokenizer_config's
// chat_template) writes "<function=".
func usesXMLTools(modelDir string) bool {
	if b, err := os.ReadFile(filepath.Join(modelDir, "chat_template.jinja")); err == nil {
		return strings.Contains(string(b), "<function=")
	}
	b, err := os.ReadFile(filepath.Join(modelDir, "tokenizer_config.json"))
	if err != nil {
		return false
	}
	var cfg struct {
		ChatTemplate json.RawMessage `json:"chat_template"`
	}
	if json.Unmarshal(b, &cfg) != nil {
		return false
	}
	return strings.Contains(string(cfg.ChatTemplate), "<function=")
}

// renderToolsXML is the tools block the Qwen3.5 template writes.
func renderToolsXML(tools []chatTool) string {
	var b strings.Builder
	b.WriteString("# Tools\n\nYou have access to the following functions:\n\n<tools>")
	for _, t := range tools {
		b.WriteByte('\n')
		b.Write(toolSchema(t))
	}
	b.WriteString("\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>")
	return b.String()
}

// renderCallXML is a past tool call as the template replays it: a string
// argument as it is, any other as JSON.
func renderCallXML(name, arguments string) string {
	var b strings.Builder
	b.WriteString(openTag + "\n<function=" + name + ">\n")
	var args map[string]json.RawMessage
	_ = json.Unmarshal([]byte(arguments), &args)
	for _, k := range orderedKeys(arguments, args) {
		v := args[k]
		var s string
		if json.Unmarshal(v, &s) != nil {
			s = string(v)
		}
		b.WriteString("<parameter=" + k + ">\n" + s + "\n</parameter>\n")
	}
	b.WriteString("</function>\n" + closeTag)
	return b.String()
}

// orderedKeys is the keys of a JSON object in the order the object wrote
// them: a replayed call reads as the model wrote it.
func orderedKeys(raw string, args map[string]json.RawMessage) []string {
	dec := json.NewDecoder(strings.NewReader(raw))
	var keys []string
	if t, err := dec.Token(); err != nil || t != json.Delim('{') {
		return keys
	}
	for dec.More() {
		t, err := dec.Token()
		if err != nil {
			break
		}
		k, _ := t.(string)
		var skip json.RawMessage
		if dec.Decode(&skip) != nil {
			break
		}
		if _, ok := args[k]; ok {
			keys = append(keys, k)
		}
	}
	return keys
}

var (
	xmlCallBlock = regexp.MustCompile(`(?s)<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>`)
	xmlParam     = regexp.MustCompile(`(?s)<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>`)
)

// parseCallsXML is parseCalls for the XML form. A parameter's value is text;
// one the tool's schema types as anything but a string is read as JSON when
// it parses as JSON, as vLLM's qwen3_coder parser does.
func parseCallsXML(reply string, tools []chatTool) (string, []toolCall) {
	types := paramTypes(tools)
	var calls []toolCall
	for i, m := range xmlCallBlock.FindAllStringSubmatch(reply, -1) {
		name := m[1]
		var parts []string
		for _, p := range xmlParam.FindAllStringSubmatch(m[2], -1) {
			key, val := p[1], p[2]
			var v json.RawMessage
			if t := types[name][key]; t != "" && t != "string" && json.Valid([]byte(strings.TrimSpace(val))) {
				v = json.RawMessage(strings.TrimSpace(val))
			} else {
				v, _ = json.Marshal(val)
			}
			k, _ := json.Marshal(key)
			parts = append(parts, string(k)+":"+string(v))
		}
		tc := toolCall{ID: fmt.Sprintf("call_%d", i+1), Type: "function"}
		tc.Function.Name, tc.Function.Arguments = name, "{"+strings.Join(parts, ",")+"}"
		calls = append(calls, tc)
	}
	text := xmlCallBlock.ReplaceAllString(reply, "")
	text = thinkBlock.ReplaceAllString(text, "")
	return strings.TrimSpace(text), calls
}

// paramTypes is each tool's parameters' JSON schema type, by tool and name.
func paramTypes(tools []chatTool) map[string]map[string]string {
	out := map[string]map[string]string{}
	for _, t := range tools {
		var schema struct {
			Properties map[string]struct {
				Type json.RawMessage `json:"type"`
			} `json:"properties"`
		}
		_ = json.Unmarshal(t.Function.Parameters, &schema)
		m := map[string]string{}
		for k, p := range schema.Properties {
			var s string
			if json.Unmarshal(p.Type, &s) != nil {
				s = "any" // a list of types: read as JSON when it parses
			}
			m[k] = s
		}
		out[t.Function.Name] = m
	}
	return out
}
