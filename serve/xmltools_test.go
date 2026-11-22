package serve

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// installXMLFake is installFake for a model whose chat template asks for
// tools in the XML form (the Qwen3.5 family).
func installXMLFake(t *testing.T, name string) {
	t.Helper()
	installFake(t, name, 262144)
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, ".cache/huggingface/hub/models--mlx-community--"+name, "snapshots", "abc")
	tpl := "{{- '<tool_call>\\n<function=' + tool_call.name + '>\\n' }}"
	if err := os.WriteFile(filepath.Join(dir, "chat_template.jinja"), []byte(tpl), 0o644); err != nil {
		t.Fatal(err)
	}
}

const xmlTools = `[{"type":"function","function":{"name":"write_file","description":"Write a file","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"},"mode":{"type":"integer"},"force":{"type":"boolean"}},"required":["path","content"]}}}]`

// A model trained on XML tool calls is asked in its own form, its calls come
// back as OpenAI tool_calls with arguments typed by the schema, the JSON
// grammar stays off, and the next turn replays the call in the XML form.
func TestAnXMLToolModelCallsToolsInItsOwnForm(t *testing.T) {
	installXMLFake(t, "Qwen3.5-0.8B-MLX-4bit")
	eng := &fakeEngine{reply: "I will write it.\n<tool_call>\n<function=write_file>\n<parameter=path>\nhello.py\n</parameter>\n<parameter=content>\nprint(\"hello\")\nprint(1 < 2)\n</parameter>\n<parameter=mode>\n420\n</parameter>\n<parameter=force>\ntrue\n</parameter>\n</function>\n</tool_call>"}
	srv := newTestServer(t, eng)
	body := `{"model":"Qwen3.5-0.8B-MLX-4bit","tools":` + xmlTools + `,"messages":[{"role":"user","content":"create hello.py"}]}`

	resp, v := post(t, srv.URL+"/v1/chat/completions", body)
	if resp.StatusCode != 200 {
		t.Fatalf("status %d: %v", resp.StatusCode, v)
	}
	choice := v["choices"].([]any)[0].(map[string]any)
	msg := choice["message"].(map[string]any)
	calls, _ := msg["tool_calls"].([]any)
	if choice["finish_reason"] != "tool_calls" || len(calls) != 1 || msg["content"] != "I will write it." {
		t.Fatalf("an XML call is a tool call: %v", choice)
	}
	fn := calls[0].(map[string]any)["function"].(map[string]any)
	var args map[string]any
	if err := json.Unmarshal([]byte(fn["arguments"].(string)), &args); err != nil {
		t.Fatalf("arguments are JSON: %v %q", err, fn["arguments"])
	}
	if fn["name"] != "write_file" || args["path"] != "hello.py" || args["content"] != "print(\"hello\")\nprint(1 < 2)" || args["mode"] != 420.0 || args["force"] != true {
		t.Fatalf("arguments typed by the schema: %v", args)
	}
	sys := eng.got[0].Content
	if !strings.Contains(sys, "You have access to the following functions") || !strings.Contains(sys, "<function=example_function_name>") || strings.Contains(sys, "return a json object") || eng.opts.ToolCallGrammar {
		t.Fatalf("the tools in the XML form, no JSON grammar: %q %+v", sys, eng.opts)
	}

	// The next turn replays the call as the template writes it.
	eng.reply = "Done."
	followUp := `{"model":"Qwen3.5-0.8B-MLX-4bit","tools":` + xmlTools + `,"messages":[{"role":"user","content":"create hello.py"},{"role":"assistant","content":"I will write it.","tool_calls":[{"id":"call_1","type":"function","function":{"name":"write_file","arguments":"{\"path\":\"hello.py\",\"content\":\"print(1)\",\"mode\":420}"}}]},{"role":"tool","tool_call_id":"call_1","content":"wrote hello.py"}]}`
	post(t, srv.URL+"/v1/chat/completions", followUp)
	want := "I will write it.\n<tool_call>\n<function=write_file>\n<parameter=path>\nhello.py\n</parameter>\n<parameter=content>\nprint(1)\n</parameter>\n<parameter=mode>\n420\n</parameter>\n</function>\n</tool_call>"
	if len(eng.got) != 4 || eng.got[2].Content != want || eng.got[3].Content != "<tool_response>\nwrote hello.py\n</tool_response>" {
		t.Fatalf("the transcript replayed in the XML form:\n%q\nwant\n%q", eng.got[2].Content, want)
	}

	// Streamed, the call's tag never leaks as text.
	eng.reply = "<tool_call>\n<function=write_file>\n<parameter=path>\na.py\n</parameter>\n<parameter=content>\nx\n</parameter>\n</function>\n</tool_call>"
	resp, _ = post(t, srv.URL+"/v1/chat/completions", strings.Replace(body, `"model"`, `"stream":true,"model"`, 1))
	text, finish, _, _ := readStream(t, resp)
	if text != "" || finish != "tool_calls" {
		t.Fatalf("streamed XML call: text=%q finish=%q", text, finish)
	}

	// A Hermes call the model writes anyway is still a call.
	eng.reply = "<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"path\": \"b.py\", \"content\": \"y\"}}\n</tool_call>"
	_, v = post(t, srv.URL+"/v1/chat/completions", body)
	if c := v["choices"].([]any)[0].(map[string]any); c["finish_reason"] != "tool_calls" {
		t.Fatalf("a Hermes call from an XML model: %v", c)
	}
}

// The tools block is the template's, word for word: checked against an
// installed Qwen3.5 template when there is one.
func TestTheXMLToolsBlockIsTheTemplates(t *testing.T) {
	home, _ := os.UserHomeDir()
	dirs, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--mlx-community--Qwen3.5-*/snapshots/*/chat_template.jinja"))
	if len(dirs) == 0 {
		t.Skip("no Qwen3.5 template installed")
	}
	b, err := os.ReadFile(dirs[0])
	if err != nil {
		t.Fatal(err)
	}
	tpl := string(b)
	at := strings.Index(tpl, `'\n\nIf you choose to call a function`)
	if at < 0 {
		t.Fatal("the template's call instructions are not where this test looks")
	}
	end := strings.Index(tpl[at:], `</IMPORTANT>'`)
	literal := strings.ReplaceAll(tpl[at+1:at+end+len("</IMPORTANT>")], `\n`, "\n")
	got := renderToolsXML([]chatTool{})
	if !strings.Contains(got, literal) || !strings.HasPrefix(got, "# Tools\n\nYou have access to the following functions:\n\n<tools>") {
		t.Fatalf("the tools block differs from %s", dirs[0])
	}
	if !usesXMLTools(filepath.Dir(dirs[0])) {
		t.Fatal("an installed Qwen3.5 is not seen as an XML tool model")
	}
}
