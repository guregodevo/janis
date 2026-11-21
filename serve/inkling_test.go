package serve

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/guregodevo/janis/chat"
)

// INKLING 2 SEES TOOLS AS ITS OWN TEMPLATE WRITES THEM.
//
// The server renders tools the Hermes way for every family. For Inkling 2
// the chat template turns that text back into the model's own messages.
// testdata/inkling2_tools.json holds OpenAI requests (tools, a call, two
// calls with text, results) and what the published chat_template.jinja
// renders for each.
func TestInklingSeesToolsAsItsTemplateWritesThem(t *testing.T) {
	raw, err := os.ReadFile("testdata/inkling2_tools.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases map[string]struct {
		Request chatRequest `json:"request"`
		Want    string      `json:"want"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases) == 0 {
		t.Fatal("no cases")
	}
	for name, c := range cases {
		if got := chat.ApplyTemplateEffort("inkling2_loop", renderChat(c.Request), "none"); got != c.Want {
			t.Errorf("%s:\n got %q\nwant %q", name, got, c.Want)
		}
	}
}
