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

// A PICTURE COMES AS ITS BYTES, OR NOT AT ALL.
func TestContentImagesReadsDataURLsOnly(t *testing.T) {
	images, err := contentImages(json.RawMessage(`[{"type":"text","text":"x"},{"type":"image_url","image_url":{"url":"data:image/png;base64,aGVsbG8="}},{"type":"input_image","image_url":"data:image/jpeg;base64,aGk="}]`))
	if err != nil || len(images) != 2 || string(images[0]) != "hello" || string(images[1]) != "hi" {
		t.Errorf("images %q, err %v", images, err)
	}
	if images, err := contentImages(json.RawMessage(`"plain text"`)); err != nil || images != nil {
		t.Errorf("a plain string gave %q, %v", images, err)
	}
	for _, bad := range []string{
		`[{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}]`,
		`[{"type":"image_url","image_url":{"url":"data:image/png;base64,***"}}]`,
	} {
		if _, err := contentImages(json.RawMessage(bad)); err == nil {
			t.Errorf("%s was accepted", bad)
		}
	}
}

// A request with a picture for a model that reads none is refused.
func TestAPictureForAModelThatReadsNoneIsRefused(t *testing.T) {
	var req chatRequest
	if err := json.Unmarshal([]byte(`{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,aGVsbG8="}}]}]}`), &req); err != nil {
		t.Fatal(err)
	}
	s := &Server{name: "text-only"}
	if err := s.checkImages(req, renderChat(req)); err == nil {
		t.Error("the request was accepted")
	}
	var plain chatRequest
	_ = json.Unmarshal([]byte(`{"messages":[{"role":"user","content":"hi"}]}`), &plain)
	if err := s.checkImages(plain, renderChat(plain)); err != nil {
		t.Errorf("a request with no picture was refused: %v", err)
	}
}
