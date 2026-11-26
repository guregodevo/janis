package serve

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/guregodevo/janis/chat"
)

// The prompt janis writes for a Qwen3.5-family model is the one the model's
// own chat template writes, byte for byte: a conversation with tools, calls
// and results; a plain request; a system prompt with no tools. The .txt
// files are the template's output (testdata/qwen3_5/render.py).
func TestTheQwen35PromptIsItsTemplates(t *testing.T) {
	for _, name := range []string{"tools", "plain", "sysonly"} {
		t.Run(name, func(t *testing.T) {
			b, err := os.ReadFile(filepath.Join("testdata", "qwen3_5", name+".json"))
			if err != nil {
				t.Fatal(err)
			}
			var req chatRequest
			if err := json.Unmarshal(b, &req); err != nil {
				t.Fatal(err)
			}
			want, err := os.ReadFile(filepath.Join("testdata", "qwen3_5", name+".txt"))
			if err != nil {
				t.Fatal(err)
			}
			got := chat.ApplyTemplate("qwen3_5", renderChatFor(req, true))
			if got != string(want) {
				i := 0
				for i < len(got) && i < len(want) && got[i] == want[i] {
					i++
				}
				lo := max(i-80, 0)
				t.Fatalf("differs from the template at byte %d\njanis:    %q\ntemplate: %q", i, got[lo:min(i+80, len(got))], string(want)[lo:min(i+80, len(want))])
			}
		})
	}
}
