package llm

import (
	"fmt"
	"testing"
)

// A THINKING MESSAGE IS NOT PART OF THE REPLY.
//
// Inkling 2 writes its thoughts as a message of their own, between two
// markers that decode to nothing: left in, the thoughts would read as the
// start of the answer. A message still open (a stream in the middle of a
// thought) shows nothing yet.
func TestAThinkingMessageIsNotPartOfTheReply(t *testing.T) {
	const model, text, think, end = 1, 2, 8, 10
	e := &Engine{hideOpen: think, hideClose: end, hides: true}
	for _, tc := range []struct{ in, want []int32 }{
		{[]int32{model, text, 50, 51, end}, []int32{model, text, 50, 51, end}},
		{[]int32{model, think, 60, 61, end, model, text, 50, end}, []int32{model, model, text, 50, end}},
		{[]int32{model, think, 60, 61}, []int32{model}},
		{nil, []int32{}},
	} {
		if got := e.visible(tc.in); fmt.Sprint(got) != fmt.Sprint(tc.want) {
			t.Errorf("visible(%v) = %v, want %v", tc.in, got, tc.want)
		}
	}
	plain := &Engine{}
	if got := plain.visible([]int32{think, 60, end}); len(got) != 3 {
		t.Errorf("a family with no hidden messages lost tokens: %v", got)
	}
}
