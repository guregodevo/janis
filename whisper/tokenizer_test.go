package whisper

import "testing"

func TestTokenizerRoundTrip(t *testing.T) {
	tk, err := NewTokenizer()
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range []string{"Hello world.", " The death, then, of a beautiful woman", "الصيام المتقطع", "こんにちは 世界"} {
		if got := tk.Decode(tk.Encode(s)); got != s {
			t.Errorf("round trip %q → %q", s, got)
		}
	}
	if LangToken("en") != 50259 || LangToken("ar") != 50272 || LangToken("yue") != 50358 {
		t.Errorf("language tokens: en=%d ar=%d yue=%d", LangToken("en"), LangToken("ar"), LangToken("yue"))
	}
	if !IsTimestamp(TokTimestampBase+50) || Timestamp(TokTimestampBase+50) != 1.0 {
		t.Error("timestamp tokens")
	}
	if s := tk.DecodeWithSpecial([]int{TokSOT, LangToken("en"), TokTranscribe, TokTimestampBase}); s != "<|startoftranscript|><|en|><|transcribe|><|0.00|>" {
		t.Errorf("specials: %q", s)
	}
}
