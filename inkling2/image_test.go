package inkling2

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"math"
	"testing"
)

func pngOf(t *testing.T, img image.Image) []byte {
	t.Helper()
	var b bytes.Buffer
	if err := png.Encode(&b, img); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

// A picture is padded up to whole blocks with the reference's values:
// colours on [-1, 1], padding at -1 on the 0-255 scale, a transparent
// pixel on the gray checkerboard.
func TestPrepareImagePadsAndNormalizes(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 5, 3))
	for y := 0; y < 3; y++ {
		for x := 0; x < 5; x++ {
			img.Set(x, y, color.NRGBA{255, 0, 51, 255})
		}
	}
	img.Set(4, 2, color.NRGBA{0, 0, 0, 0}) // clear: the checkerboard's 204
	pixels, h, w, err := PrepareImage(pngOf(t, img), 4)
	if err != nil {
		t.Fatal(err)
	}
	if h != 4 || w != 8 || len(pixels) != 3*4*8 {
		t.Fatalf("%dx%d, %d values; want 8x4, 96", w, h, len(pixels))
	}
	at := func(ch, y, x int) float64 { return float64(pixels[ch*h*w+y*w+x]) }
	near := func(name string, got, want float64) {
		t.Helper()
		if math.Abs(got-want) > 1e-6 {
			t.Errorf("%s = %v, want %v", name, got, want)
		}
	}
	near("red", at(0, 0, 0), 1)
	near("green", at(1, 0, 0), -1)
	near("blue", at(2, 0, 0), (51.0/255-0.5)/0.5)
	near("the clear pixel", at(0, 2, 4), (204.0/255-0.5)/0.5)
	near("padding to the right", at(0, 0, 5), (-1.0/255-0.5)/0.5)
	near("padding below", at(1, 3, 0), (-1.0/255-0.5)/0.5)
}

// A picture of more than MaxImageTokens blocks is scaled down to fit.
func TestPrepareImageScalesALargePictureDown(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 400, 300))
	for i := range img.Pix {
		img.Pix[i] = 255
	}
	_, h, w, err := PrepareImage(pngOf(t, img), 4) // 7,500 blocks as it is
	if err != nil {
		t.Fatal(err)
	}
	if got := (h / 4) * (w / 4); got > MaxImageTokens || got < MaxImageTokens*8/10 {
		t.Errorf("%dx%d is %d tokens, want at most %d and close to it", w, h, got, MaxImageTokens)
	}
	if ratio := float64(w) / float64(h); math.Abs(ratio-4.0/3) > 0.1 {
		t.Errorf("%dx%d does not keep the 4:3 shape", w, h)
	}
	if _, _, _, err := PrepareImage([]byte("hello"), 4); err == nil {
		t.Error("bytes that are no image were decoded")
	}
}
