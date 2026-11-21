package inkling2

import (
	"bytes"
	"fmt"
	"image"
	_ "image/gif" // the formats an OpenAI client sends
	_ "image/jpeg"
	_ "image/png"
	"math"
)

// MaxImageTokens bounds one image. The reference encodes an image at its
// own size, 40 pixels a token, so a 12-megapixel photo would be 7,500
// tokens; janis scales a larger image down to fit.
const MaxImageTokens = 1024

// PrepareImage decodes a PNG, JPEG or GIF into what the vision tower
// reads: [3, height, width] row-major floats in [-1, 1], both sides padded
// up to a multiple of block. As the reference, a transparent pixel shows a
// gray checkerboard and the padding is -1 on the 0-255 scale.
func PrepareImage(data []byte, block int) (pixels []float32, height, width int, err error) {
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, 0, 0, fmt.Errorf("decode image: %w", err)
	}
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	if w == 0 || h == 0 {
		return nil, 0, 0, fmt.Errorf("the image is empty")
	}
	// 8-bit colour, not premultiplied, the checkerboard under what is
	// transparent.
	rgb := make([]float64, w*h*3)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, a := img.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA() // premultiplied, 16-bit
			c := [3]uint32{r, g, b}
			background := 153.0
			if (x/8+y/8)%2 == 0 {
				background = 204
			}
			alpha := float64(a>>8) / 255
			for ch := 0; ch < 3; ch++ {
				v := float64(c[ch] >> 8)
				if a != 0xffff {
					v = math.Floor(v + background*(1-alpha) + 0.5)
				}
				rgb[(y*w+x)*3+ch] = v
			}
		}
	}
	tokens := func(w, h int) int { return ((w + block - 1) / block) * ((h + block - 1) / block) }
	if tokens(w, h) > MaxImageTokens {
		scale := math.Sqrt(float64(MaxImageTokens*block*block) / float64(w*h))
		nw, nh := max(int(float64(w)*scale), 1), max(int(float64(h)*scale), 1)
		for tokens(nw, nh) > MaxImageTokens {
			nw, nh = max(nw*99/100, 1), max(nh*99/100, 1)
		}
		rgb, w, h = shrink(rgb, w, h, nw, nh), nw, nh
	}
	width, height = (w+block-1)/block*block, (h+block-1)/block*block
	pixels = make([]float32, 3*height*width)
	for i := range pixels {
		pixels[i] = float32((-1.0/255 - 0.5) / 0.5)
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			for ch := 0; ch < 3; ch++ {
				pixels[ch*height*width+y*width+x] = float32((rgb[(y*w+x)*3+ch]/255 - 0.5) / 0.5)
			}
		}
	}
	return pixels, height, width, nil
}

// shrink scales rgb [h, w, 3] down to nw by nh, each new pixel the mean of
// the source pixels it covers.
func shrink(rgb []float64, w, h, nw, nh int) []float64 {
	out := make([]float64, nw*nh*3)
	for y := 0; y < nh; y++ {
		y0, y1 := y*h/nh, max((y+1)*h/nh, y*h/nh+1)
		for x := 0; x < nw; x++ {
			x0, x1 := x*w/nw, max((x+1)*w/nw, x*w/nw+1)
			for ch := 0; ch < 3; ch++ {
				sum := 0.0
				for sy := y0; sy < y1; sy++ {
					for sx := x0; sx < x1; sx++ {
						sum += rgb[(sy*w+sx)*3+ch]
					}
				}
				out[(y*nw+x)*3+ch] = sum / float64((y1-y0)*(x1-x0))
			}
		}
	}
	return out
}
