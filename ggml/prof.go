//go:build ggml

package ggml

import (
	"fmt"
	"os"
	"runtime"
	"sort"
	"strings"
	"time"
)

// prof (JANIS_GGML_PROF=1) computes every result as a graph of its own,
// waits for the device and adds its time to the backend method that
// recorded it; each sweep prints the methods by time. For finding where a
// forward pass goes: a pass is several times slower this way, and a decode
// step is not recorded.
var prof = os.Getenv("JANIS_GGML_PROF") == "1"

type profEntry struct {
	n     int
	dur   time.Duration
	bytes int64
}

var profAcc = map[string]profEntry{}

// profSite names the backend method that called op, and its caller outside
// the backend.
func profSite() string {
	pc := make([]uintptr, 12)
	n := runtime.Callers(3, pc)
	fs := runtime.CallersFrames(pc[:n])
	var inner, outer string
	for {
		f, more := fs.Next()
		name := f.Function[strings.LastIndex(f.Function, "/")+1:]
		if strings.HasPrefix(name, "ggml.") {
			inner = strings.TrimPrefix(strings.TrimPrefix(name, "ggml.(*Backend)."), "ggml.")
		} else if outer == "" {
			outer = name
		}
		if !more || outer != "" {
			break
		}
	}
	return inner + " < " + outer
}

func profPrint() {
	if len(profAcc) == 0 {
		return
	}
	type row struct {
		site string
		profEntry
	}
	var rows []row
	var total time.Duration
	n := 0
	for s, e := range profAcc {
		rows = append(rows, row{s, e})
		total += e.dur
		n += e.n
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].dur > rows[j].dur })
	fmt.Fprintf(os.Stderr, "ggml prof: %d results, %.0f ms\n", n, total.Seconds()*1000)
	for i, r := range rows {
		if i == 28 {
			break
		}
		fmt.Fprintf(os.Stderr, "ggml prof: %7.1f ms %5d x %8.1f MiB  %s\n", r.dur.Seconds()*1000, r.n, float64(r.bytes)/(1<<20), r.site)
	}
	profAcc = map[string]profEntry{}
}
