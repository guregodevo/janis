package llm

import (
	"fmt"
	"log"
	"os"
	"sort"
	"strings"
	"time"
)

// batchStats is where the batched replies' time goes, logged every
// statsEvery with JANIS_BATCH_STATS=1: each phase's count and time, the
// tokens prefilled and decoded, and the steps by how they ran (replay,
// record, plain) and by batch size. Off, nothing is timed.
type batchStats struct {
	since   time.Time
	phase   map[string]*phaseStat
	sizes   map[int]int
	prefill int // tokens prefilled
	decoded int // tokens decoded by steps
	changes int // steps whose rows were not the last step's
}

type phaseStat struct {
	n int
	d time.Duration
}

const statsEvery = 10 * time.Second

var batchStatsOn = os.Getenv("JANIS_BATCH_STATS") == "1"

func newBatchStats() *batchStats {
	if !batchStatsOn {
		return nil
	}
	return &batchStats{since: time.Now(), phase: map[string]*phaseStat{}, sizes: map[int]int{}}
}

// add counts d spent in phase.
func (s *batchStats) add(phase string, d time.Duration) {
	if s == nil {
		return
	}
	p := s.phase[phase]
	if p == nil {
		p = &phaseStat{}
		s.phase[phase] = p
	}
	p.n++
	p.d += d
}

// step counts a batched step of size rows that ran as kind.
func (s *batchStats) step(kind string, rows int, d time.Duration) {
	if s == nil {
		return
	}
	s.add("step "+kind, d)
	s.sizes[rows]++
	s.decoded += rows
}

// flush logs the window and starts the next one, once statsEvery passed.
func (s *batchStats) flush() {
	if s == nil || time.Since(s.since) < statsEvery {
		return
	}
	wall := time.Since(s.since)
	var names []string
	var busy time.Duration
	for k, p := range s.phase {
		names = append(names, k)
		busy += p.d
	}
	sort.Strings(names)
	var b strings.Builder
	fmt.Fprintf(&b, "batch stats %.1fs: decoded %d tok (%.0f tok/s), prefilled %d tok, %d row changes, engine busy %.0f%%",
		wall.Seconds(), s.decoded, float64(s.decoded)/wall.Seconds(), s.prefill, s.changes, 100*busy.Seconds()/wall.Seconds())
	for _, k := range names {
		p := s.phase[k]
		fmt.Fprintf(&b, " | %s %d× %.2fs (%.1f ms each)", k, p.n, p.d.Seconds(), 1000*p.d.Seconds()/float64(p.n))
	}
	var sz []int
	for n := range s.sizes {
		sz = append(sz, n)
	}
	sort.Ints(sz)
	b.WriteString(" | steps by rows")
	for _, n := range sz {
		fmt.Fprintf(&b, " %d:%d", n, s.sizes[n])
	}
	log.Print(b.String())
	*s = *newBatchStats()
}
