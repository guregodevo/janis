package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/guregodevo/janis/hostfit"
)

// pull downloads an mlx-community model into the HuggingFace cache layout
// hostfit reads (hostfit.CacheDir()/models--mlx-community--<name>/
// snapshots/<revision>/), file by file, skipping what is already there. No
// name: the model that fits this machine's RAM and disk (pickModel). A model
// the disk has no room for is refused before a byte is fetched. HF_TOKEN is
// sent when set (gated repositories); nothing else is sent.
// hubBase is the Hub; a test points it at a local server.
var hubBase = "https://huggingface.co"

func pull(args []string) error {
	name, force := "", false
	for _, a := range args {
		if a == "--force" || a == "-f" {
			force = true
			continue
		}
		name = strings.TrimPrefix(a, "mlx-community/")
	}
	client := &http.Client{Timeout: 0}
	host, _ := hostfit.Detect()
	if name == "" {
		var err error
		if name, err = pickModel(client, host); err != nil {
			return err
		}
	} else if host.Rate(name) == hostfit.FitTooBig {
		fmt.Printf("note: %s is likely too big for %d GiB; %s would fit\n", name, host.RAMGiB, host.Recommend())
	}
	if !force {
		if installed(name) {
			fmt.Printf("✓ %s is installed · janis serve --model %s (pull --force fetches it again)\n", name, name)
			return nil
		}
	}
	repo := "mlx-community/" + name
	meta, err := repoFiles(client, repo)
	if err != nil {
		return fmt.Errorf("%s: %w", repo, err)
	}
	dir := snapshotDir(name, meta.SHA)
	if need := missingBytes(meta, dir); !roomFor(need) {
		free, _ := freeBytes(hostfit.CacheDir())
		return fmt.Errorf("%s needs %s and %s is free in %s: HF_HOME=<a disk with room> puts the models there", name, hostfit.HumanSize(need), hostfit.HumanSize(int64(free)), hostfit.CacheDir())
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	var total int64
	for _, f := range meta.Files {
		total += f.Size
	}
	fmt.Printf("%s → %s (%d files, %s)\n", repo, dir, len(meta.Files), hostfit.HumanSize(total))
	for _, f := range meta.Files {
		dst := filepath.Join(dir, f.Name)
		if st, err := os.Stat(dst); err == nil && (f.Size == 0 || st.Size() == f.Size) {
			continue
		}
		if err := download(client, hubBase+"/"+repo+"/resolve/"+meta.SHA+"/"+f.Name, dst, f.Size); err != nil {
			return fmt.Errorf("%s: %w", f.Name, err)
		}
	}
	fmt.Printf("✓ %s installed · janis serve --model %s\n", name, name)
	return nil
}

// pickModel is the model that fits this machine: its RAM tier, else the
// largest smaller tier whose download fits the disk. An installed model
// fits. The line it prints starts "no model named:", which memdoor's
// connect local shows as it is.
func pickModel(client *http.Client, host hostfit.Host) (string, error) {
	dir := hostfit.CacheDir()
	free, _ := freeBytes(dir)
	for _, name := range host.Candidates() {
		need := int64(0)
		if !installed(name) {
			meta, err := repoFiles(client, "mlx-community/"+name)
			if err != nil {
				return "", fmt.Errorf("mlx-community/%s: %w", name, err)
			}
			need = missingBytes(meta, snapshotDir(name, meta.SHA))
		}
		if roomFor(need) {
			fmt.Printf("no model named: %s fits this machine (%d GiB, %s free in %s)\n", name, host.RAMGiB, hostfit.HumanSize(int64(free)), dir)
			return name, nil
		}
		fmt.Printf("%s fits %d GiB but needs %s on disk, %s is free in %s\n", name, host.RAMGiB, hostfit.HumanSize(need), hostfit.HumanSize(int64(free)), dir)
	}
	return "", fmt.Errorf("no model fits the %s free in %s: HF_HOME=<a disk with room> puts the models there", hostfit.HumanSize(int64(free)), dir)
}

// freeBytes is hostfit.FreeBytes; a test sets a small disk.
var freeBytes = hostfit.FreeBytes

// headroom is the disk a pull leaves free beyond the model.
const headroom = 512 << 20

// roomFor is whether need more bytes fit on the cache's disk with headroom
// left. An unknown free space lets the pull go ahead.
func roomFor(need int64) bool {
	if need == 0 {
		return true
	}
	free, err := freeBytes(hostfit.CacheDir())
	return err != nil || uint64(need)+headroom <= free
}

// snapshotDir is where a model's files go in the cache layout hostfit reads.
func snapshotDir(name, sha string) string {
	return filepath.Join(hostfit.CacheDir(), "models--mlx-community--"+name, "snapshots", sha)
}

// missingBytes is the disk a pull of meta into dir still takes: the files it
// has to fetch, less the partial ones (.part) it starts over, whose room
// comes back when it does.
func missingBytes(meta repoMeta, dir string) int64 {
	var n int64
	for _, f := range meta.Files {
		dst := filepath.Join(dir, f.Name)
		if st, err := os.Stat(dst); err == nil && (f.Size == 0 || st.Size() == f.Size) {
			continue
		}
		n += f.Size
		if st, err := os.Stat(dst + ".part"); err == nil {
			n -= min(st.Size(), f.Size)
		}
	}
	return n
}

// installed is whether name is in the HuggingFace cache already.
func installed(name string) bool {
	list, _ := hostfit.ListInstalled()
	return hasModel(list, name)
}

func hasModel(list []hostfit.InstalledModel, name string) bool {
	for _, m := range list {
		if m.Name == name {
			return true
		}
	}
	return false
}

type repoMeta struct {
	SHA   string
	Files []struct {
		Name string
		Size int64
	}
}

// repoFiles reads the repository's revision and file list from the Hub API.
func repoFiles(client *http.Client, repo string) (repoMeta, error) {
	req, _ := http.NewRequest(http.MethodGet, hubBase+"/api/models/"+repo+"?blobs=true", nil)
	hfAuth(req)
	resp, err := client.Do(req)
	if err != nil {
		return repoMeta{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return repoMeta{}, fmt.Errorf("the Hub answered %s (a typo in the name, or a gated model without HF_TOKEN)", resp.Status)
	}
	var v struct {
		SHA      string `json:"sha"`
		Siblings []struct {
			Name string `json:"rfilename"`
			Size int64  `json:"size"`
		} `json:"siblings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&v); err != nil {
		return repoMeta{}, err
	}
	m := repoMeta{SHA: v.SHA}
	for _, s := range v.Siblings {
		if strings.HasPrefix(s.Name, ".") {
			continue
		}
		m.Files = append(m.Files, struct {
			Name string
			Size int64
		}{s.Name, s.Size})
	}
	if m.SHA == "" || len(m.Files) == 0 {
		return repoMeta{}, fmt.Errorf("the Hub lists no files")
	}
	return m, nil
}

func hfAuth(req *http.Request) {
	if tok := strings.TrimSpace(os.Getenv("HF_TOKEN")); tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
	req.Header.Set("User-Agent", "janis")
}

// progressOut is where download draws its line; a test reads it.
var progressOut io.Writer = os.Stdout

// progress redraws a file's line: its share of size, or the bytes so far
// when the size is unknown.
func progress(name string, done, size int64) {
	if size > 0 {
		fmt.Fprintf(progressOut, "\r  %-48s %5.1f%% of %s", name, 100*float64(done)/float64(size), hostfit.HumanSize(size))
		return
	}
	fmt.Fprintf(progressOut, "\r  %-48s %s", name, hostfit.HumanSize(done))
}

// download fetches one file to dst through a temporary name, printing
// progress on one line. The line is drawn a last time at the end: a file
// done within one redraw interval had none, and the read that returns EOF
// carries no bytes, so the last redraw stopped short of 100%.
func download(client *http.Client, url, dst string, size int64) error {
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	req, _ := http.NewRequest(http.MethodGet, url, nil)
	hfAuth(req)
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("the Hub answered %s", resp.Status)
	}
	tmp := dst + ".part"
	out, err := os.Create(tmp)
	if err != nil {
		return err
	}
	name := filepath.Base(dst)
	var done int64
	last := time.Now()
	buf := make([]byte, 1<<20)
	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := out.Write(buf[:n]); werr != nil {
				out.Close()
				return werr
			}
			done += int64(n)
			if time.Since(last) > 200*time.Millisecond {
				progress(name, done, size)
				last = time.Now()
			}
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			out.Close()
			return rerr
		}
	}
	progress(name, done, size)
	fmt.Fprintln(progressOut)
	if err := out.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, dst)
}
