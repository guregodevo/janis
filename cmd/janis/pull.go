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

	llm "github.com/guregodevo/janis"
	"github.com/guregodevo/janis/hostfit"
)

// pull downloads an mlx-community model into the HuggingFace cache layout
// hostfit reads (~/.cache/huggingface/hub/models--mlx-community--<name>/
// snapshots/<revision>/), file by file, skipping what is already there. No
// name: the model hostfit recommends for this machine's RAM. HF_TOKEN is
// sent when set (gated repositories); nothing else is sent.
// hubBase is the Hub; a test points it at a local server.
var hubBase = "https://huggingface.co"

func pull(args []string) {
	name, force := "", false
	for _, a := range args {
		if a == "--force" || a == "-f" {
			force = true
			continue
		}
		name = strings.TrimPrefix(a, "mlx-community/")
	}
	host, _ := hostfit.Detect(llm.HasLocalEngine())
	if name == "" {
		name = host.Recommend()
		fmt.Printf("no model named: %s fits this machine (%d GiB)\n", name, host.RAMGiB)
	} else if host.Rate(name) == hostfit.FitTooBig {
		fmt.Printf("note: %s is likely too big for %d GiB; %s would fit\n", name, host.RAMGiB, host.Recommend())
	}
	if !force {
		if installed, _ := hostfit.ListInstalled(); hasModel(installed, name) {
			fmt.Printf("✓ %s is installed · janis serve --model %s (pull --force fetches it again)\n", name, name)
			return
		}
	}
	repo := "mlx-community/" + name
	client := &http.Client{Timeout: 0}
	meta, err := repoFiles(client, repo)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s: %v\n", repo, err)
		os.Exit(1)
	}
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, ".cache/huggingface/hub/models--mlx-community--"+name, "snapshots", meta.SHA)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
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
			fmt.Fprintf(os.Stderr, "\n%s: %v\n", f.Name, err)
			os.Exit(1)
		}
	}
	fmt.Printf("✓ %s installed · janis serve --model %s\n", name, name)
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

// download fetches one file to dst through a temporary name, printing
// progress on one line.
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
			if time.Since(last) > 200*time.Millisecond || rerr == io.EOF {
				if size > 0 {
					fmt.Printf("\r  %-48s %5.1f%% of %s", name, 100*float64(done)/float64(size), hostfit.HumanSize(size))
				} else {
					fmt.Printf("\r  %-48s %s", name, hostfit.HumanSize(done))
				}
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
	fmt.Println()
	if err := out.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, dst)
}
