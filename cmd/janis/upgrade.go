package main

import (
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strings"
	"time"
)

// releases is where janis's releases are. A var: a test serves its own.
var releases = "https://github.com/guregodevo/janis/releases"

// latestRelease is the tag of the newest release: where releases/latest
// redirects. No API call, so no rate limit and no token.
func latestRelease(client *http.Client) (string, error) {
	c := *client
	c.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	resp, err := c.Head(releases + "/latest")
	if err != nil {
		return "", err
	}
	resp.Body.Close()
	loc := resp.Header.Get("Location")
	if resp.StatusCode/100 != 3 || !strings.Contains(loc, "/tag/") {
		return "", fmt.Errorf("%s/latest answered %s, not a redirect to a release", releases, resp.Status)
	}
	return path.Base(loc), nil
}

// upgrade replaces this binary with a release: the newest, or the one
// named. It runs that release's install.sh, the script a first install
// runs, on the directory this binary is in: the download is checked against
// the release's sha256, and on a Mac the MLX libraries beside it follow.
func upgrade(args []string) error {
	fs := flag.NewFlagSet("upgrade", flag.ExitOnError)
	check := fs.Bool("check", false, "say whether a newer release exists, and change nothing")
	force := fs.Bool("force", false, "install even when this binary is that release already")
	_ = fs.Parse(args)
	want := fs.Arg(0)

	client := &http.Client{Timeout: 30 * time.Second}
	if want == "" {
		var err error
		if want, err = latestRelease(client); err != nil {
			return fmt.Errorf("the latest release: %w", err)
		}
	}
	if want == version && !*force {
		fmt.Printf("janis %s is the release asked for: nothing to do (--force installs it again)\n", version)
		return nil
	}
	if *check {
		fmt.Printf("janis %s is installed, %s is available: janis upgrade\n", version, want)
		return nil
	}

	exe, err := os.Executable()
	if err != nil {
		return err
	}
	if exe, err = filepath.EvalSymlinks(exe); err != nil {
		return err
	}
	resp, err := client.Get(releases + "/download/" + want + "/install.sh")
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("release %s has no install.sh (%s)", want, resp.Status)
	}
	script, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	fmt.Printf("janis %s → %s in %s\n", version, want, filepath.Dir(exe))
	cmd := exec.Command("sh", "-s")
	cmd.Stdin = strings.NewReader(string(script))
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	cmd.Env = append(os.Environ(), "JANIS_VERSION="+want, "JANIS_INSTALL_DIR="+filepath.Dir(exe))
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("the install of %s failed: %w", want, err)
	}
	fmt.Println("  a janis already serving runs the old release until: janis restart")
	return nil
}
