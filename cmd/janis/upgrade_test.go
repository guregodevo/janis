package main

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// releaseServer stands in for GitHub: releases/latest redirects to a tag,
// and each release's install.sh writes the version it was asked for and the
// directory it was given into a file there.
func releaseServer(t *testing.T, latest string, asked *[]string) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		*asked = append(*asked, r.Method+" "+r.URL.Path)
		switch {
		case r.URL.Path == "/latest":
			http.Redirect(w, r, "/tag/"+latest, http.StatusFound)
		case strings.HasSuffix(r.URL.Path, "/install.sh") && !strings.Contains(r.URL.Path, "v0.0.0"):
			w.Write([]byte(`echo "$JANIS_VERSION" > "$JANIS_INSTALL_DIR/installed"` + "\n"))
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	old := releases
	releases = srv.URL
	t.Cleanup(func() { releases = old })
}

// installedBy is what the stand-in install.sh wrote beside the test binary.
func installedBy(t *testing.T) string {
	t.Helper()
	exe, _ := os.Executable()
	exe, _ = filepath.EvalSymlinks(exe)
	marker := filepath.Join(filepath.Dir(exe), "installed")
	t.Cleanup(func() { os.Remove(marker) })
	raw, err := os.ReadFile(marker)
	if err != nil {
		return ""
	}
	os.Remove(marker)
	return strings.TrimSpace(string(raw))
}

// UPGRADE INSTALLS THE LATEST RELEASE WHERE THIS BINARY IS.
//
// The latest tag is read off the redirect of releases/latest; that
// release's install.sh runs with the tag and this binary's directory. A
// binary that is the release already is left alone, --check changes
// nothing, and a version can be named.
func TestUpgradeInstallsTheLatestReleaseWhereThisBinaryIs(t *testing.T) {
	var asked []string
	releaseServer(t, "v9.9.9", &asked)
	old := version
	t.Cleanup(func() { version = old })

	version = "v9.9.8"
	if err := upgrade(nil); err != nil {
		t.Fatal(err)
	}
	if got := installedBy(t); got != "v9.9.9" {
		t.Errorf("install.sh ran for %q in this binary's directory, want v9.9.9", got)
	}
	if want := "GET /download/v9.9.9/install.sh"; len(asked) != 2 || asked[1] != want {
		t.Errorf("asked %v, want HEAD /latest then %s", asked, want)
	}

	if err := upgrade([]string{"--check"}); err != nil {
		t.Fatal(err)
	}
	if got := installedBy(t); got != "" {
		t.Errorf("--check installed %q", got)
	}

	version = "v9.9.9"
	if err := upgrade(nil); err != nil {
		t.Fatal(err)
	}
	if got := installedBy(t); got != "" {
		t.Errorf("a binary that is the latest release was replaced by %q", got)
	}
	if err := upgrade([]string{"--force"}); err != nil {
		t.Fatal(err)
	}
	if got := installedBy(t); got != "v9.9.9" {
		t.Errorf("--force installed %q, want v9.9.9", got)
	}

	if err := upgrade([]string{"v9.9.1"}); err != nil {
		t.Fatal(err)
	}
	if got := installedBy(t); got != "v9.9.1" {
		t.Errorf("a named version installed %q, want v9.9.1", got)
	}
	if err := upgrade([]string{"v0.0.0"}); err == nil || !strings.Contains(err.Error(), "v0.0.0") {
		t.Errorf("a release that does not exist: error %v", err)
	}
}
