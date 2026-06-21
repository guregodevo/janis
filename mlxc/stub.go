//go:build !(darwin && arm64)

// Package mlxc is Apple-Silicon-only (see mlxc.go). This stub keeps the package
// valid on other platforms, where nothing imports it — the pure-Go llm/cpu
// backend is used instead.
package mlxc
