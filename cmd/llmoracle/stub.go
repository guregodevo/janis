//go:build !(darwin && arm64 && cgo)

// The oracle drives the MLX engine directly, so it exists only where MLX
// does. Elsewhere it is a command that says why.
package main

import "fmt"

func main() { fmt.Println("llmoracle needs Apple Silicon: it drives the MLX engine directly") }
