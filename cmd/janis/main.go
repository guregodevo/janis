// Command janis runs the local models on this machine the way llama.cpp or
// Ollama does, for any OpenAI-compatible client:
//
//	janis pull [NAME]                download a model (none: the one that fits this machine)
//	janis list                       the installed models
//	janis serve [--port N] [--model NAME] [--host ADDR]
//
// serve answers GET /v1/models and POST /v1/chat/completions (streamed or
// not, with tools) on 127.0.0.1:8090 by default. A client on another
// machine reaches it through the host the gateway runs on (Memdoor:
// `memdoor --gateway ssh://you@box connect local --base http://localhost:8090/v1`).
package main

import (
	"flag"
	"fmt"
	"net"
	"net/http"
	"os"

	llm "github.com/guregodevo/janis"
	"github.com/guregodevo/janis/hostfit"
	"github.com/guregodevo/janis/serve"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	switch os.Args[1] {
	case "list":
		list()
	case "pull":
		pull(os.Args[2:])
	case "serve":
		run(os.Args[2:])
	default:
		usage()
		os.Exit(2)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: janis pull [NAME] | janis list | janis serve [--port 8090] [--host 127.0.0.1] [--model NAME]")
}

func list() {
	models, err := hostfit.ListInstalled()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if len(models) == 0 {
		fmt.Println("no model installed: janis serves the mlx-community models in ~/.cache/huggingface/hub")
		return
	}
	host, _ := hostfit.Detect(llm.HasLocalEngine())
	for _, m := range models {
		fmt.Printf("%s %-40s %s\n", host.Rate(m.Name).Flag(), m.Name, hostfit.HumanSize(m.SizeBytes))
	}
}

func run(args []string) {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	port := fs.Int("port", 8090, "the port to listen on")
	host := fs.String("host", "127.0.0.1", "the address to bind (0.0.0.0 for the network)")
	model := fs.String("model", "", "the installed model to load now (default: the first a request names)")
	_ = fs.Parse(args)
	srv, err := serve.New(*model)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer srv.Close()
	addr := net.JoinHostPort(*host, fmt.Sprint(*port))
	fmt.Printf("janis serving on http://%s/v1 — GET /v1/models, POST /v1/chat/completions\n", addr)
	if err := http.ListenAndServe(addr, srv.Handler()); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
