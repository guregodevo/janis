// Command janis runs the local models on this machine the way llama.cpp or
// Ollama does, for any OpenAI-compatible client:
//
//	janis pull [NAME]                download a model (none: the one that fits this machine)
//	janis list                       the installed models
//	janis serve [--port N] [--model NAME] [--host ADDR] [--detach]
//	janis start [--port N] [--model NAME] [--host ADDR] [--log FILE]   serve --detach
//	janis stop [--port N]            stop it gracefully (SIGKILL after 10 s)
//	janis restart [--port N] [flags] stop, then start on the same flags, or the ones given; none running: start
//	janis status [--port N]          pid, version, model, uptime, and whether /health answers
//	janis version                    the release this binary is
//
// serve answers GET /v1/models and POST /v1/chat/completions (streamed or
// not, with tools) on 127.0.0.1:8090 by default. A client on another
// machine reaches it through the host the gateway runs on (Memdoor:
// `memdoor --gateway ssh://you@box connect local --base http://localhost:8090/v1`).
package main

import (
	"fmt"
	"os"

	"github.com/guregodevo/janis/hostfit"
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
		if err := pull(os.Args[2:]); err != nil {
			fail(err)
		}
	case "serve":
		run(os.Args[2:])
	case "start":
		startCmd(os.Args[2:])
	case "stop":
		stopCmd(os.Args[2:])
	case "restart":
		restartCmd(os.Args[2:])
	case "status":
		statusCmd(os.Args[2:])
	case "version", "--version", "-v":
		fmt.Println("janis " + version)
	default:
		usage()
		os.Exit(2)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, `usage: janis pull [NAME]
       janis list
       janis serve   [--port 8090] [--host 127.0.0.1] [--model NAME] [--detach] [--log FILE]
       janis start   [--port 8090] [--host 127.0.0.1] [--model NAME] [--log FILE]
       janis stop    [--port 8090]
       janis restart [--port 8090] [--host ADDR] [--model NAME] [--log FILE]
       janis status  [--port 8090]
       janis version`)
}

func list() {
	models, err := hostfit.ListInstalled()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if len(models) == 0 {
		fmt.Println("no model installed: janis serves the mlx-community models in " + hostfit.CacheDir())
		return
	}
	host, _ := hostfit.Detect()
	for _, m := range models {
		fmt.Printf("%s %-40s %s\n", host.Rate(m.Name).Flag(), m.Name, hostfit.HumanSize(m.SizeBytes))
	}
}

func run(args []string) {
	f := newServeFlags("serve")
	detach := f.fs.Bool("detach", false, "pull the model if missing, serve in the background, and return once it has answered (janis start)")
	_ = f.fs.Parse(args)
	if *detach {
		detachServe(*f.host, *f.port, f.modelName(), *f.log)
		return
	}
	serveForeground(f)
}
