// Command janis runs the local models on this machine the way llama.cpp or
// Ollama does, for any OpenAI-compatible client:
//
//	janis pull [NAME]                download a model (none: the one that fits this machine)
//	janis list                       the installed models
//	janis serve [--port N] [--model NAME] [--host ADDR] [--detach] [--edit]
//	janis start [--port N] [--model NAME] [--host ADDR] [--log FILE] [--edit]   serve --detach
//	janis stop [--port N]            stop it gracefully (SIGKILL after 10 s)
//	janis restart [--port N] [flags] stop, then start on the same flags, or the ones given; none running: start
//	janis status [--port N]          pid, version, model, uptime, and whether /health answers
//	janis eval --problems F --out S  a problems file asked of a server with fixed seeds, scored
//	janis eval compare BASE EDITED   two scores paired problem by problem: PASS or FAIL
//	janis weights shift|revert|save|load   the weight edits of a server started with --edit
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
	case "eval":
		if err := evalCmd(os.Args[2:], os.Stdout); err != nil {
			fail(err)
		}
	case "weights":
		if err := weightsCmd(os.Args[2:], os.Stdout); err != nil {
			fail(err)
		}
	case "upgrade":
		if err := upgrade(os.Args[2:]); err != nil {
			fail(err)
		}
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
       janis serve   [--port 8090] [--host 127.0.0.1] [--model NAME] [--detach] [--log FILE] [--edit]
       janis start   [--port 8090] [--host 127.0.0.1] [--model NAME] [--log FILE] [--edit]
       janis stop    [--port 8090]
       janis restart [--port 8090] [--host ADDR] [--model NAME] [--log FILE] [--edit]
       janis status  [--port 8090]
       janis eval    --problems FILE.jsonl --out SCORE.json [--url URL] [--model NAME] [--seed 1000] [--max-tokens 16384] [--parallel 1] [--effort high]
       janis eval compare [--alpha 0.05] BASE.json EDITED.json
       janis weights shift --match REGEXP --by DELTA [--url URL]
       janis weights revert|save|load [--url URL] [PATH]
       janis upgrade [--check] [--force] [VERSION]
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
		detachServe(*f.host, *f.port, f.modelName(), *f.log, *f.edit)
		return
	}
	serveForeground(f)
}
