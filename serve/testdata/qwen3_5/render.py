#!/usr/bin/env python3
"""render.py <chat_template.jinja> <request.json>: the prompt the model's own chat template writes for an
OpenAI chat request, thinking off. How the .txt files beside it were made, from Qwen/Qwen3.8-27B's
chat_template.jinja (revision 37561816e66b): python3 render.py chat_template.jinja tools.json > tools.txt
Needs jinja2. The template's generation markers (transformers' own extension) are dropped."""
import json, re, sys, jinja2
src = re.sub(r"\{%-?\s*(end)?generation\s*-?%\}", "", open(sys.argv[1]).read())
env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])
def raise_exception(m): raise Exception(m)
env.globals["raise_exception"] = raise_exception
env.filters["tojson"] = lambda v, **k: json.dumps(v, ensure_ascii=False)
req = json.load(open(sys.argv[2]))
for m in req["messages"]:
    for tc in m.get("tool_calls") or []:
        tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
sys.stdout.write(env.from_string(src).render(messages=req["messages"], tools=req.get("tools"), add_generation_prompt=True, enable_thinking=False))
