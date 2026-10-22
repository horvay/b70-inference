"""The chat server's prompt side against DeepSeek's reference (no GPU): the vendored encoder reproduces encoding/tests'
golden prompts; xpu/serve.py renders OpenAI-shaped requests (tools on the request, tool results as tool messages, the
reasoning effort, NQ's prefill) to the same prompts; streamed DSML tool calls come back as OpenAI tool_calls.
  scripts/in_rootfs.sh python3 tests/t_encoding.py"""
import copy
import json
import os
import sys

sys.path.insert(0, "/w/dsv41/xpu")
import dsv_encoding as E  # noqa: E402
import serve  # noqa: E402

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "encoding_cases")
fails = 0


def check(name, ok, detail=""):
    global fails
    fails += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name}{'' if ok else ': ' + detail}", flush=True)


# 1. the vendored encoder, as upstream's test_examples_encoding_golden_outputs runs it
for i in range(1, 6):
    case = E.load_cases(f"{D}/test_input_{i}.json")[0]
    prompt, _ = E.encode_case(case, thinking_mode="chat")
    check(f"golden prompt {i}", prompt == open(f"{D}/test_output_{i}.txt").read())

# 2. the server's renderer over the request shape a client sends (tools beside the messages, not on them)
R = serve.Renderer()
for i in range(1, 5):
    raw = json.load(open(f"{D}/test_input_{i}.json"))
    case = raw if isinstance(raw, dict) else {"messages": raw}
    mode = case.get("thinking_mode") or "chat"
    kw = {"enable_thinking": mode == "thinking"}
    if case.get("reasoning_effort") is not None:
        kw["reasoning_effort"] = case["reasoning_effort"]
    got = R.render(copy.deepcopy(case["messages"]), tools=case.get("tools"), kwargs=kw)
    check(f"server renders case {i}", got == open(f"{D}/test_output_{i}.txt").read())

msgs = [{"role": "system", "content": "You are a storyteller."}, {"role": "user", "content": "Hello."}]
for effort, budget in (("minimal", 50), ("low", 50), ("medium", 75), ("high", 75), ("xhigh", 100), ("max", 100),
                       (30, 30), ("85", 85), (None, 75)):
    p = R.render(msgs, kwargs={"enable_thinking": True, "reasoning_effort": effort})
    check(f"effort {effort!r} -> {budget}", f"Reasoning Effort: {budget} " in p, p[:120])
p = R.render(msgs, kwargs={"enable_thinking": False, "reasoning_effort": "max"})
check("thinking off: no effort, closed thought", "Reasoning Effort" not in p and p.endswith("<｜Assistant｜></think>"))
p = R.render(msgs, kwargs={"enable_thinking": True, "nq_prefill": "The scene opens"})
check("NQ prefill follows the open thought", p.endswith("<｜Assistant｜><think>The scene opens"))
p = R.render(msgs, tools=[{"type": "function", "function": {"name": "roll", "parameters": {"type": "object"}}}],
             kwargs={"enable_thinking": True})
check("tools join the leading system message", p.startswith("<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75")
      and "You are a storyteller.\n\n## Tools" in p and '{"name": "roll"' in p)

# 3. a streamed completion with a tool call (upstream's V41_TOOL_CALL_OUTPUT, the EOS being a stop token here)
out = ('  reason  </think>summary\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="lookup">\n'
       '<｜DSML｜ parameter name="query" string="true">value</｜DSML｜ parameter>\n'
       '<｜DSML｜ parameter name="limit" string="false">2</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>')
tools = [{"type": "function", "function": {"name": "lookup"}}]
for step in (1, 3, 7, len(out)):
    sp = serve.Splitter(True, tools)
    parts = []
    for k in range(0, len(out), step):
        parts += sp.feed(out[k:k + step])
    parts += sp.feed("", final=True)
    reasoning = "".join(t for kind, t in parts if kind == "reasoning")
    content = "".join(t for kind, t in parts if kind == "content")
    rest, calls = serve.parse_tool_calls(sp.held, tools)
    ok = (reasoning == "  reason  " and content.strip() == "summary" and not rest and len(calls) == 1
          and calls[0]["function"]["name"] == "lookup"
          and json.loads(calls[0]["function"]["arguments"]) == {"query": "value", "limit": 2})
    check(f"streamed tool call, {step}-char chunks", ok, repr((reasoning, content, rest, calls)))
rest, calls = serve.parse_tool_calls('<｜DSML｜ calls>\n<｜DSML｜ invoke nam', tools)
check("a broken tool call stays visible", rest.startswith("<｜DSML｜ calls>") and not calls)

print("ALL OK" if not fails else f"{fails} FAILED", flush=True)
sys.exit(1 if fails else 0)
