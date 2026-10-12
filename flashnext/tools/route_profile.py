#!/usr/bin/env python3
"""Drive the server with prompt sets to profile expert routing (EXL3_MOE_PROFILE) and time prefill and decode.

    python3 route_profile.py <phase> <system prompt file or -> <out.json> [max_tokens]

Phases A and B are game-master turns: the system prompt is a text-adventure game-master prompt (the file given) and
the user turns are synthetic player actions (written here, not taken from any campaign). Phases G and H are general use
(coding, math, explanations, chat, writing, summaries, translation, reasoning) with no system prompt; the ones marked
think run with thinking on. A and G build hot lists; B and H are held out to evaluate them. Streams each reply, so
prefill (time to first token) and decode are timed apart. Ends with a tiny request that makes the server write its
routing counts (they are flushed at a prefill).
"""
import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:30250/v1/chat/completions"

A = [
    "I push open the door of the Gilded Stag inn, shake the rain off my cloak, and ask the innkeeper for a room and a hot meal.",
    "I draw my sword and step between the bandit and the merchant's daughter. 'Drop the knife. Nobody needs to bleed today.'",
    "I kneel by the stream and study the tracks in the mud. How many riders came through here, and how long ago?",
    "I haggle with the blacksmith over the price of a steel shield, pointing out the dent near its rim.",
    "I light a torch and climb down the crumbling stairs into the crypt beneath the old chapel.",
    "I ask the old ferryman what he knows about the lights people keep seeing on the far shore of the lake at night.",
    "I cast a ward of protection around the campfire before we sleep, then take the first watch.",
    "I try to pick the lock on the noble's study while the guards change shifts in the courtyard below.",
    "I sit with the wounded scout, bandage her leg, and ask her what she saw at the border fort.",
    "I climb the bell tower to get a view of the whole city and look for the smoke the courier mentioned.",
]
B = [
    "I offer the goblin chieftain a sack of salt in exchange for safe passage through the hills.",
    "I search the shipwreck on the beach for anything useful, starting with the captain's cabin.",
    "I challenge the arrogant knight to a duel at dawn in front of the whole market square.",
    "I follow the strange child through the fog-filled alleys, keeping my distance so she doesn't notice me.",
    "I read the faded letter we found in the desk aloud to my companions and ask what they make of it.",
    "I bargain with the witch in the swamp for a cure for the sleeping sickness spreading through the village.",
]
# (prompt, thinking on)
G = [
    ("Write a Python function that merges overlapping intervals in a list of (start, end) tuples, with a docstring and two tests.", False),
    ("Explain how TCP congestion control works, covering slow start, congestion avoidance and fast recovery.", False),
    ("A train leaves at 9:40 and travels 312 km at an average of 78 km/h. With a 12-minute stop, when does it arrive?", True),
    ("What are the main differences between PostgreSQL and SQLite, and when would you choose each?", False),
    ("Write a short story opening about a lighthouse keeper who finds a message in a bottle addressed to her.", False),
    ("Summarize the causes and consequences of the 2008 financial crisis in a few paragraphs.", False),
    ("Translate into French and German: 'The meeting has been moved to Thursday afternoon because the client is travelling.'", False),
    ("Find the bug: def avg(xs): return sum(xs) / len(xs) - it crashes for some inputs. Fix it and explain.", True),
    ("I'm feeling burned out at work. What are some practical ways to recover without quitting?", False),
    ("Prove that the square root of 2 is irrational.", True),
    ("Write a bash script that finds the ten largest files under a directory and prints their sizes in human-readable form.", False),
    ("Compare the French and American revolutions: causes, course, and outcomes.", False),
]
H = [
    ("Write a TypeScript function that debounces another function, with types, and explain how it works.", False),
    ("How do vaccines train the immune system? Explain for a curious teenager.", False),
    ("If 3 painters paint a house in 8 days, how long do 4 painters take, and what assumptions does that rely on?", True),
    ("Draft a polite email to a landlord asking for a broken heater to be fixed this week.", False),
    ("Explain what a hash map is and how collisions are handled, with a small example in Go.", False),
    ("Give me a weekly vegetarian meal plan for two people on a budget.", False),
    ("Is it ever rational to buy a lottery ticket? Reason it through.", True),
    ("Write a haiku sequence about autumn in a city.", False),
]


def post(body: dict):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=300)


def turn(system: str | None, user: str, max_tokens: int, think: bool = False) -> dict:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    body = {"model": "flashnext-uncensored", "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": think},
            "messages": msgs}
    t0 = time.perf_counter()
    ttft, usage, n_chunks = None, {}, 0
    with post(body) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            delta = d["choices"][0].get("delta", {}) if d.get("choices") else {}
            if delta.get("content") or delta.get("reasoning_content"):
                n_chunks += 1
                if ttft is None:
                    ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0
    out = usage.get("completion_tokens", n_chunks)
    dec = (out - 1) / (total - ttft) if ttft and total > ttft and out > 1 else 0.0
    return {"prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": out, "ttft_s": round(ttft or 0, 2),
            "total_s": round(total, 2), "decode_tok_s": round(dec, 1),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens")}


def main():
    phase, sys_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
    max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 250
    system = None if sys_path == "-" else open(sys_path).read()
    prompts = {"A": [(u, False) for u in A], "B": [(u, False) for u in B], "G": G, "H": H}[phase]
    rows = []
    for i, (user, think) in enumerate(prompts):
        r = turn(system, user, max_tokens, think)
        rows.append(r)
        print(f"PROFILE {phase}{i}: {r}", flush=True)
    time.sleep(2.5)       # the server writes counts at most every 2 s; this tiny prefill flushes the last decode counts
    turn("You are terse.", "Say OK.", 2)
    dec = [r["decode_tok_s"] for r in rows if r["decode_tok_s"]]
    summary = {"phase": phase, "requests": rows, "mean_decode_tok_s": round(sum(dec) / len(dec), 2) if dec else None}
    json.dump(summary, open(out, "w"), indent=1)
    print(f"PROFILE {phase} mean decode {summary['mean_decode_tok_s']} tok/s over {len(dec)} requests", flush=True)


if __name__ == "__main__":
    main()
