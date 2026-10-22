"""CPU-side profile of decode steps (cProfile) + decode speed without the prompt.
  NTOK=24 python3 tests/cprof.py"""
import os, time, cProfile, pstats, io
os.environ.setdefault("DSV_SAVE_PROFILE", "0")     # tests do not move the routing profile
import torch
from dsv_xpu.runner import load

model, config, cache, tok, tier = load("/models", "/nvx", cache_tokens=8192, max_batch_size=1,
                                       ram_expert_gb=float(os.environ.get("RAM_GB", "16")))
tier.warm(tier.load_profile())
from exllamav3 import Generator, Job
gen = Generator(model=model, cache=cache, tokenizer=tok)
gen.generate(prompt="Once upon a time, in a small village by the sea,", max_new_tokens=24, add_bos=True)
N = int(os.environ.get("NTOK", "24"))


def decode_run(prompt, n, prof=None):
    j = Job(input_ids=tok.encode(prompt, add_bos=True), max_new_tokens=n, min_new_tokens=n, stop_conditions=[])
    gen.enqueue(j)
    t_first = None
    cnt = 0
    while gen.num_remaining_jobs():
        if t_first is not None and prof is not None:
            prof.enable()
        rs = gen.iterate()
        if prof is not None:
            prof.disable()
        for r in rs:
            if r.get("stage") == "streaming" and r.get("token_ids") is not None:
                if t_first is None:
                    t_first = time.time()
                else:
                    cnt += r["token_ids"].numel()
    return cnt / (time.time() - t_first)


print(f"decode {decode_run('The history of the Roman Empire', N):.2f} tok/s", flush=True)
print(f"decode {decode_run('A short note about rivers and lakes', N):.2f} tok/s", flush=True)
pr = cProfile.Profile()
sp = decode_run("The old lighthouse keeper looked at the horizon and", N, pr)
print(f"decode under cProfile {sp:.2f} tok/s", flush=True)
for key in ("tottime", "cumtime"):
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats(key).print_stats(45)
    print(s.getvalue(), flush=True)
print("tier", tier.stats)
