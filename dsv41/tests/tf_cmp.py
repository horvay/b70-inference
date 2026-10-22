"""Compare teacher-forced runs: top-1 agreement and KL (over the reference's top-64) against a reference run."""
import sys, torch
ref = torch.load(sys.argv[1])
for p in sys.argv[2:]:
    run = torch.load(p)
    n = min(len(ref), len(run))
    agree = 0; kl = 0.0; nll = 0.0
    for a, b in zip(ref[:n], run[:n]):
        agree += int(a["top_i"][0] == b["top_i"][0])
        pa = a["top_v"].exp(); pa = pa / pa.sum()
        # b's log-probs at a's top-64 tokens (missing -> its 64th value)
        bm = {int(i): float(v) for i, v in zip(b["top_i"], b["top_v"])}
        lb = torch.tensor([bm.get(int(i), float(b["top_v"][-1])) for i in a["top_i"]])
        lb = lb - torch.logsumexp(lb, 0)
        kl += float((pa * (pa.log() - lb)).sum())
        nll += -b["lp_tok"]
    print(f"{p}: top-1 agreement {agree / n * 100:.1f}%, KL {kl / n:.4f}, NLL {nll / n:.3f} (ref NLL {-sum(r['lp_tok'] for r in ref[:n]) / n:.3f}), {n} steps")
