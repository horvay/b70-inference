"""Like tf_cmp.py but aligns steps by (text, step) so runs with different TF_N / TF_TEXTS compare correctly."""
import sys, torch
ref = {(r["text"], r["i"]): r for r in torch.load(sys.argv[1])}
for p in sys.argv[2:]:
    run = [r for r in torch.load(p) if (r["text"], r["i"]) in ref]
    agree = kl = nll = rnll = 0.0
    for b in run:
        a = ref[(b["text"], b["i"])]
        agree += int(a["top_i"][0] == b["top_i"][0])
        pa = a["top_v"].exp(); pa = pa / pa.sum()
        bm = {int(i): float(v) for i, v in zip(b["top_i"], b["top_v"])}
        lb = torch.tensor([bm.get(int(i), float(b["top_v"][-1])) for i in a["top_i"]])
        lb = lb - torch.logsumexp(lb, 0)
        kl += float((pa * (pa.log() - lb)).sum())
        nll += -b["lp_tok"]; rnll += -a["lp_tok"]
    n = len(run)
    print(f"{p}: {n} aligned steps, top-1 agreement {agree / n * 100:.1f}%, KL {kl / n:.4f}, NLL {nll / n:.3f} vs ref {rnll / n:.3f}")
