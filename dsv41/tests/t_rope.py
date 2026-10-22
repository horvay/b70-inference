import torch
from dsv_xpu import ops
dev = torch.device("xpu")
torch.manual_seed(0)
inv = 1.0 / (10000 ** (torch.arange(0, 64, 2).float() / 64))
inv = inv.to(dev)
def both(fn_args):
    outs = []
    for f in (ops.rope, ops.rope_torch):
        a = [x.clone() if isinstance(x, torch.Tensor) else x for x in fn_args]
        f(*a)
        outs.append(a)
    return outs
# q (1, 5, 64, 512) + kv (1, 5, 1, 512) with k_norm, rotate offset 448, position 37
q = torch.randn(1, 5, 64, 512, device=dev).half(); kv = torch.randn(1, 5, 1, 512, device=dev).half()
w = (torch.rand(512, device=dev) + 0.5).bfloat16()
A, B = both([q, q, kv, kv, inv, 37, None, None, 1, 1.0, None, w, 1e-6, 0.0, 0.0, 0, 1, 448])
# out tensors are q, kv in place: positions 0 and 2 (same object as 1 and 3 in each clone list? no: cloned separately)
for i, name in ((1, "q"), (3, "kv")):
    print(name, (A[i].float() - B[i].float()).abs().max().item(), "changed", (A[i].float() - q.float() if name == "q" else A[i].float() - kv.float()).abs().max().item())
# trailing-slice view with position_ids
k = torch.randn(1, 7, 1, 128, device=dev).half()
gpos = (torch.arange(7, device=dev) * 2).int().unsqueeze(0)
k1, k2 = k.clone(), k.clone()
ops.rope(k1[..., -64:], k1[..., -64:], None, None, inv, 0, None, gpos, 1, 1.0, None, None, 1e-6, 0.0, 0.0, 0, 1, 0)
ops.rope_torch(k2[..., -64:], k2[..., -64:], None, None, inv, 0, None, gpos, 1, 1.0, None, None, 1e-6, 0.0, 0.0, 0, 1, 0)
print("slice view", (k1.float() - k2.float()).abs().max().item())
