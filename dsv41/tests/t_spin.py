"""Which GPU polling method sees a host write to pinned memory? Per method and host delay: trials, failures, polls."""
import sys, time, ctypes, threading
import torch
sys.path.insert(0, "/w/dsv41/xpu")
from dsv_xpu import moe
G, X = moe.G(), moe.X()
dev = torch.device("xpu", 0)
S = 64
lib = ctypes.CDLL("/w/dsv41/xpu/lib/dsv_spin.so")
lib.wait_ge.restype = ctypes.c_int64
lib.wait_ge.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
p = torch.zeros(6, dtype=torch.int64, device=dev)
MAXIT = 20000
for method in ("atomic", 1, 2, 3):
    for delay in (0.0, 5e-5, 2e-3):
        flags = torch.zeros(8 * S, dtype=torch.int64).pin_memory(); f = flags.numpy()
        hp = torch.zeros(16, dtype=torch.int64).pin_memory()
        waits = torch.zeros(2 * S, dtype=torch.int64).pin_memory(); w = waits.numpy()
        fails, polls = 0, []
        t0 = time.time()
        for step in range(1, 11):
            def host():
                lib.wait_ge(flags.data_ptr(), step, int(5e9), 0)
                if delay:
                    t = time.perf_counter()
                    while time.perf_counter() - t < delay:
                        pass
                f[S] = step
            th = threading.Thread(target=host); th.start()
            if method == "atomic":
                G.sync_point(p, hp, flags, 0, S, 2 * S, waits[0:1], step, MAXIT, 0)
            else:
                G.sync_point(p, hp, flags, 0, S, 2 * S, waits[0:1], step, MAXIT, 2)
                X.spin_test(flags, S, waits[0:2], step, MAXIT, method, p)
            torch.xpu.synchronize()
            th.join()
            if method == "atomic":
                fails += int(f[2 * S] == step); polls.append(int(w[0]))
            else:
                v = int(w[0]); fails += int((v >> 32) & 1); polls.append(v & 0xFFFFFFFF)
        dt = time.time() - t0
        print(f"method {method!s:6} delay {delay * 1e6:6.0f} us: {fails}/10 timed out, polls {polls}, {dt * 100:.1f} ms/trial",
              flush=True)
