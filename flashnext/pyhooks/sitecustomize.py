# Imported by every Python process of the server (serve_b70.sh puts this folder first on PYTHONPATH).
#
# Hand freed heap memory back to the OS. glibc keeps memory that Python, torch and the tokenizers free, and loading the
# model frees a lot: on 2026-10-06 the scheduler held a 2.4 GB heap with 0.2 GB in use, the tokenizer manager and
# detokenizer ~1.5 GB more. The kernel then swaps those idle pages into zram, where they still cost RAM. malloc_trim(0)
# releases the free pages of every arena (swapped-out ones are dropped without being read back in).
# TRIM_EVERY_S (default 300; 0 = off): first trims at 60 and 180 s (after the model load), then every TRIM_EVERY_S.
import ctypes
import importlib.util
import os
import sys
import threading
import time


def _trimmer(every):
    trim = ctypes.CDLL("libc.so.6").malloc_trim
    for wait in (60, 120):
        time.sleep(wait)
        trim(0)
    while True:
        time.sleep(every)
        trim(0)


_every = float(os.environ.get("TRIM_EVERY_S", "300"))
if _every > 0 and sys.platform.startswith("linux"):
    threading.Thread(target=_trimmer, args=(_every,), name="malloc-trim", daemon=True).start()

# chain to the sitecustomize this one shadows, if any
_here = os.path.dirname(os.path.abspath(__file__))
for _p in sys.path:
    if _p and os.path.abspath(_p) != _here and os.path.isfile(os.path.join(_p, "sitecustomize.py")):
        _spec = importlib.util.spec_from_file_location("_sitecustomize_next", os.path.join(_p, "sitecustomize.py"))
        try:
            if _spec is not None and _spec.loader is not None:
                _spec.loader.exec_module(importlib.util.module_from_spec(_spec))
        except Exception:
            pass
        break
