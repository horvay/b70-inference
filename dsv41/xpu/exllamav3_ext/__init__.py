"""XPU stand-in for exllamav3's compiled CUDA extension, DeepSeek-V4.1-Flash build.

The GLM port's ops (glm_xpu.ops: EXL3 linears on exl3xpu ESIMD, norms, sampling, ...) plus the ops DeepSeek-V4.1 adds
(dsv_xpu.ops). Any op not implemented resolves to a stub that raises when called, naming the op.
"""
import os
from glm_xpu import compat as _compat

_compat.install()

from glm_xpu.ops import *          # noqa: F401,F403
from dsv_xpu.ops import *          # noqa: F401,F403

_TRACE = os.environ.get("GLM_XPU_TRACE_MISSING") == "1"
_missing_seen = set()


class _Missing:
    def __init__(self, name):
        self.__name__ = name

    def __call__(self, *a, **k):
        raise NotImplementedError(f"exllamav3_ext.{self.__name__} is not implemented on XPU")

    def __repr__(self):
        return f"<missing exllamav3_ext.{self.__name__}>"

    def __bool__(self):
        return False


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    if _TRACE and name not in _missing_seen:
        _missing_seen.add(name)
        print(f" -- exllamav3_ext: missing op {name}", flush=True)
    return _Missing(name)
