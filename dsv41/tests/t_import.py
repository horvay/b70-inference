import os
os.environ.setdefault("EXL3_XPU", "1")
from glm_xpu import compat
compat.install()
import torch
from exllamav3 import Config, Model
cfg = Config.from_directory("/models", load_method="python")
print(type(cfg).__name__, cfg.num_hidden_layers, cfg.layer_types[:3])
m = Model.from_config(cfg)
print(len(m.modules), [type(x).__name__ for x in m.modules[:6]])
