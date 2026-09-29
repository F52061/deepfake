# -*- coding: utf-8 -*-
"""
CPU-受限包装入口: 在任何重计算库 import 之前设线程数, 将 CPU 占用压到最低。
GPU 不受限。用于 eval_all_datasets.py 等长任务的启动。

用法(项目根目录):
  python vit_module/run_eval_cpu_limited.py -- <eval_all_datasets.py 的参数...>
例:
  python vit_module/run_eval_cpu_limited.py --model-type bridge \
      --checkpoint ./checkpoints/stage_1/bridge_v2_phase1.pth --device cuda:0
"""
import os, sys

# ── 1) import torch/cv2/sklearn 之前设置线程环境变量(Win 只在进程启动读) ──
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.abspath(os.path.join(_HERE, ".."))
for p in (_HERE, _PROJECT):
    if p not in sys.path:
        sys.path.insert(0, p)

# ── 2) 限制 torch / cv2 / sklearn 线程 ──
import torch
torch.set_num_threads(1)
try:
    import cv2
    cv2.setNumThreads(0)
except Exception:
    pass
import numpy as np  # noqa

# ── 3) 转发参数给目标脚本的 main() ──
if "--" in sys.argv:
    idx = sys.argv.index("--")
    argv = sys.argv[idx + 1:]
else:
    argv = sys.argv[1:]

import importlib.util
TARGET = os.path.join(_HERE, "eval_all_datasets.py")
spec = importlib.util.spec_from_file_location("eval_all_datasets", TARGET)
mod = importlib.util.module_from_spec(spec)
sys.argv = ["eval_all_datasets.py"] + argv
spec.loader.exec_module(mod)
if hasattr(mod, "main"):
    mod.main()
else:
    print("target has no main()")
