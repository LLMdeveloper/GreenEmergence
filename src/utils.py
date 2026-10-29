# -*- coding: utf-8 -*-
"""通用工具：随机种子、日志、权重覆盖。"""
import os
import time
import random

import numpy as np
import torch

NEED_KEYS = ['physiological', 'safety', 'social', 'esteem', 'self_actualization']


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self.f = open(path, 'a', encoding='utf-8')
        self.t0 = time.time()

    def log(self, msg, console=False):
        line = f"[{time.strftime('%H:%M:%S')} +{(time.time()-self.t0)/60:5.1f}m] {msg}"
        self.f.write(line + '\n')
        self.f.flush()
        if console:
            print(line, flush=True)

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


def apply_weight_override(nw, override):
    if override is None:
        return dict(nw)
    dim, val = override
    w = dict(nw)
    others = [k for k in NEED_KEYS if k != dim]
    s = sum(w[k] for k in others)
    if s <= 1e-12:
        for k in others:
            w[k] = (1.0 - val) / len(others)
    else:
        for k in others:
            w[k] = w[k] / s * (1.0 - val)
    w[dim] = float(val)
    return w