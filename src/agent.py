# -*- coding: utf-8 -*-
"""Agent 定义。"""
import numpy as np
from collections import deque

from .utils import NEED_KEYS


class Agent:
    def __init__(self, idx, nw, needs, decay, income, theta, inc_mean, inc_std, recent_len):
        self.id = idx
        self.base_need_weights = nw.copy()
        self.need_weights = nw.copy()
        self.base_needs = needs.copy()
        self.decay_rates = decay.copy()
        self.needs = needs.copy()
        self.income = income
        self.threshold = theta
        self.income_mean = inc_mean
        self.income_std = inc_std
        self.neighbors = []
        self.recent_actions = deque(maxlen=recent_len)
        self.last_green = False
        self.is_seed = False

    def reset(self, rng):
        self.needs = {k: float(np.clip(v + rng.uniform(-0.3, 0.3), 1.0, 9.0))
                      for k, v in self.base_needs.items()}
        self.income = max(10.0, float(rng.normal(self.income_mean, self.income_std)))
        self.recent_actions.clear()
        self.last_green = False
        self.is_seed = False

    def update_decay(self):
        for k, d in self.decay_rates.items():
            self.needs[k] = max(0.0, self.needs[k] - d)