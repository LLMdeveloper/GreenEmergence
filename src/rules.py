# -*- coding: utf-8 -*-
"""规则基线。"""
import numpy as np


def rule_action(rule, env, aid):
    a = env.agents[aid]
    if rule == 'always_green':
        if a.income >= env.eff_green:
            return 2
        if a.income >= env.eff_normal:
            return 1
        return 0
    if rule == 'always_regular':
        return 1 if a.income >= env.eff_normal else 0
    if rule == 'random':
        return int(np.random.randint(3))
    if rule == 'threshold':
        r_local = env._local_green_ratio(aid)
        if r_local > a.threshold and a.income >= env.eff_green:
            return 2
        return 1 if a.income >= env.eff_normal else 0
    return 0