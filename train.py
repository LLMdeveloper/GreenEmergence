# -*- coding: utf-8 -*-
"""GreenEmergence 主入口。

运行模式：
  RUN_MODE='core'         → 5 组 × 5 种子 × 200 轮
  RUN_MODE='extended'     → subsidy/carbontax/single_dim/price_neutral
  RUN_MODE='rule_based'   → 4 条规则基线
  RUN_MODE='all'          → 以上全部
  RUN_MODE='quick_test'   → 1 种子 × 50 轮，用于自检
  RUN_MODE='summary_only' → 只汇总
"""
import os
import torch

from config import build_conditions, GROUP_SEEDS
from src import run_experiment, run_rule_based, summarize


def run_all(groups, seeds_override, episodes, device, out_root):
    C = build_conditions()
    for g in groups:
        for name, cfg in C[g].items():
            if g == 'rule_based':
                print(f"[start] {g}/{name}", flush=True)
                run_rule_based(cfg, name, g, cfg['rule'], n_eps=10, device=device, out_root=out_root)
                continue
            seeds = seeds_override if seeds_override else GROUP_SEEDS.get(g, [42])
            for s in seeds:
                print(f"[start] {g}/{name}/seed{s}", flush=True)
                run_experiment(cfg, name, g, s, episodes, device, out_root)
    summarize(out_root)


RUN_MODE = 'core'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT_ROOT = 'runs_B_v3'
EPISODES = 200
QUICK = 50

if __name__ == '__main__':
    print(f"MODE={RUN_MODE} DEVICE={DEVICE} OUT={OUT_ROOT}")
    print("V3 Fix-4: social_r uses result==2 (actual purchase), not a==2 (intent)")
    print("=" * 60)
    if RUN_MODE == 'quick_test':
        run_all(['core'], [42], QUICK, DEVICE, OUT_ROOT)
    elif RUN_MODE == 'core':
        run_all(['core'], None, EPISODES, DEVICE, OUT_ROOT)
    elif RUN_MODE == 'extended':
        run_all(['subsidy_scan', 'carbontax_scan', 'single_dim', 'price_neutral'],
                None, EPISODES, DEVICE, OUT_ROOT)
    elif RUN_MODE == 'rule_based':
        run_all(['rule_based'], None, EPISODES, DEVICE, OUT_ROOT)
    elif RUN_MODE == 'all':
        run_all(['core', 'subsidy_scan', 'carbontax_scan', 'single_dim',
                 'price_neutral', 'rule_based'], None, EPISODES, DEVICE, OUT_ROOT)
    elif RUN_MODE == 'summary_only':
        summarize(OUT_ROOT)
    else:
        raise ValueError(RUN_MODE)