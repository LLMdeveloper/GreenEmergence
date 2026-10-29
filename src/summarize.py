# -*- coding: utf-8 -*-
"""汇总所有实验结果。"""
import os
import json

import numpy as np
import pandas as pd
from scipy import stats

from .evaluation import format_ei_worked


def summarize(out_root):
    sd = os.path.join(out_root, 'summary')
    os.makedirs(sd, exist_ok=True)
    all_rows, seed_rows, rule_rows = [], [], []

    for root, _, files in os.walk(out_root):
        if 'records.json' in files:
            with open(os.path.join(root, 'records.json')) as f:
                d = json.load(f)
            for r in d['records']:
                all_rows.append(r)
            if d['records']:
                last = d['records'][-1]
                ts = d['train_shares']
                seed_rows.append({
                    'group': d['group'], 'exp': d['exp'], 'seed': d['seed'],
                    'R_init': np.mean(ts[:min(50, len(ts))]) if len(ts) >= 5 else np.nan,
                    'R_final': np.mean(ts[-min(50, len(ts)):]) if len(ts) >= 5 else np.nan,
                    'adopt_all': last.get('adopt_all', np.nan),
                    'green_share': last.get('green_share', np.nan),
                    'conditional_green_rate': last.get('conditional_green_rate', np.nan),
                    'purchase_rate': last.get('purchase_rate', np.nan),
                    'rejected_rate': last.get('rejected_rate', np.nan),
                    'afford_green_ratio': last.get('afford_green_ratio', np.nan),
                    'persistence': last.get('persistence', np.nan),
                    'EI': last.get('ei_EI', np.nan),
                })
        elif 'result.json' in files:
            with open(os.path.join(root, 'result.json')) as f:
                d = json.load(f)
            rule_rows.append(d)

    pd.DataFrame(all_rows).to_csv(os.path.join(sd, 'all_episodes.csv'), index=False)
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(os.path.join(sd, 'seed_level.csv'), index=False)
    if rule_rows:
        pd.DataFrame(rule_rows).to_csv(os.path.join(sd, 'rule_based.csv'), index=False)

    if seed_df.empty:
        print("no records")
        return

    gs = seed_df.groupby(['group', 'exp']).agg({
        'R_init': ['mean', 'std'], 'R_final': ['mean', 'std'],
        'green_share': ['mean', 'std'],
        'conditional_green_rate': 'mean',
        'purchase_rate': 'mean', 'rejected_rate': 'mean',
        'afford_green_ratio': 'mean',
        'persistence': ['mean', 'std'], 'EI': 'mean',
    }).reset_index()
    gs.columns = ['_'.join(c).strip('_') for c in gs.columns]
    gs.to_csv(os.path.join(sd, 'group_summary.csv'), index=False)

    sub_df = gs[gs['group'] == 'subsidy_scan'][['exp', 'green_share_mean', 'adopt_all_mean',
                                                 'conditional_green_rate_mean', 'rejected_rate_mean']]
    if not sub_df.empty:
        sub_df.to_csv(os.path.join(sd, 'subsidy_scan.csv'), index=False)

    tax_df = gs[gs['group'] == 'carbontax_scan'][['exp', 'green_share_mean', 'adopt_all_mean',
                                                   'afford_green_ratio_mean', 'rejected_rate_mean']]
    if not tax_df.empty:
        tax_df.to_csv(os.path.join(sd, 'carbontax_scan.csv'), index=False)

    def get_val(exp, col):
        row = gs[(gs['group'] == 'core') & (gs['exp'] == exp)]
        return float(row[col].values[0]) if not row.empty else float('nan')

    b_share = get_val('Baseline', 'green_share_mean')
    b_persist = get_val('Baseline', 'persistence_mean')
    no_net = get_val('NoNetwork', 'green_share_mean')
    no_soc = get_val('NoSocial_030', 'green_share_mean')
    b_rej = get_val('Baseline', 'rejected_rate_mean')

    B_ratio_net = (b_share / no_net) if (no_net and not np.isnan(no_net)) else float('nan')
    B_ratio_soc = (b_share / no_soc) if (no_soc and not np.isnan(no_soc)) else float('nan')
    B_cond_net = (B_ratio_net > 3) if not np.isnan(B_ratio_net) else False
    B_cond_soc = (B_ratio_soc > 1.5) if not np.isnan(B_ratio_soc) else False
    B_cond_persist = b_persist > 0
    B_pass = B_cond_net and B_cond_soc and B_cond_persist

    verdicts = []
    for _, r in gs[gs['group'] == 'core'].iterrows():
        exp = r['exp']
        share = float(r['green_share_mean'])
        persist = float(r['persistence_mean'])
        if exp == 'Baseline':
            v = 'B: Mechanism Emergence' if B_pass else 'Neither'
            over = np.nan
        elif exp == 'NoNetwork':
            over = b_share / share if share > 0 else float('inf')
            v = 'Network Necessary' if over > 3 else 'Network Not Necessary'
        elif exp == 'NoSocial_030':
            over = b_share / share if share > 0 else float('inf')
            v = 'Social Necessary' if over > 1.5 else 'Social Not Necessary'
        elif exp == 'NoSeed':
            over = b_share / share if share > 0 else float('inf')
            v = 'Seed Necessary' if over > 1.4 else 'Seed Not Necessary'
        else:
            over = b_share / share if share > 0 else float('inf')
            v = '—'
        verdicts.append({
            'exp': exp,
            'R_init': float(r['R_init_mean']), 'R_final': float(r['R_final_mean']),
            'green_share': share,
            'green_share_sd': float(r['green_share_std']) if not pd.isna(r['green_share_std']) else 0.0,
            'persistence': persist,
            'rejected_rate': float(r['rejected_rate_mean']),
            'baseline_over_this': over,
            'verdict': v,
        })
    verdict_df = pd.DataFrame(verdicts)
    verdict_df.to_csv(os.path.join(sd, 'emergence_verdict.csv'), index=False)

    stats_rows = []
    core_base = seed_df[(seed_df['group'] == 'core') & (seed_df['exp'] == 'Baseline')]
    if not core_base.empty:
        base_green = core_base['green_share'].dropna().values
        base_delta = (core_base['R_final'] - core_base['R_init']).dropna().values
        for exp in ['NoSeed', 'NoSocial_030', 'Full_060', 'NoNetwork']:
            sub = seed_df[(seed_df['group'] == 'core') & (seed_df['exp'] == exp)]
            if len(sub) < 2 or len(base_green) < 2:
                continue
            e_green = sub['green_share'].dropna().values
            e_delta = (sub['R_final'] - sub['R_init']).dropna().values
            try:
                t, p_t = stats.ttest_ind(e_green, base_green, equal_var=False)
                u, p_u = stats.mannwhitneyu(e_green, base_green, alternative='two-sided')
            except Exception:
                t = p_t = u = p_u = np.nan
            did = float(np.mean(e_delta) - np.mean(base_delta)) if len(e_delta) > 0 else np.nan
            rng = np.random.default_rng(0)
            boot = []
            for _ in range(2000):
                a_ = rng.choice(e_delta, len(e_delta), replace=True)
                b_ = rng.choice(base_delta, len(base_delta), replace=True)
                boot.append(np.mean(a_) - np.mean(b_))
            ci = np.percentile(boot, [2.5, 97.5]) if boot else [np.nan, np.nan]
            stats_rows.append({
                'group': 'core', 'exp': exp, 'baseline': 'Baseline',
                'n_seeds_exp': len(e_green), 'n_seeds_base': len(base_green),
                'mean_green_exp': float(np.mean(e_green)),
                'sd_green_exp': float(np.std(e_green, ddof=1)) if len(e_green) > 1 else 0.0,
                'mean_green_base': float(np.mean(base_green)),
                'sd_green_base': float(np.std(base_green, ddof=1)) if len(base_green) > 1 else 0.0,
                'welch_t': t, 'p_welch': p_t, 'mwu_u': u, 'p_mwu': p_u,
                'DID': did, 'DID_ci_low': float(ci[0]), 'DID_ci_high': float(ci[1]),
            })
    if stats_rows:
        pd.DataFrame(stats_rows).to_csv(os.path.join(sd, 'stats.csv'), index=False)

    worked_path = os.path.join(sd, 'ei_worked_example.txt')
    if os.path.exists(worked_path):
        os.remove(worked_path)
    found = False
    for root, _, files in os.walk(out_root):
        if 'records.json' not in files:
            continue
        with open(os.path.join(root, 'records.json')) as f:
            d = json.load(f)
        ts = d.get('train_shares')
        if ts and len(ts) >= 60:
            label = f"{d['group']}/{d['exp']}/seed{d['seed']}"
            with open(worked_path, 'a') as f:
                f.write(format_ei_worked(ts, label))
            found = True
            break
    if not found:
        with open(worked_path, 'w') as f:
            f.write("no train_shares >= 60 episodes; run core first\n")

    print("\n" + "=" * 70)
    print("V3 B 方案判定")
    print("=" * 70)
    print(f"  Baseline rejected_rate = {b_rej:.3f}   (V3 目标: <0.15)")
    print(f"  Baseline/NoNetwork     = {B_ratio_net:.2f}  (> 3)   → {B_cond_net}")
    print(f"  Baseline/NoSocial      = {B_ratio_soc:.2f}  (> 1.5) → {B_cond_soc}")
    print(f"  persistence            = {b_persist:.3f}   (> 0)   → {B_cond_persist}")
    print(f"  >>> {'B PASS' if B_pass else 'B FAIL'} <<<\n")

    if not verdict_df.empty:
        print("=== Core Verdicts ===")
        print(verdict_df.to_string(index=False))

    if not sub_df.empty:
        print("\n=== Subsidy Scan ===")
        print(sub_df.to_string(index=False))

    if not tax_df.empty:
        print("\n=== Carbon Tax Scan ===")
        print(tax_df.to_string(index=False))

    print(f"\n结果保存在 {sd}/")