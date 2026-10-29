# -*- coding: utf-8 -*-
"""
print_ch5_tables.py — 第五章实验部分的表格（打印到控制台）

覆盖 6 张表：
  Table 1  All conditions descriptive       (综合描述)
  Table 2  Core mechanism comparison         (主表)
  Table 3  Decomposition of norm effect      (关键发现)
  Table 4  Subsidy scan                      (政策)
  Table 5  Carbon tax scan                   (政策)
  Table 6  Single-dim & price-neutral        (稳健性)

用法：python analysis/print_ch5_tables.py
依赖：pandas, numpy, scipy
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import json
import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# 探测目录
# ============================================================
def find_all_records():
    candidates = ['.', 'runs_B_v3', '全量涌现', '全量涌现/runs_B_v3']
    try:
        for d in sorted(os.listdir('.')):
            if not os.path.isdir(d):
                continue
            candidates.append(d)
            candidates.append(os.path.join(d, 'runs_B_v3'))
    except Exception:
        pass
    groups = ['core', 'subsidy_scan', 'carbontax_scan', 'single_dim', 'price_neutral']
    seen = {}
    for root in candidates:
        if not os.path.isdir(root):
            continue
        for group in groups:
            gdir = os.path.join(root, group)
            if not os.path.isdir(gdir):
                continue
            for exp_dir in sorted(os.listdir(gdir)):
                edir = os.path.join(gdir, exp_dir)
                if not os.path.isdir(edir):
                    continue
                for sdir in sorted(os.listdir(edir)):
                    if not sdir.startswith('seed'):
                        continue
                    p = os.path.join(edir, sdir, 'records.json')
                    if not os.path.exists(p):
                        continue
                    try:
                        with open(p) as f:
                            d = json.load(f)
                    except Exception:
                        continue
                    key = (d.get('group', group), d.get('exp', exp_dir),
                           d.get('seed', int(sdir.replace('seed', ''))))
                    if key in seen:
                        continue
                    ts = d.get('train_shares', [])
                    if not ts:
                        continue
                    r = np.array(ts, dtype=float)
                    n = len(r)
                    frozen = d.get('frozen', {})
                    seen[key] = {
                        'group': d.get('group', group),
                        'exp': d.get('exp', exp_dir),
                        'seed': d.get('seed', int(sdir.replace('seed', ''))),
                        'R_overall': float(np.mean(r)) if n > 0 else np.nan,
                        'R_init': float(np.mean(r[:10])) if n >= 10 else np.nan,
                        'R_final': float(np.mean(r[-50:])) if n >= 50 else np.nan,
                        'green_share': frozen.get('green_share', np.nan),
                        'persistence': frozen.get('persistence', np.nan),
                        'purchase_rate': frozen.get('purchase_rate', np.nan),
                        'afford_green_ratio': frozen.get('afford_green_ratio', np.nan),
                    }
    return pd.DataFrame(list(seen.values()))


# ============================================================
# 统计工具
# ============================================================
def desc(vals):
    v = np.array([x for x in vals if not pd.isna(x)])
    if len(v) == 0:
        return np.nan, np.nan, 0
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0), len(v)


def paired(b, o):
    b = np.array(b)
    o = np.array(o)
    d = o - b
    n = len(d)
    md = float(np.mean(d))
    sd = float(np.std(d, ddof=1)) if n > 1 else 0.0
    if n > 1 and sd > 0:
        t, p = stats.ttest_1samp(d, 0)
        dz = md / sd
        se = sd / np.sqrt(n)
        ci = stats.t.interval(0.95, df=n - 1, loc=md, scale=se)
    else:
        t = p = dz = np.nan
        ci = (np.nan, np.nan)
    return dict(n=n, md=md, sd=sd, t=t, p=p, dz=dz,
                ci_lo=ci[0], ci_hi=ci[1], d=d)


def fmt_p(p):
    if pd.isna(p):
        return '—'
    if p < 0.001:
        return '<.001'
    return f'{p:.3f}'


def stars(p):
    if pd.isna(p):
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    if p < 0.10:
        return '†'
    return 'n.s.'


def fmt_md(m, s):
    if pd.isna(m):
        return '—'
    return f'{m:.3f}±{s:.3f}'


# ============================================================
# 打印工具
# ============================================================
def print_table(title, note, headers, rows):
    all_rows = [headers] + rows
    widths = []
    for j in range(len(headers)):
        w = max(len(str(r[j])) for r in all_rows)
        widths.append(w)
    sep = '+' + '+'.join('-' * (w + 2) for w in widths) + '+'

    print("\n" + "=" * len(sep))
    print(f"  {title}")
    print("=" * len(sep))
    print(sep)
    print('|' + '|'.join(f' {str(h).ljust(w)} ' for h, w in zip(headers, widths)) + '|')
    print(sep)
    for r in rows:
        print('|' + '|'.join(f' {str(c).ljust(w)} ' for c, w in zip(r, widths)) + '|')
    print(sep)
    if note:
        print(f"  {note}")
    print()


# ============================================================
# 表 1：所有条件描述性统计
# ============================================================
def table1(df):
    rows = []
    for g in ['core', 'subsidy_scan', 'carbontax_scan', 'single_dim', 'price_neutral']:
        sub = df[df['group'] == g]
        if sub.empty:
            continue
        for exp in sorted(sub['exp'].unique()):
            s = sub[sub['exp'] == exp]
            m, sd, n = desc(s['R_overall'].values)
            gs_mean, _, _ = desc(s['green_share'].values)
            pe_mean, _, _ = desc(s['persistence'].values)
            rows.append([
                exp, g, n,
                fmt_md(m, sd),
                f'{gs_mean:.3f}' if not pd.isna(gs_mean) else '—',
                f'{pe_mean:.3f}' if not pd.isna(pe_mean) else '—',
            ])
    print_table(
        "Table 1.  All conditions — descriptive statistics (R_overall primary)",
        "Format: mean±SD. R_overall = mean green share over episodes 1–200.",
        ['Condition', 'Group', 'n', 'R_overall', 'green_share', 'persistence'],
        rows)


# ============================================================
# 表 2：core 配对对比
# ============================================================
def table2(df):
    core = df[df['group'] == 'core']
    base = core[core['exp'] == 'Baseline'].set_index('seed')['R_overall']
    if base.empty:
        return
    rows = []
    order = ['NoNetwork', 'Full_060', 'NoSocial_030', 'NoSeed']
    for exp in order:
        sub = core[core['exp'] == exp].set_index('seed')['R_overall']
        common = sorted(set(base.index) & set(sub.index))
        if len(common) < 2:
            continue
        b = base.loc[common].values
        o = sub.loc[common].values
        st = paired(b, o)
        m_base, sd_base, _ = desc(b)
        m_cond, sd_cond, _ = desc(o)
        sd_ratio = sd_cond / sd_base if sd_base > 0 else np.nan
        rows.append([
            f'Baseline − {exp}', st['n'],
            fmt_md(m_base, sd_base),
            fmt_md(m_cond, sd_cond),
            f'{st["md"]:+.3f}',
            f'{st["dz"]:+.2f}',
            f'[{st["ci_lo"]:+.3f}, {st["ci_hi"]:+.3f}]',
            fmt_p(st['p']),
            stars(st['p']),
            f'{sd_ratio:.2f}×' if not pd.isna(sd_ratio) else '—',
        ])
    print_table(
        "Table 2.  Core mechanism comparison (paired, ref = Baseline)",
        "Δ = Baseline − condition (negative = condition lower). d_z = Cohen's d_z. Paired t-test, n=5.",
        ['Contrast', 'n', 'Baseline', 'Condition', 'Δ', 'd_z', '95% CI', 'p', 'Sig.', 'SD ratio'],
        rows)


# ============================================================
# 表 3：规范效应分解
# ============================================================
def table3(df):
    core = df[df['group'] == 'core']
    base = core[core['exp'] == 'Baseline'].set_index('seed')['R_overall']
    full = core[core['exp'] == 'Full_060'].set_index('seed')['R_overall']
    nosoc = core[core['exp'] == 'NoSocial_030'].set_index('seed')['R_overall']
    if base.empty or full.empty or nosoc.empty:
        return
    common = sorted(set(base.index) & set(full.index) & set(nosoc.index))
    if len(common) < 2:
        return
    b = base.loc[common].values
    f = full.loc[common].values
    n = nosoc.loc[common].values

    rows = []
    components = [
        ('Apparent norm (Baseline − No Social)', b - n),
        ('Reward-scale (Baseline − Full 0.6)', b - f),
        ('Pure norm (Full 0.6 − No Social)', f - n),
    ]
    for name, d in components:
        sd = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
        if len(d) > 1 and sd > 0:
            t, p = stats.ttest_1samp(d, 0)
            dz = float(np.mean(d)) / sd
            se = sd / np.sqrt(len(d))
            ci = stats.t.interval(0.95, df=len(d) - 1, loc=float(np.mean(d)), scale=se)
        else:
            t = p = dz = np.nan
            ci = (np.nan, np.nan)
        rows.append([
            name,
            f'{float(np.mean(d)):+.3f}',
            f'{sd:.3f}',
            f'{dz:+.2f}' if not pd.isna(dz) else '—',
            f'[{ci[0]:+.3f}, {ci[1]:+.3f}]' if not pd.isna(ci[0]) else '—',
            fmt_p(p),
            stars(p),
        ])
    print_table(
        "Table 3.  Decomposition of the social-norm effect",
        "Key finding: 66% of the apparent norm effect is a reward-scale artifact.",
        ['Component', 'Δ', 'SD', 'd_z', '95% CI', 'p', 'Sig.'],
        rows)


# ============================================================
# 表 4：subsidy 扫描
# ============================================================
def table4(df):
    sub = df[df['group'] == 'subsidy_scan']
    if sub.empty:
        return
    base_sd = None
    for exp in ['Subsidy_00']:
        if exp in sub['exp'].values:
            _, base_sd, _ = desc(sub[sub['exp'] == exp]['R_overall'].values)
    rows = []
    for exp in ['Subsidy_00', 'Subsidy_10', 'Subsidy_20', 'Subsidy_30']:
        if exp not in sub['exp'].values:
            continue
        s = sub[sub['exp'] == exp]
        m, sd, n = desc(s['R_overall'].values)
        gs, _, _ = desc(s['green_share'].values)
        pe, _, _ = desc(s['persistence'].values)
        pr, _, _ = desc(s['purchase_rate'].values)
        sd_ratio = sd / base_sd if base_sd and base_sd > 0 else np.nan
        rate = int(exp.replace('Subsidy_', ''))
        rows.append([
            f'{rate}%', n,
            fmt_md(m, sd),
            f'{sd_ratio:.2f}×' if not pd.isna(sd_ratio) else '—',
            f'{gs:.3f}' if not pd.isna(gs) else '—',
            f'{pe:.3f}' if not pd.isna(pe) else '—',
            f'{pr:.3f}' if not pd.isna(pr) else '—',
        ])
    print_table(
        "Table 4.  Subsidy scan (3 seeds per level)",
        "SD ratio = SD at subsidy / SD at 0%. green_share & persistence from frozen eval.",
        ['Subsidy', 'n', 'R_overall', 'SD ratio', 'green_share', 'persistence', 'purchase_rate'],
        rows)


# ============================================================
# 表 5：碳税扫描
# ============================================================
def table5(df):
    sub = df[df['group'] == 'carbontax_scan']
    if sub.empty:
        print("\n  [Table 5]  Carbon tax scan — no data yet\n")
        return
    rows = []
    exps = sorted(sub['exp'].unique())
    for exp in exps:
        s = sub[sub['exp'] == exp]
        m, sd, n = desc(s['R_overall'].values)
        gs, _, _ = desc(s['green_share'].values)
        af, _, _ = desc(s['afford_green_ratio'].values)
        rate = exp.replace('CarbonTax_', '')
        rows.append([
            f'{int(rate)}%', n,
            fmt_md(m, sd),
            f'{gs:.3f}' if not pd.isna(gs) else '—',
            f'{af:.3f}' if not pd.isna(af) else '—',
        ])
    print_table(
        "Table 5.  Carbon tax scan",
        "afford_green = share of decisions where green was affordable.",
        ['Carbon tax', 'n', 'R_overall', 'green_share', 'afford_green'],
        rows)


# ============================================================
# 表 6：单维操纵 + 价格中性
# ============================================================
def table6(df):
    rows = []

    sd_group = df[df['group'] == 'single_dim']
    if not sd_group.empty:
        exps = sorted(sd_group['exp'].unique())
        if len(exps) >= 2:
            m_high, s_high, _ = desc(sd_group[sd_group['exp'] == exps[0]]['R_overall'].values)
            m_low, s_low, _ = desc(sd_group[sd_group['exp'] == exps[1]]['R_overall'].values)
            delta = m_high - m_low
            interpretation = 'High > Low' if delta > 0 else 'Low > High'
            rows.append(['Single-dim',
                         f'{exps[0].replace("_1D", "")} vs {exps[1].replace("_1D", "")}',
                         fmt_md(m_high, s_high),
                         fmt_md(m_low, s_low),
                         f'{delta:+.3f}',
                         interpretation])

    pn_group = df[df['group'] == 'price_neutral']
    if not pn_group.empty:
        exps = sorted(pn_group['exp'].unique())
        if len(exps) >= 2:
            m_0, s_0, _ = desc(pn_group[pn_group['exp'] == exps[0]]['R_overall'].values)
            m_20, s_20, _ = desc(pn_group[pn_group['exp'] == exps[1]]['R_overall'].values)
            delta = m_20 - m_0
            interpretation = '20% > 0%' if delta > 0 else '0% > 20%'
            rows.append(['Price-neutral',
                         '20% subsidy vs 0% subsidy',
                         fmt_md(m_20, s_20),
                         fmt_md(m_0, s_0),
                         f'{delta:+.3f}',
                         interpretation])

    if not rows:
        print("\n  [Table 6]  Single-dim & price-neutral — no data yet\n")
        return

    print_table(
        "Table 6.  Single-dimension manipulation & price-neutrality control",
        "Descriptive comparison (n=3 per condition). Interpretation is directional only.",
        ['Test', 'Contrast', 'Condition A', 'Condition B', 'Δ (A − B)', 'Interpretation'],
        rows)


# ============================================================
# 主入口
# ============================================================
def main():
    print("[scan] detecting all runs_B_v3 directories...")
    df = find_all_records()
    if df.empty:
        print("[error] No records.json found")
        sys.exit(1)

    print(f"[load] total {len(df)} seed-samples, "
          f"{df['group'].nunique()} groups, {df['exp'].nunique()} conditions")
    for g in ['core', 'subsidy_scan', 'carbontax_scan', 'single_dim', 'price_neutral']:
        n = df[df['group'] == g]['exp'].nunique() if g in df['group'].values else 0
        m = len(df[df['group'] == g]) if g in df['group'].values else 0
        print(f"  {g:20s}: {n:2d} conditions, {m:2d} seeds")

    print("\n" + "█" * 70)
    print("  CHAPTER 5 — EXPERIMENTAL RESULTS")
    print("█" * 70)

    table1(df)
    table2(df)
    table3(df)
    table4(df)
    table5(df)
    table6(df)

    print("\n" + "█" * 70)
    print("  End of Chapter 5 tables")
    print("█" * 70)


if __name__ == '__main__':
    main()