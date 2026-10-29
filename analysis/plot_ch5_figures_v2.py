# -*- coding: utf-8 -*-
"""
plot_ch5_figures_v2.py — 第五章 6 张论文级图（Nature NPG 配色）

Figure 1  Overview (3×3)               主图，整体统揽
Figure 2  Temporal & variance (1×3)    R_init/R_final/R_overall dz + 全指标 + SD ratio
Figure 3  Core dynamics (1×2)          学习曲线 + 山脊
Figure 4  Norm decomposition (1×3)     瀑布 + 雷达 + Dumbbell 30%
Figure 5  Policy tools (2×2)           双轴 + 山脊subsidy + 山脊purchase + 雷达30%

输出：figures_ch5/
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
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10,
    'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': 0.7,
    'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'lines.linewidth': 1.4,
    'figure.dpi': 150, 'savefig.dpi': 600,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

C = {
    'Baseline': '#3C5488', 'NoNetwork': '#E64B35',
    'NoSocial_030': '#925E9F', 'NoSeed': '#00A087',
    'Full_060': '#F39B7F',
    'Subsidy_00': '#A6CEE3', 'Subsidy_10': '#4DBBD5',
    'Subsidy_20': '#3C5488', 'Subsidy_30': '#1F2A44',
    'CarbonTax_10': '#FFDC91', 'CarbonTax_20': '#F39B7F',
    'CarbonTax_30': '#E64B35',
    'Safety_High_1D': '#BC3C29', 'Safety_Low_1D': '#0072B5',
    'PriceNeutral_0Sub': '#4DBBD5', 'PriceNeutral_20Sub': '#3C5488',
}
MK = {'Baseline': 'o', 'NoNetwork': 'X', 'NoSocial_030': 's',
      'NoSeed': 'D', 'Full_060': '^'}
LBL = {'Baseline': 'Baseline', 'NoNetwork': 'No Network', 'NoSocial_030': 'No Social',
       'NoSeed': 'No Seed', 'Full_060': 'Full (w=0.6)'}
CORE_ORDER = ['Baseline', 'NoNetwork', 'NoSocial_030', 'NoSeed', 'Full_060']
OUT = 'figures_ch5'
os.makedirs(OUT, exist_ok=True)


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
                    fr = d.get('frozen', {})
                    seen[key] = {
                        'group': d.get('group', group),
                        'exp': d.get('exp', exp_dir),
                        'seed': d.get('seed', int(sdir.replace('seed', ''))),
                        'R_init': float(np.mean(r[:10])) if n >= 10 else np.nan,
                        'R_overall': float(np.mean(r)) if n > 0 else np.nan,
                        'R_final': float(np.mean(r[-50:])) if n >= 50 else np.nan,
                        'green_share': fr.get('green_share', np.nan),
                        'cond_g': fr.get('conditional_green_rate', np.nan),
                        'persistence': fr.get('persistence', np.nan),
                        'purchase_rate': fr.get('purchase_rate', np.nan),
                        'afford_green_ratio': fr.get('afford_green_ratio', np.nan),
                        'records': d.get('records', []),
                    }
    return pd.DataFrame(list(seen.values()))


def curves_for(df, exp, group):
    rows = []
    for _, r in df[(df['group'] == group) & (df['exp'] == exp)].iterrows():
        for ep in r['records']:
            if 'final' in ep:
                continue
            rows.append({'seed': r['seed'],
                         'episode': ep.get('episode', np.nan),
                         'green_share': ep.get('green_share', np.nan)})
    if not rows:
        return None
    cdf = pd.DataFrame(rows)
    g = cdf.groupby('episode')['green_share']
    return dict(ep=np.array(sorted(cdf['episode'].unique())),
                mean=g.mean().values, sd=g.std(ddof=1).values)


def paired(b, o):
    b = np.array(b)
    o = np.array(o)
    d = b - o
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
    return dict(delta=md, sd=sd, p=p, dz=dz, ci_lo=ci[0], ci_hi=ci[1], raw=d)


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


def clean(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def tag(ax, s):
    ax.text(-0.14, 1.1, s, transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top')


def save(fig, name):
    for ext in ['pdf', 'svg']:
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), dpi=600)
    plt.show()
    plt.close(fig)
    print(f"[plot] {name}.pdf/svg")


df = find_all_records()
if df.empty:
    print("[error] no data")
    sys.exit(1)
print(f"[load] {len(df)} seeds, {df['group'].nunique()} groups, {df['exp'].nunique()} conditions")


# ============================================================
# Figure 1（主图，整体统揽，3×3）
# ============================================================
def figure1_overview(df):
    fig = plt.figure(figsize=(12, 10), dpi=300)
    gs = GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)

    base = df[(df['group'] == 'core') & (df['exp'] == 'Baseline')].set_index('seed')['R_overall']

    # (a) 森林
    ax = fig.add_subplot(gs[0, 0])
    items = []
    for group in ['core', 'subsidy_scan', 'carbontax_scan', 'single_dim', 'price_neutral']:
        for exp in df[df['group'] == group]['exp'].unique():
            if exp == 'Baseline':
                continue
            sub = df[(df['group'] == group) & (df['exp'] == exp)].set_index('seed')['R_overall']
            common = sorted(set(base.index) & set(sub.index))
            if len(common) < 2:
                continue
            st = paired(base.loc[common].values, sub.loc[common].values)
            items.append({'exp': exp, 'group': group, **st})
    if items:
        idf = pd.DataFrame(items).sort_values('delta').reset_index(drop=True)
        ax.axvline(0, color='black', lw=0.7, ls='--', alpha=0.5)
        for i, r in idf.iterrows():
            c = {'core': C['Baseline'], 'subsidy_scan': C['Subsidy_10'],
                 'carbontax_scan': C['CarbonTax_30'], 'single_dim': C['NoSeed'],
                 'price_neutral': C['Full_060']}.get(r['group'], 'gray')
            ax.plot([r['ci_lo'], r['ci_hi']], [i, i], color=c, lw=1.5)
            ax.plot(r['delta'], i, 'o', color=c, ms=5,
                    markeredgecolor='black', markeredgewidth=0.3)
        ax.set_yticks(range(len(idf)))
        ax.set_yticklabels([it.exp.replace('_', ' ')[:14] for it in idf.itertuples()],
                           fontsize=5.5)
        ax.set_xlabel(r'$\Delta$ vs Baseline')
        clean(ax)
        ax.grid(axis='x', alpha=0.22, ls=':')
    ax.set_title('Forest: all vs Baseline', pad=3, fontsize=9)
    tag(ax, 'a')

    # (b) 散点
    ax = fig.add_subplot(gs[0, 1])
    for group, c in [('core', C['Baseline']), ('subsidy_scan', C['Subsidy_10']),
                     ('carbontax_scan', C['CarbonTax_30']), ('single_dim', C['NoSeed']),
                     ('price_neutral', C['Full_060'])]:
        sub = df[df['group'] == group]
        ax.scatter(sub['R_overall'], sub['persistence'], s=40,
                   color=c, edgecolor='black', lw=0.3, alpha=0.75,
                   label=group, marker='o')
    ax.set_xlabel(r'R$_\mathrm{overall}$')
    ax.set_ylabel('Persistence')
    clean(ax)
    ax.grid(alpha=0.22, ls=':')
    ax.legend(frameon=False, fontsize=6, loc='upper left')
    ax.set_title('R_overall vs persistence', pad=3, fontsize=9)
    tag(ax, 'b')

    # (c) 热图
    ax = fig.add_subplot(gs[0, 2])
    metrics = ['R_overall', 'green_share', 'persistence', 'purchase_rate']
    metric_lbl = ['R_overall', 'Green', 'Persist.', 'Purchase']
    all_exps = []
    for group in ['core', 'subsidy_scan', 'carbontax_scan', 'single_dim', 'price_neutral']:
        for exp in df[df['group'] == group]['exp'].unique():
            all_exps.append((group, exp))
    mat = []
    for group, exp in all_exps:
        row = []
        for m in metrics:
            v = df[(df['group'] == group) & (df['exp'] == exp)][m].dropna()
            row.append(v.mean() if len(v) > 0 else np.nan)
        mat.append(row)
    mat = np.array(mat)
    im = ax.imshow(mat, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center',
                        fontsize=5.5, color='black')
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metric_lbl, fontsize=6)
    ax.set_yticks(range(len(all_exps)))
    ax.set_yticklabels([e[1].replace('_', ' ')[:14] for e in all_exps], fontsize=5)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    ax.set_title('Heatmap: all conditions', pad=3, fontsize=9)
    tag(ax, 'c')

    # (d) Violin per-group
    ax = fig.add_subplot(gs[1, 0])
    groups_order = ['core', 'subsidy_scan', 'carbontax_scan', 'single_dim', 'price_neutral']
    data, lbls, cols = [], [], []
    for g in groups_order:
        v = df[df['group'] == g]['R_overall'].dropna().values
        if len(v) < 2:
            continue
        data.append(v)
        lbls.append(g)
        cols.append({'core': C['Baseline'], 'subsidy_scan': C['Subsidy_10'],
                     'carbontax_scan': C['CarbonTax_30'], 'single_dim': C['NoSeed'],
                     'price_neutral': C['Full_060']}[g])
    if data:
        vp = ax.violinplot(data, positions=range(len(data)), widths=0.6,
                           showmeans=True, showmedians=True)
        for p, c in zip(vp['bodies'], cols):
            p.set_facecolor(c)
            p.set_alpha(0.7)
            p.set_edgecolor('black')
        for part in ['cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes']:
            if part in vp:
                vp[part].set_color('black')
                vp[part].set_linewidth(0.5)
        ax.set_xticks(range(len(lbls)))
        ax.set_xticklabels(lbls, rotation=20, ha='right', fontsize=7)
        ax.set_ylabel(r'R$_\mathrm{overall}$')
        clean(ax)
        ax.grid(axis='y', alpha=0.22, ls=':')
    ax.set_title('Violin: per-group', pad=3, fontsize=9)
    tag(ax, 'd')

    # (e) Slope 30%
    ax = fig.add_subplot(gs[1, 1])
    sub30 = df[(df['group'] == 'subsidy_scan') & (df['exp'] == 'Subsidy_30')].set_index('seed')['R_overall']
    tax30 = df[(df['group'] == 'carbontax_scan') & (df['exp'] == 'CarbonTax_30')].set_index('seed')['R_overall']
    common = sorted(set(sub30.index) & set(tax30.index))
    for s in common:
        ax.plot([0, 1], [sub30.loc[s], tax30.loc[s]], color='gray',
                alpha=0.6, lw=0.9, marker='o', ms=4, markerfacecolor='white')
    if len(common) > 0:
        m0 = np.mean([sub30.loc[s] for s in common])
        m1 = np.mean([tax30.loc[s] for s in common])
        ax.plot([0, 1], [m0, m1], color=C['CarbonTax_30'], lw=2.5,
                marker='o', ms=10, markeredgecolor='black', markeredgewidth=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Subsidy 30%', 'Carbon tax 30%'])
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylabel(r'R$_\mathrm{overall}$')
    clean(ax)
    ax.grid(axis='y', alpha=0.22, ls=':')
    ax.set_title('Slope: subsidy vs tax', pad=3, fontsize=9)
    tag(ax, 'e')

    # (f) 极坐标 core
    ax = fig.add_subplot(gs[1, 2], projection='polar')
    core_exps = ['Baseline', 'NoNetwork', 'NoSocial_030', 'NoSeed', 'Full_060']
    angles = np.linspace(0, 2 * np.pi, len(core_exps), endpoint=False)
    vals = []
    for exp in core_exps:
        v = df[(df['group'] == 'core') & (df['exp'] == exp)]['R_overall'].dropna()
        vals.append(v.mean() if len(v) > 0 else 0)
    width = 2 * np.pi / len(core_exps) * 0.72
    colors_c = [C.get(e, 'gray') for e in core_exps]
    ax.bar(angles, vals, width=width, color=colors_c,
           edgecolor='black', lw=0.5, alpha=0.85)
    for ang, v, exp in zip(angles, vals, core_exps):
        ax.text(ang, v + 0.06, f'{v:.2f}', ha='center', fontsize=7)
    ax.set_xticks(angles)
    ax.set_xticklabels([LBL[e] for e in core_exps], fontsize=6.5)
    ax.set_ylim(0, 0.9)
    ax.set_yticks([0.3, 0.6])
    ax.set_yticklabels(['0.3', '0.6'], fontsize=6)
    ax.set_title('Polar: core conditions', pad=10, fontsize=9)
    tag(ax, 'f')

    # (g) Bump
    ax = fig.add_subplot(gs[2, 0])
    metrics_bump = ['R_overall', 'green_share', 'persistence']
    items_b = []
    for group in ['core', 'subsidy_scan', 'carbontax_scan', 'single_dim', 'price_neutral']:
        for exp in df[df['group'] == group]['exp'].unique():
            items_b.append((group, exp))
    for m_idx, m in enumerate(metrics_bump):
        vals_b = []
        for group, exp in items_b:
            v = df[(df['group'] == group) & (df['exp'] == exp)][m].dropna()
            vals_b.append(v.mean() if len(v) > 0 else np.nan)
        order = np.argsort([-v if not np.isnan(v) else 999 for v in vals_b])
        rank = [0] * len(vals_b)
        for r, oi in enumerate(order):
            rank[oi] = r + 1
        color_b = [C['Baseline'], C['Subsidy_10'], C['CarbonTax_30']][m_idx]
        ax.scatter(rank, [m_idx] * len(rank), s=15, alpha=0.6, color=color_b)
    ax.set_yticks(range(len(metrics_bump)))
    ax.set_yticklabels(metrics_bump)
    ax.set_xlabel('Rank (1 = highest)')
    ax.set_title('Bump: metric rankings', pad=3, fontsize=9)
    clean(ax)
    ax.grid(axis='x', alpha=0.22, ls=':')
    tag(ax, 'g')

    # (h) 相关矩阵
    ax = fig.add_subplot(gs[2, 1])
    metric_corr = ['R_overall', 'green_share', 'persistence', 'purchase_rate', 'afford_green_ratio']
    corr = df[metric_corr].corr(method='spearman')
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    for i in range(len(metric_corr)):
        for j in range(len(metric_corr)):
            ax.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center',
                    fontsize=6.5, color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black')
    ax.set_xticks(range(len(metric_corr)))
    ax.set_xticklabels([m.replace('_', ' ')[:8] for m in metric_corr],
                       fontsize=6, rotation=20, ha='right')
    ax.set_yticks(range(len(metric_corr)))
    ax.set_yticklabels([m.replace('_', ' ')[:8] for m in metric_corr], fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    ax.set_title('Correlation matrix', pad=3, fontsize=9)
    tag(ax, 'h')

    # (i) Bar 工具汇总
    ax = fig.add_subplot(gs[2, 2])
    sub_levels = ['Subsidy_00', 'Subsidy_10', 'Subsidy_20', 'Subsidy_30']
    tax_levels = ['CarbonTax_10', 'CarbonTax_20', 'CarbonTax_30']
    labels_i, vals_i, colors_i = [], [], []
    for exp in sub_levels:
        v = df[(df['group'] == 'subsidy_scan') & (df['exp'] == exp)]['R_overall'].dropna()
        if len(v) == 0:
            continue
        labels_i.append('S' + exp.replace('Subsidy_', ''))
        vals_i.append(v.mean())
        colors_i.append(C.get(exp, 'gray'))
    for exp in tax_levels:
        v = df[(df['group'] == 'carbontax_scan') & (df['exp'] == exp)]['R_overall'].dropna()
        if len(v) == 0:
            continue
        labels_i.append('T' + exp.replace('CarbonTax_', ''))
        vals_i.append(v.mean())
        colors_i.append(C.get(exp, 'gray'))
    xs = np.arange(len(labels_i))
    ax.bar(xs, vals_i, color=colors_i, edgecolor='black', lw=0.5,
           alpha=0.85, width=0.7)
    for i, v in enumerate(vals_i):
        ax.text(i, v + 0.015, f'{v:.2f}', ha='center', fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels_i, rotation=30, ha='right', fontsize=7)
    ax.set_ylabel(r'R$_\mathrm{overall}$')
    ax.set_ylim(0.5, 1.0)
    clean(ax)
    ax.grid(axis='y', alpha=0.22, ls=':')
    ax.set_title('Policy tools: all levels', pad=3, fontsize=9)
    tag(ax, 'i')
    plt.show()
    save(fig, '图1')


figure1_overview(df)


# ============================================================
# Figure 2（时间尺度 + 方差，1×3）
# ============================================================
def figure2_temporal(df):
    core = df[df['group'] == 'core']
    exps = [e for e in CORE_ORDER if e in core['exp'].values]
    if not exps or 'Baseline' not in exps:
        print("[skip] Fig2")
        return
    base = core[core['exp'] == 'Baseline']
    other = [e for e in exps if e != 'Baseline']

    fig = plt.figure(figsize=(10, 2.5), dpi=300)
    gs = GridSpec(1, 3, figure=fig, wspace=0.38)

    # (a) 三指标 dz
    ax = fig.add_subplot(gs[0, 0])
    metrics = ['R_init', 'R_final', 'R_overall']
    metric_labels = [r'R$_\mathrm{init}$', r'R$_\mathrm{final}$', r'R$_\mathrm{overall}$']
    colors_m = ['#A6CEE3', '#4DBBD5', '#3C5488']
    x = np.arange(len(other))
    width = 0.26
    for j, (m, lab, cm) in enumerate(zip(metrics, metric_labels, colors_m)):
        vals = []
        for exp in other:
            sub = core[core['exp'] == exp]
            common = sorted(set(base['seed']) & set(sub['seed']))
            if len(common) < 2:
                vals.append(np.nan)
                continue
            b = base[base['seed'].isin(common)].sort_values('seed')[m].values
            o = sub[sub['seed'].isin(common)].sort_values('seed')[m].values
            st = paired(b, o)
            vals.append(st['dz'] if not np.isnan(st['dz']) else np.nan)
        ax.bar(x + (j - 1) * width, vals, width, color=cm,
               edgecolor='black', lw=0.4, label=lab)
    ax.axhline(0.8, color='gray', ls=':', lw=0.6)
    ax.text(len(other) - 2.6, 0.85, ' large\n(d=0.8)',
            fontsize=6.5, color='gray')
    ax.set_xticks(x)
    ax.set_xticklabels([LBL[e] for e in other], rotation=15, ha='right', fontsize=7)
    ax.set_ylabel("Cohen's $d_z$")
    ax.set_title('Effect size by temporal scale', pad=3, fontsize=9)
    ax.legend(frameon=False, loc='upper right', fontsize=6.5)
    clean(ax)
    ax.grid(axis='y', alpha=0.22, ls=':')
    tag(ax, 'a')

    # (b) 全指标柱状
    ax = fig.add_subplot(gs[0, 1])
    metrics2 = ['green_share', 'cond_g', 'persistence']
    labels2 = ['Green\nshare', 'Cond.\ngreen', 'Persist.']
    x = np.arange(len(metrics2))
    all_exps_b = ['Baseline'] + other
    width_b = 0.8 / (len(all_exps_b) + 0.5)
    for i, exp in enumerate(all_exps_b):
        vals = []
        for m in metrics2:
            v = core[core['exp'] == exp][m].dropna()
            vals.append(v.mean() if len(v) > 0 else np.nan)
        offset = (i - len(all_exps_b) / 2) * width_b
        ax.bar(x + offset, vals, width_b, label=LBL[exp],
               color=C[exp], edgecolor='black', lw=0.4, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels2, fontsize=7.5)
    ax.set_ylabel('Value (frozen, mean)')
    ax.set_ylim(0, 1.05)
    ax.set_title('Auxiliary metrics', pad=3, fontsize=9)
    ax.legend(frameon=False, loc='upper right', fontsize=6, ncol=2)
    clean(ax)
    ax.grid(axis='y', alpha=0.22, ls=':')
    tag(ax, 'b')

    # (c) SD ratio
    ax = fig.add_subplot(gs[0, 2])
    base_sd = base['R_overall'].std(ddof=1)
    vals_sr, lbl_sr, col_sr = [], [], []
    for exp in other:
        v = core[core['exp'] == exp]['R_overall'].dropna()
        if len(v) < 2:
            continue
        vals_sr.append(v.std(ddof=1) / base_sd if base_sd > 0 else np.nan)
        lbl_sr.append(LBL[exp])
        col_sr.append(C[exp])
    bars = ax.bar(range(len(vals_sr)), vals_sr, color=col_sr,
                  edgecolor='black', lw=0.4, alpha=0.9, width=0.62)
    for i, v in enumerate(vals_sr):
        ax.text(i, v + 0.02, f'{v:.2f}×', ha='center', fontsize=7.5)
    ax.axhline(1.0, color='gray', ls=':', lw=0.7)
    ax.text(len(vals_sr) - 4, 1.03, 'Baseline', fontsize=6.5, color='gray')
    ax.set_xticks(range(len(vals_sr)))
    ax.set_xticklabels(lbl_sr, rotation=20, ha='right', fontsize=7.5)
    ax.set_ylabel('SD ratio vs. Baseline')
    ax.set_title('Variance relative to Baseline', pad=3, fontsize=9)
    clean(ax)
    ax.grid(axis='y', alpha=0.22, ls=':')
    tag(ax, 'c')
    plt.show()
    save(fig, '图2')


figure2_temporal(df)


# ============================================================
# Figure 3（核心训练动态，1×2）
# ============================================================
def figure3_core_dynamics(df):
    core = df[df['group'] == 'core']
    exps = [e for e in CORE_ORDER if e in core['exp'].values]
    if not exps:
        print("[skip] Fig3")
        return

    fig = plt.figure(figsize=(7, 2.3), dpi=300)
    gs = GridSpec(1, 2, figure=fig, wspace=0.15)

    # (a) 学习曲线
    ax = fig.add_subplot(gs[0, 0])
    for exp in exps:
        c = curves_for(df, exp, 'core')
        if not c:
            continue
        ax.plot(c['ep'], c['mean'], color=C[exp], lw=1.5, label=LBL[exp])
        ax.fill_between(c['ep'],
                        np.clip(c['mean'] - c['sd'], 0, 1),
                        np.clip(c['mean'] + c['sd'], 0, 1),
                        color=C[exp], alpha=0.15, lw=0)
    ax.axhline(0.6, color='gray', ls=':', lw=0.7, alpha=0.6)
    ax.set_xlabel('Episode', fontsize=9.5)
    ax.set_ylabel('Green share', fontsize=9.5)
    ax.set_xlim(0, 205)
    ax.set_ylim(0, 1.02)
    ax.tick_params(labelsize=7.5)
    clean(ax)
    ax.grid(alpha=0.22, ls=':')
    ax.legend(frameon=False, fontsize=6, loc='lower right', handlelength=1.4,
              handletextpad=0.5, labelspacing=0.3)
    ax.set_title('Learning curves', pad=4, fontsize=9)
    tag(ax, 'a')

    # (b) 山脊 core R_overall
    ax = fig.add_subplot(gs[0, 1])
    offset = 0.0
    for exp in exps:
        vals = core[core['exp'] == exp]['R_overall'].dropna().values
        if len(vals) < 2:
            continue
        xs = np.linspace(0, 1, 200)
        if len(vals) > 2:
            kde = gaussian_kde(vals, bw_method=0.5)
            ys = kde(xs)
        else:
            ys = np.zeros_like(xs)
            for v in vals:
                ys += np.exp(-0.5 * ((xs - v) / 0.05) ** 2)
        ys = ys / ys.max() * 0.9
        ax.fill_between(xs, offset, offset + ys, color=C[exp],
                        alpha=0.72, lw=0.6, edgecolor='black')
        ax.plot(xs, offset + ys, color='black', lw=0.45)
        ax.axvline(vals.mean(), color=C[exp], lw=1.3, ls='--', alpha=0.9)
        ax.text(0.02, offset + 0.18, LBL[exp], ha='left', va='center',
                fontsize=7, color='black',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7))
        offset += 1.0
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.2, len(exps) + 0.2)
    ax.set_xlabel(r'R$_\mathrm{overall}$', fontsize=9.5)
    ax.set_yticks([])
    ax.tick_params(axis='x', labelsize=7.5)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Ridgeline: core distribution', pad=4, fontsize=9)
    tag(ax, 'b')

    plt.show()
    save(fig, '图3')


figure3_core_dynamics(df)


# ============================================================
# Figure 4（规范效应分解，1×3）
# ============================================================
def figure4_decomposition(df):
    core = df[df['group'] == 'core']
    if not all(e in core['exp'].values for e in ['Baseline', 'Full_060', 'NoSocial_030']):
        print("[skip] Fig4")
        return
    base = core[core['exp'] == 'Baseline'].set_index('seed')['R_overall']
    full = core[core['exp'] == 'Full_060'].set_index('seed')['R_overall']
    nosoc = core[core['exp'] == 'NoSocial_030'].set_index('seed')['R_overall']
    common = sorted(set(base.index) & set(full.index) & set(nosoc.index))
    if len(common) < 2:
        return
    b = base.loc[common].values
    f = full.loc[common].values
    n = nosoc.loc[common].values

    fig = plt.figure(figsize=(11, 3), dpi=300)
    gs = GridSpec(1, 3, figure=fig, wspace=0.42)

    # (a) 瀑布
    ax = fig.add_subplot(gs[0, 0])
    segs = [('Baseline', b.mean(), C['Baseline']),
            ('Reward\nscale', -np.mean(b - f), C['Full_060']),
            ('Pure\nnorm', -np.mean(f - n), C['NoSocial_030']),
            ('No Social', n.mean(), C['NoNetwork'])]
    cum = 0
    for i, (lab, val, c) in enumerate(segs):
        if i == 0 or i == len(segs) - 1:
            ax.bar(i, val, color=c, edgecolor='black', lw=0.5,
                   alpha=0.85, width=0.65)
            ax.text(i, val + 0.02, f'{val:.3f}', ha='center', fontsize=8)
            cum = val
        else:
            if val < 0:
                ax.bar(i, abs(val), bottom=cum + val, color=c,
                       edgecolor='black', lw=0.5, alpha=0.85, width=0.65)
            else:
                ax.bar(i, val, bottom=cum, color=c,
                       edgecolor='black', lw=0.5, alpha=0.85, width=0.65)
            ax.text(i, cum + val / 2, f'{val:+.3f}', ha='center', fontsize=8)
            cum += val
    ax.set_xticks(range(len(segs)))
    ax.set_xticklabels([s[0] for s in segs], fontsize=7.5)
    ax.set_ylabel(r'R$_\mathrm{overall}$')
    ax.set_ylim(0, 0.75)
    ax.set_title('Waterfall: Baseline → No Social', pad=3, fontsize=9)
    clean(ax)
    ax.grid(axis='y', alpha=0.22, ls=':')
    tag(ax, 'a')

    # (b) 雷达
    ax = fig.add_subplot(gs[0, 1], projection='polar')
    metrics = ['R_init', 'R_overall', 'R_final', 'green_share', 'persistence']
    metric_lbl = ['R_init', 'R_overall', 'R_final', 'Green', 'Persist.']
    allvals = []
    for m in metrics:
        v = core[core['exp'].isin(['Baseline', 'Full_060', 'NoSocial_030'])][m].dropna()
        allvals.append((v.min() if len(v) > 0 else 0,
                        v.max() if len(v) > 0 else 1))
    groups = [('Baseline', C['Baseline']),
              ('Full_060', C['Full_060']),
              ('NoSocial_030', C['NoSocial_030'])]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    for g, c in groups:
        vals = []
        for i, m in enumerate(metrics):
            v = core[core['exp'] == g][m].dropna()
            if len(v) == 0:
                vals.append(0)
                continue
            lo, hi = allvals[i]
            norm = (v.mean() - lo) / (hi - lo + 1e-9)
            vals.append(max(0, min(1, norm)))
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', color=c, lw=1.8, ms=5,
                markeredgecolor='black', markeredgewidth=0.4, label=LBL[g])
        ax.fill(angles, vals, color=c, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_lbl, fontsize=7.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=6.5)
    ax.set_title('Radar: multi-metric', pad=10, fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc='lower left',
              bbox_to_anchor=(-0.1, -0.2))
    tag(ax, 'b')

    # (c) Dumbbell 30%
    ax = fig.add_subplot(gs[0, 2])
    sub = df[df['group'] == 'subsidy_scan']
    tax = df[df['group'] == 'carbontax_scan']
    metrics_f = ['R_overall', 'green_share', 'persistence', 'purchase_rate']
    metric_lbl_f = ['R_overall', 'Green', 'Persist.', 'Purchase']
    vals_sub, vals_tax = [], []
    for m in metrics_f:
        v_sub = sub[sub['exp'] == 'Subsidy_30'][m].dropna()
        v_tax = tax[tax['exp'] == 'CarbonTax_30'][m].dropna()
        vals_sub.append(v_sub.mean() if len(v_sub) > 0 else np.nan)
        vals_tax.append(v_tax.mean() if len(v_tax) > 0 else np.nan)
    ys = np.arange(len(metrics_f))
    for i, (vs, vt) in enumerate(zip(vals_sub, vals_tax)):
        ax.plot([vs, vt], [i, i], color='gray', lw=1.5, zorder=1)
        ax.scatter(vs, i, s=50, color=C['Subsidy_30'],
                   edgecolor='black', lw=0.5, zorder=3, marker='s')
        ax.scatter(vt, i, s=50, color=C['CarbonTax_30'],
                   edgecolor='black', lw=0.5, zorder=3, marker='D')
        ax.text((vs + vt) / 2, i + 0.1, f'Δ={vt-vs:+.3f}',
                ha='center', fontsize=6.5, color='gray')
    ax.scatter([], [], s=50, color=C['Subsidy_30'], marker='s',
               edgecolor='black', lw=0.5, label='Subsidy 30%')
    ax.scatter([], [], s=50, color=C['CarbonTax_30'], marker='D',
               edgecolor='black', lw=0.5, label='Carbon tax 30%')
    ax.set_yticks(ys)
    ax.set_yticklabels(metric_lbl_f)
    ax.set_xlabel('Metric value')
    ax.set_xlim(0, 1.05)
    clean(ax)
    ax.grid(axis='x', alpha=0.22, ls=':')
    ax.legend(frameon=False, fontsize=7, loc='lower left')
    ax.set_title('Dumbbell: tools at 30%', pad=5, fontsize=9)
    tag(ax, 'c')
    plt.show()
    save(fig, '图4')


figure4_decomposition(df)


# ============================================================
# Figure 5（政策工具，2×2）
# ============================================================
def figure5_policy(df):
    sub = df[df['group'] == 'subsidy_scan']
    tax = df[df['group'] == 'carbontax_scan']
    if sub.empty:
        print("[skip] Fig5")
        return
    sub_levels = ['Subsidy_00', 'Subsidy_10', 'Subsidy_20', 'Subsidy_30']

    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.3)

    # (a) 双轴
    ax = fig.add_subplot(gs[0, 0])
    xs, m, sd = [], [], []
    for exp in sub_levels:
        s = sub[sub['exp'] == exp]['R_overall'].dropna()
        if len(s) == 0:
            continue
        xs.append(int(exp.replace('Subsidy_', '')) / 100)
        m.append(s.mean())
        sd.append(s.std(ddof=1) if len(s) > 1 else 0)
    if xs:
        ax.errorbar(xs, m, yerr=sd, fmt='o-', color=C['Subsidy_20'],
                    lw=2, ms=6, capsize=5, markeredgecolor='black',
                    markeredgewidth=0.5, label='R_overall')
    ax2 = ax.twinx()
    if xs:
        ratio = [sd[i] / sd[0] if sd[0] > 0 else np.nan for i in range(len(sd))]
        ax2.plot(xs, ratio, 's--', color=C['NoSocial_030'], lw=1.8, ms=6,
                 markeredgecolor='black', markeredgewidth=0.5, label='SD ratio')
    ax.set_xlabel('Subsidy rate')
    ax.set_ylabel(r'R$_\mathrm{overall}$', color=C['Subsidy_20'])
    ax2.set_ylabel('SD ratio', color=C['NoSocial_030'])
    ax.set_xlim(-0.02, 0.32)
    ax.set_ylim(0.55, 1.0)
    ax2.set_ylim(0, 1.2)
    ax.set_xticks([0, 0.1, 0.2, 0.3])
    ax.set_xticklabels(['0%', '10%', '20%', '30%'])
    ax.tick_params(axis='y', labelcolor=C['Subsidy_20'])
    ax2.tick_params(axis='y', labelcolor=C['NoSocial_030'])
    clean(ax)
    ax.grid(alpha=0.22, ls=':')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False,
              fontsize=7, loc='upper right')
    ax.set_title('Subsidy: dual axis', pad=3, fontsize=9)
    tag(ax, 'a')

    # (b) 山脊 subsidy
    ax = fig.add_subplot(gs[0, 1])
    offset = 0
    for exp in sub_levels:
        vals = sub[sub['exp'] == exp]['R_overall'].dropna().values
        if len(vals) < 2:
            continue
        xs_l = np.linspace(0.4, 1.0, 200)
        kde = gaussian_kde(vals, bw_method=0.5)
        ys = kde(xs_l)
        ys = ys / ys.max() * 0.9
        rate = int(exp.replace('Subsidy_', ''))
        ax.fill_between(xs_l, offset, offset + ys, color=C.get(exp, 'gray'),
                        alpha=0.75, lw=0.5, edgecolor='black')
        ax.plot(xs_l, offset + ys, color='black', lw=0.4)
        ax.axvline(vals.mean(), color=C.get(exp, 'gray'), lw=1.2, ls='--')
        ax.text(0.42, offset + 0.1, f'{rate}%', ha='left', va='center', fontsize=8)
        offset += 1.0
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(-0.2, len(sub_levels) + 0.2)
    ax.set_xlabel(r'R$_\mathrm{overall}$')
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Ridgeline: subsidy levels', pad=3, fontsize=9)
    tag(ax, 'b')

    # (c) 山脊 purchase_rate
    ax = fig.add_subplot(gs[1, 0])
    offset = 0
    for exp in sub_levels:
        vals = sub[sub['exp'] == exp]['purchase_rate'].dropna().values
        if len(vals) < 2:
            continue
        xs_l = np.linspace(0.3, 0.75, 200)
        kde = gaussian_kde(vals, bw_method=0.5)
        ys = kde(xs_l)
        ys = ys / ys.max() * 0.85
        rate = int(exp.replace('Subsidy_', ''))
        ax.fill_between(xs_l, offset, offset + ys, color=C.get(exp, 'gray'),
                        alpha=0.75, lw=0.5, edgecolor='black')
        ax.plot(xs_l, offset + ys, color='black', lw=0.4)
        ax.axvline(vals.mean(), color=C.get(exp, 'gray'), lw=1.2, ls='--')
        ax.text(0.31, offset + 0.1, f'{rate}%', ha='left', va='center', fontsize=8)
        offset += 1.0
    ax.set_xlim(0.3, 0.75)
    ax.set_ylim(-0.2, len(sub_levels) + 0.2)
    ax.set_xlabel('Purchase rate')
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Ridgeline: purchase rate', pad=3, fontsize=9)
    tag(ax, 'c')

    # (d) 雷达 30% vs 30%
    ax = fig.add_subplot(gs[1, 1], projection='polar')
    metrics = ['R_overall', 'green_share', 'persistence', 'purchase_rate', 'afford_green_ratio']
    metric_lbl = ['R_overall', 'Green', 'Persist.', 'Purchase', 'Afford']
    subs30 = sub[sub['exp'] == 'Subsidy_30']
    tax30 = tax[tax['exp'] == 'CarbonTax_30']
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    for grp, c, lab in [(subs30, C['Subsidy_30'], 'Subsidy 30%'),
                        (tax30, C['CarbonTax_30'], 'Carbon tax 30%')]:
        vals = []
        for m in metrics:
            v = grp[m].dropna()
            vals.append(v.mean() if len(v) > 0 else 0)
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', color=c, lw=1.8, ms=6,
                markeredgecolor='black', markeredgewidth=0.5, label=lab)
        ax.fill(angles, vals, color=c, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_lbl, fontsize=7.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.5, 1.0])
    ax.set_yticklabels(['0.5', '1.0'], fontsize=6)
    ax.set_title('Radar: tools at 30%', pad=10, fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc='lower left',
              bbox_to_anchor=(-0.3, -0.1))
    tag(ax, 'd')
    plt.show()
    save(fig, '图5')


figure5_policy(df)