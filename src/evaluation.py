# -*- coding: utf-8 -*-
"""评估与 EI 计算。"""
import numpy as np
import torch


@torch.no_grad()
def evaluate(env, policies, device, n_eps=5, record_steps=False):
    agg = {'green': 0, 'regular': 0, 'no_purchase': 0, 'rejected': 0}
    ep_shares = []
    aff_reg = aff_grn = 0
    traces = []
    for ep in range(n_eps):
        rng = np.random.RandomState(seed=1000 + ep)
        obs = env.reset(rng=rng)
        done = False
        steps = []
        step_trace = []
        while not done:
            actions = []
            for i, p in enumerate(policies):
                pr = p(torch.FloatTensor(obs[i]).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
                actions.append(int(np.argmax(pr)))
            n_obs, _, done, info = env.step(actions)
            for k in agg:
                agg[k] += info[k]
            aff_reg += info['n_afford_regular']
            aff_grn += info['n_afford_green']
            steps.append(info['green_share_step'])
            if record_steps:
                step_trace.append({'step': env.current_step,
                                   'green': info['green'], 'regular': info['regular'],
                                   'no_purchase': info['no_purchase'], 'rejected': info['rejected'],
                                   'green_share_step': info['green_share_step'],
                                   'r_local_mean': info['r_local_mean'],
                                   'afford_normal': env.affordability()['afford_normal'],
                                   'afford_green': env.affordability()['afford_green']})
            obs = n_obs
        ep_shares.append(steps)
        if record_steps:
            traces.append(step_trace)
    tot = agg['green'] + agg['regular'] + agg['no_purchase'] + agg['rejected']
    ep_final = [np.mean(s[-10:]) if len(s) >= 10 else (np.mean(s) if s else 0.0) for s in ep_shares]
    persist = []
    for s in ep_shares:
        run = long = 0
        for v in s:
            if v >= 0.6:
                run += 1
                long = max(long, run)
            else:
                run = 0
        persist.append(long / len(s) if s else 0.0)
    return {**agg, 'total': tot,
            'adopt_all': agg['green'] / tot if tot else 0.0,
            'green_share': agg['green'] / (agg['green'] + agg['regular'])
                           if (agg['green'] + agg['regular']) else 0.0,
            'purchase_rate': (agg['green'] + agg['regular']) / tot if tot else 0.0,
            'rejected_rate': agg['rejected'] / tot if tot else 0.0,
            'conditional_green_rate': agg['green'] / aff_grn if aff_grn else 0.0,
            'conditional_purchase_rate': (agg['green'] + agg['regular']) / aff_reg if aff_reg else 0.0,
            'afford_green_ratio': aff_grn / max(tot, 1),
            'afford_regular_ratio': aff_reg / max(tot, 1),
            'ep_final_share': float(np.mean(ep_final)),
            'persistence': float(np.mean(persist)),
            'step_traces': traces,
            'episode_shares': ep_shares}


def compute_ei(rates, iw=10, fw=50, pw=10, step=2, floor=0.02):
    r = np.array(rates, dtype=float)
    if len(r) < max(iw, fw):
        return None
    R_i = float(np.mean(r[:iw]))
    R_f = float(np.mean(r[-fw:]))
    std = float(np.std(r[-fw:], ddof=1)) if len(r[-fw:]) > 1 else 0.0
    S = 1.0 / (max(std, floor) + 0.01)
    vv = [float(np.var(r[s:s + pw], ddof=1)) for s in range(0, len(r) - pw + 1, step)
          if len(r[s:s + pw]) == pw]
    V = max(vv) if vv else 0.0
    EI = S * V / (1.0 + abs(R_i - R_f))
    return {'R_init': R_i, 'R_final': R_f, 'std': std, 'S': S, 'V_peak': V, 'EI': EI}


def format_ei_worked(rates, label):
    d = compute_ei(rates)
    if d is None:
        return ""
    return "\n".join([
        f"=== EI worked example ({label}) ===",
        f"  len = {len(rates)}",
        f"  R_init = mean(first 10) = {d['R_init']:.6f}",
        f"  R_final = mean(last 50) = {d['R_final']:.6f}",
        f"  std_final = {d['std']:.6f}",
        f"  S = 1/(max(std,0.02)+0.01) = {d['S']:.6f}",
        f"  V_peak = max window variance = {d['V_peak']:.8f}",
        f"  EI = S*V_peak/(1+|R_init-R_final|) = {d['EI']:.6f}",
        f"  注：EI 无界，论文中 ×100 仅为缩放，不作概率解读",
        ""])