# -*- coding: utf-8 -*-
"""单次实验与规则基线运行。"""
import os
import json
import time

import numpy as np

from .utils import set_seed, Logger
from .environment import Env
from .algorithm import IPPO
from .evaluation import evaluate, compute_ei
from .rules import rule_action


def run_experiment(config, exp_name, group, seed, episodes, device, out_root, eval_every=10):
    exp_dir = os.path.join(out_root, group, exp_name, f'seed{seed}')
    os.makedirs(exp_dir, exist_ok=True)
    marker = os.path.join(exp_dir, 'done.json')
    if os.path.exists(marker):
        try:
            with open(marker) as f:
                if json.load(f).get('episodes') == episodes:
                    return
        except Exception:
            pass

    cfg = dict(config)
    cfg['seed'] = seed
    set_seed(seed)
    logger = Logger(os.path.join(exp_dir, 'train.log'))
    logger.log(f"=== {group}/{exp_name}/seed{seed} ep={episodes} ===")
    logger.log(f"config={json.dumps(cfg, ensure_ascii=False)}")

    env = Env(cfg)
    obs_dim = len(env._get_obs(0))
    trainer = IPPO(env.N, obs_dim, device=device)

    records = []
    train_shares = []
    snapshots = []
    t0 = time.time()
    for ep in range(1, episodes + 1):
        batch = trainer.collect(env)
        trainer.update(batch)
        r1 = evaluate(env, trainer.policies, device, n_eps=1)
        train_shares.append(r1['ep_final_share'])
        if ep % eval_every == 0 or ep == episodes:
            det = evaluate(env, trainer.policies, device, n_eps=5)
            ei = compute_ei(train_shares)
            rec = {'group': group, 'exp': exp_name, 'seed': seed, 'episode': ep,
                   'green': det['green'], 'regular': det['regular'],
                   'no_purchase': det['no_purchase'], 'rejected': det['rejected'],
                   'adopt_all': det['adopt_all'], 'green_share': det['green_share'],
                   'conditional_green_rate': det['conditional_green_rate'],
                   'purchase_rate': det['purchase_rate'], 'rejected_rate': det['rejected_rate'],
                   'afford_green_ratio': det['afford_green_ratio'],
                   'afford_regular_ratio': det['afford_regular_ratio'],
                   'ep_final_share': det['ep_final_share'], 'persistence': det['persistence']}
            if ei:
                rec.update({f'ei_{k}': v for k, v in ei.items()})
            records.append(rec)
            snapshots.append({
                'episode': ep,
                'mean_needs': {k: float(np.mean([a.needs[k] for a in env.agents]))
                               for k in ['physiological', 'safety', 'social', 'esteem', 'self_actualization']},
                'mean_weights': {k: float(np.mean([a.need_weights[k] for a in env.agents]))
                                 for k in ['physiological', 'safety', 'social', 'esteem', 'self_actualization']},
                'mean_income': float(np.mean([a.income for a in env.agents])),
                'green_share': det['green_share'],
                'afford_green_ratio': det['afford_green_ratio'],
            })
            logger.log(f"ep={ep:3d} g={det['green']:4d} r={det['regular']:4d} "
                       f"no={det['no_purchase']:4d} rej={det['rejected']:4d} "
                       f"share={det['green_share']:.3f} adopt={det['adopt_all']:.3f} "
                       f"cond_g={det['conditional_green_rate']:.3f} "
                       f"purch={det['purchase_rate']:.3f} rej_r={det['rejected_rate']:.3f} "
                       f"persist={det['persistence']:.3f}")
            if ep % 25 == 0 or ep == episodes:
                logger.log(f"[{group}/{exp_name}/s{seed}] ep={ep}/{episodes} "
                           f"share={det['green_share']:.3f} persist={det['persistence']:.3f} "
                           f"purch={det['purchase_rate']:.3f}", console=True)

    frozen = evaluate(env, trainer.policies, device, n_eps=10, record_steps=True)
    with open(os.path.join(exp_dir, 'records.json'), 'w') as f:
        json.dump({'records': records, 'train_shares': train_shares,
                   'snapshots': snapshots,
                   'frozen': {k: frozen[k] for k in ['adopt_all', 'green_share',
                                                     'conditional_green_rate',
                                                     'purchase_rate', 'rejected_rate',
                                                     'afford_green_ratio',
                                                     'persistence', 'ep_final_share']},
                   'frozen_traces': frozen['step_traces'],
                   'config': cfg, 'group': group, 'exp': exp_name, 'seed': seed}, f, indent=2)
    with open(marker, 'w') as f:
        json.dump({'episodes': episodes, 'finished_at': time.time()}, f)
    logger.log(f"DONE in {(time.time()-t0)/60:.1f} min")
    logger.close()


def run_rule_based(config, exp_name, group, rule, n_eps, device, out_root):
    exp_dir = os.path.join(out_root, group, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    logger = Logger(os.path.join(exp_dir, 'train.log'))
    set_seed(42)
    env = Env(config)
    agg = {'green': 0, 'regular': 0, 'no_purchase': 0, 'rejected': 0}
    ep_shares = []
    aff_reg = aff_grn = 0
    for ep in range(n_eps):
        rng = np.random.RandomState(seed=1000 + ep)
        env.reset(rng=rng)
        done = False
        steps = []
        while not done:
            actions = [rule_action(rule, env, i) for i in range(env.N)]
            _, _, done, info = env.step(actions)
            for k in agg:
                agg[k] += info[k]
            aff_reg += info['n_afford_regular']
            aff_grn += info['n_afford_green']
            steps.append(info['green_share_step'])
        ep_shares.append(steps)
    tot = agg['green'] + agg['regular'] + agg['no_purchase'] + agg['rejected']
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
    result = {
        'group': group, 'exp': exp_name, 'rule': rule,
        'green': agg['green'], 'regular': agg['regular'],
        'no_purchase': agg['no_purchase'], 'rejected': agg['rejected'],
        'adopt_all': agg['green'] / tot if tot else 0.0,
        'green_share': agg['green'] / (agg['green'] + agg['regular'])
                       if (agg['green'] + agg['regular']) else 0.0,
        'purchase_rate': (agg['green'] + agg['regular']) / tot if tot else 0.0,
        'rejected_rate': agg['rejected'] / tot if tot else 0.0,
        'conditional_green_rate': agg['green'] / aff_grn if aff_grn else 0.0,
        'afford_green_ratio': aff_grn / max(tot, 1),
        'persistence': float(np.mean(persist)),
    }
    with open(os.path.join(exp_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)
    logger.log(f"[rule_based/{exp_name}] share={result['green_share']:.3f} "
               f"purch={result['purchase_rate']:.3f} persist={result['persistence']:.3f}",
               console=True)
    logger.close()