# -*- coding: utf-8 -*-
"""实验配置。"""

BASE = {
    'num_agents': 30, 'days': 7, 'steps_per_day': 8,
    'network': 'ws', 'ws_k': 4, 'ws_p': 0.1,
    'subsidy': 0.0, 'carbon_tax': 0.0,
    'normal_price': 10.0, 'green_base_price': 12.0,
    'income_mean': 18.0, 'income_std': 3.0,
    'income_replenish': 5, 'replenish_amount': 1.5,
    'need_alpha': [2.0, 2.0, 2.0, 1.0, 0.8],
    'initial_need_mean': 5.0, 'initial_need_std': 0.8,
    'threshold_mean': 0.35, 'threshold_std': 0.12, 'threshold_temperature': 0.08,
    'w_self': 0.5, 'w_social': 0.4, 'w_norm': 0.1,
    'norm_penalty': 0.1,
    'green_base_reward': 0.1,
    'price_cost_scale': 0.05,
    'regular_need': {'physiological': 1.0, 'safety': 0.2},
    'green_need': {'safety': 1.2, 'social': 0.6},
    'recent_actions_len': 5,
    'initial_green_seeds': 8, 'seed_mode': 'cluster',
    'seed_forced_steps': 5,
    'social_reward_enabled': True,
    'rejected_penalty': -0.1,
    'idle_penalty': 0.0,
    'weight_override': None,
}


def build_conditions():
    C = {k: {} for k in ['core', 'subsidy_scan', 'carbontax_scan',
                          'single_dim', 'price_neutral', 'rule_based']}

    C['core']['Baseline'] = {**BASE}
    C['core']['NoSeed'] = {**BASE, 'initial_green_seeds': 0}
    C['core']['NoSocial_030'] = {**BASE, 'w_social': 0.0, 'w_self': 0.5, 'w_norm': 0.1}
    C['core']['Full_060'] = {**BASE, 'w_social': 0.24, 'w_self': 0.30, 'w_norm': 0.06}
    C['core']['NoNetwork'] = {**BASE, 'network': 'none'}

    for s in [0.0, 0.10, 0.20, 0.30]:
        C['subsidy_scan'][f'Subsidy_{int(round(s*100)):02d}'] = {**BASE, 'subsidy': s}

    for t in [0.10, 0.20, 0.30]:
        C['carbontax_scan'][f'CarbonTax_{int(round(t*100)):02d}'] = {**BASE, 'subsidy': 0.0, 'carbon_tax': t}

    C['single_dim']['Safety_High_1D'] = {**BASE, 'weight_override': ('safety', 0.50)}
    C['single_dim']['Safety_Low_1D'] = {**BASE, 'weight_override': ('safety', 0.10)}

    C['price_neutral']['PriceNeutral_0Sub'] = {**BASE, 'green_base_price': 10.0, 'subsidy': 0.0}
    C['price_neutral']['PriceNeutral_20Sub'] = {**BASE, 'green_base_price': 10.0, 'subsidy': 0.20}

    for r in ['always_green', 'always_regular', 'random', 'threshold']:
        C['rule_based'][f'Rule_{r}'] = {**BASE, 'rule': r}

    return C


GROUP_SEEDS = {
    'core':           [42, 43, 44, 45, 46],
    'subsidy_scan':   [42, 43, 44],
    'carbontax_scan': [42, 43, 44],
    'single_dim':     [42, 43, 44],
    'price_neutral':  [42, 43, 44],
}