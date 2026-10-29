# -*- coding: utf-8 -*-
"""GreenTown 环境。"""
import math

import numpy as np
import networkx as nx

from .agent import Agent
from .utils import NEED_KEYS, apply_weight_override


class Env:
    def __init__(self, config):
        self.config = config
        self.N = config['num_agents']
        self.steps_per_day = config['steps_per_day']
        self.max_steps = config['days'] * config['steps_per_day']
        self.normal_price = config['normal_price']
        self.green_price = config['green_base_price']
        self.subsidy = config['subsidy']
        self.carbon_tax = config['carbon_tax']
        self.eff_normal = self.normal_price * (1 + self.carbon_tax)
        self.eff_green = self.green_price * (1 - self.subsidy)
        self.need_alpha = config['need_alpha']
        self.theta_mean = config['threshold_mean']
        self.theta_std = config['threshold_std']
        self.tau = config['threshold_temperature']
        self.w_self = config['w_self']
        self.w_social = config['w_social']
        self.w_norm = config['w_norm']
        self.norm_pen = config['norm_penalty']
        self.green_base = config['green_base_reward']
        self.price_scale = config['price_cost_scale']
        self.reg_need = config['regular_need']
        self.grn_need = config['green_need']
        self.inc_mean = config['income_mean']
        self.inc_std = config['income_std']
        self.inc_rep = config['income_replenish']
        self.repl_amount = config['replenish_amount']
        self.recent_len = config['recent_actions_len']
        self.social_on = config.get('social_reward_enabled', True)
        self.rej_pen = config.get('rejected_penalty', -0.1)
        self.idle_pen = config.get('idle_penalty', 0.0)
        self.seed_forced_steps = config.get('seed_forced_steps', 5)

        self.G = self._build_network()
        self.agents = []
        self._init_agents()
        for i, a in enumerate(self.agents):
            a.neighbors = list(self.G.neighbors(i))
        self.current_step = 0
        self.seed_ids = []

    def _build_network(self):
        seed = self.config['seed']
        net = self.config['network']
        if net == 'ws':
            return nx.watts_strogatz_graph(self.N, self.config['ws_k'],
                                           self.config['ws_p'], seed=seed)
        if net == 'ba':
            return nx.barabasi_albert_graph(self.N, 2, seed=seed)
        if net == 'none':
            return nx.empty_graph(self.N)
        raise ValueError(net)

    def _init_agents(self):
        base_decay = {'physiological': 0.20, 'safety': 0.10, 'social': 0.15,
                      'esteem': 0.10, 'self_actualization': 0.05}
        rng = np.random.RandomState(self.config['seed'] + 1)
        override = self.config.get('weight_override')
        for i in range(self.N):
            w = rng.dirichlet(self.need_alpha)
            nw = {k: float(w[j]) for j, k in enumerate(NEED_KEYS)}
            nw = apply_weight_override(nw, override)
            ini = {k: float(np.clip(rng.normal(self.config['initial_need_mean'],
                                              self.config['initial_need_std']), 2.0, 8.0))
                   for k in NEED_KEYS}
            dec = {k: float(np.clip(rng.normal(base_decay[k], 0.02), 0.01, 0.5))
                   for k in NEED_KEYS}
            inc = max(10.0, float(rng.normal(self.inc_mean, self.inc_std)))
            th = float(np.clip(rng.normal(self.theta_mean, self.theta_std), 0.05, 0.95))
            self.agents.append(Agent(i, nw, ini, dec, inc, th,
                                     self.inc_mean, self.inc_std, self.recent_len))

    def _select_seed_agents(self, k, mode, rng):
        if k <= 0:
            return []
        if mode == 'cluster':
            center = int(rng.randint(0, self.N))
            cands = [center]
            for n in self.G.neighbors(center):
                if len(cands) >= k:
                    break
                cands.append(int(n))
            if len(cands) < k:
                pool = [i for i in range(self.N) if i not in cands]
                rng.shuffle(pool)
                cands += pool[:k - len(cands)]
            return cands
        return list(rng.choice(self.N, size=min(k, self.N), replace=False))

    def reset(self, rng=None):
        rng = rng if rng is not None else np.random
        self.current_step = 0
        for a in self.agents:
            a.reset(rng)
        k = self.config.get('initial_green_seeds', 0)
        self.seed_ids = self._select_seed_agents(k, self.config.get('seed_mode', 'cluster'), rng)
        for sid in self.seed_ids:
            self.agents[sid].is_seed = True
            for _ in range(self.recent_len):
                self.agents[sid].recent_actions.append(2)
            self.agents[sid].last_green = True
            self.agents[sid].income = max(self.agents[sid].income, self.eff_green * 5)
        return [self._get_obs(i) for i in range(self.N)]

    def _local_green_ratio(self, i):
        nb = self.agents[i].neighbors
        if not nb:
            return 0.0
        tot_g = tot_a = 0
        for n in nb:
            for a in self.agents[n].recent_actions:
                if a == 2:
                    tot_g += 1
                tot_a += 1
        return tot_g / tot_a if tot_a > 0 else 0.0

    def _local_majority(self, i):
        nb = self.agents[i].neighbors
        if not nb:
            return 0
        cnt = {0: 0, 1: 0, 2: 0}
        for n in nb:
            if self.agents[n].recent_actions:
                cnt[self.agents[n].recent_actions[-1]] += 1
        return max(cnt, key=cnt.get)

    def _exec(self, agent, action):
        if action == 0:
            return 0, 0.0, False
        if action == 1:
            if agent.income >= self.eff_normal:
                agent.income -= self.eff_normal
                old = agent.needs.copy()
                for k, v in self.reg_need.items():
                    agent.needs[k] = min(10.0, agent.needs[k] + v)
                gain = sum(agent.need_weights[k] * (agent.needs[k] - old[k]) for k in NEED_KEYS)
                cost = (self.eff_normal - self.normal_price) * self.price_scale
                return 1, gain - cost, False
            return 0, 0.0, True
        if action == 2:
            if agent.income >= self.eff_green:
                agent.income -= self.eff_green
                old = agent.needs.copy()
                for k, v in self.grn_need.items():
                    agent.needs[k] = min(10.0, agent.needs[k] + v)
                gain = sum(agent.need_weights[k] * (agent.needs[k] - old[k]) for k in NEED_KEYS)
                cost = (self.eff_green - self.normal_price) * self.price_scale
                return 2, gain - cost + self.green_base, False
            return 0, 0.0, True
        return 0, 0.0, False

    def step(self, actions):
        if self.inc_rep > 0 and self.current_step > 0 and self.current_step % self.inc_rep == 0:
            for a in self.agents:
                a.income += self.inc_mean * self.repl_amount

        rewards = np.zeros(self.N, dtype=np.float32)
        info = {'green': 0, 'regular': 0, 'no_purchase': 0, 'rejected': 0,
                'n_afford_regular': 0, 'n_afford_green': 0,
                'r_local_mean': 0.0, 'frac_above': 0.0, 'green_share_step': 0.0}
        r_local_list = []
        forced_ids = set(self.seed_ids) if self.current_step < self.seed_forced_steps else set()

        for i, a in enumerate(actions):
            agent = self.agents[i]
            if i in forced_ids:
                a = 2
            aff_reg = agent.income >= self.eff_normal
            aff_grn = agent.income >= self.eff_green
            result, self_r, rej = self._exec(agent, a)
            if aff_reg:
                info['n_afford_regular'] += 1
            if aff_grn:
                info['n_afford_green'] += 1
            if not rej:
                agent.recent_actions.append(result)
            if rej:
                info['rejected'] += 1
            elif result == 2:
                info['green'] += 1
            elif result == 1:
                info['regular'] += 1
            else:
                info['no_purchase'] += 1

            r_local = self._local_green_ratio(i)
            majority = self._local_majority(i)
            r_local_list.append(r_local)

            if self.social_on and result == 2:
                social_r = 0.5 * float(np.tanh((r_local - agent.threshold) / self.tau)) + 0.3
            else:
                social_r = 0.0

            norm_r = -self.norm_pen if (result in [1, 2] and majority in [1, 2] and result != majority) else 0.0
            rej_r = self.rej_pen if rej else 0.0
            idle_r = self.idle_pen if (result == 0 and not rej and aff_reg) else 0.0

            rewards[i] = (self.w_self * self_r + self.w_social * social_r
                          + self.w_norm * norm_r + rej_r + idle_r)
            agent.update_decay()

        self.current_step += 1
        info['r_local_mean'] = float(np.mean(r_local_list))
        info['frac_above'] = float(np.mean(
            [1.0 if r_local_list[i] > self.agents[i].threshold else 0.0 for i in range(self.N)]))
        buys = info['green'] + info['regular']
        info['green_share_step'] = info['green'] / buys if buys > 0 else 0.0
        done = self.current_step >= self.max_steps
        return [self._get_obs(i) for i in range(self.N)], rewards, done, info

    def _get_obs(self, i):
        a = self.agents[i]
        nv = [a.needs[k] for k in NEED_KEYS]
        r_local = self._local_green_ratio(i)
        return np.array(nv + [
            r_local, self.eff_normal, self.eff_green, a.income / 20.0,
            math.sin(2 * math.pi * (self.current_step % self.steps_per_day) / self.steps_per_day),
            math.cos(2 * math.pi * (self.current_step % self.steps_per_day) / self.steps_per_day),
            self.subsidy, self.carbon_tax,
        ], dtype=np.float32)

    def affordability(self):
        n = len(self.agents)
        return {
            'afford_normal': sum(1 for a in self.agents if a.income >= self.eff_normal) / n,
            'afford_green': sum(1 for a in self.agents if a.income >= self.eff_green) / n,
            'mean_income': float(np.mean([a.income for a in self.agents])),
            'normal_price': self.eff_normal,
            'green_price': self.eff_green,
        }