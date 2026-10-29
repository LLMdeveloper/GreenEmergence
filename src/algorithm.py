# -*- coding: utf-8 -*-
"""IPPO 算法与网络。"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class PolicyNet(nn.Module):
    def __init__(self, od, ad=3, hidden=(128, 64)):
        super().__init__()
        layers, prev = [], od
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, ad), nn.Softmax(dim=-1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ValueNet(nn.Module):
    def __init__(self, od, hidden=(128, 64)):
        super().__init__()
        layers, prev = [], od
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def compute_gae(rs, vs, ds, gamma=0.99, lam=0.95):
    adv = np.zeros_like(rs)
    last = 0.0
    for t in reversed(range(len(rs))):
        nv = vs[t + 1] if t < len(rs) - 1 else 0.0
        nt = 1.0 - ds[t]
        delta = rs[t] + gamma * nv * nt - vs[t]
        last = delta + gamma * lam * nt * last
        adv[t] = last
    return adv, adv + np.array(vs)


class IPPO:
    def __init__(self, N, od, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2,
                 epochs=4, bs=128, device='cpu'):
        self.device = torch.device(device)
        self.N = N
        self.ad = 3
        self.policies = [PolicyNet(od, self.ad).to(self.device) for _ in range(N)]
        self.values = [ValueNet(od).to(self.device) for _ in range(N)]
        self.p_optims = [optim.Adam(p.parameters(), lr=lr) for p in self.policies]
        self.v_optims = [optim.Adam(v.parameters(), lr=lr) for v in self.values]
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.bs = epochs, bs

    def collect(self, env):
        N = self.N
        ob = [[] for _ in range(N)]
        ac = [[] for _ in range(N)]
        rw = [[] for _ in range(N)]
        dn = [[] for _ in range(N)]
        obs = env.reset()
        for _ in range(env.max_steps):
            for i in range(N):
                ob[i].append(obs[i])
            actions = []
            for i, p in enumerate(self.policies):
                with torch.no_grad():
                    pr = p(torch.FloatTensor(obs[i]).unsqueeze(0).to(self.device)).squeeze(0).cpu().numpy()
                actions.append(int(np.random.choice(self.ad, p=pr)))
                ac[i].append(actions[-1])
            n_obs, rs, done, _ = env.step(actions)
            for i in range(N):
                rw[i].append(rs[i])
                dn[i].append(done)
            obs = n_obs
            if done:
                break
        batch = []
        for i in range(N):
            vs = [self.values[i](torch.FloatTensor(ob[i][t]).unsqueeze(0).to(self.device)).item()
                  for t in range(len(ob[i]))]
            a, r = compute_gae(np.array(rw[i]), vs, np.array(dn[i]), self.gamma, self.lam)
            for t in range(len(ob[i])):
                batch.append({'id': i, 'obs': ob[i][t], 'act': ac[i][t],
                              'adv': float(a[t]), 'ret': float(r[t])})
        return batch

    def update(self, batch):
        for i in range(self.N):
            data = [d for d in batch if d['id'] == i]
            if not data:
                continue
            obs = torch.FloatTensor(np.stack([d['obs'] for d in data])).to(self.device)
            acts = torch.LongTensor([d['act'] for d in data]).to(self.device)
            adv = torch.FloatTensor([d['adv'] for d in data]).to(self.device)
            ret = torch.FloatTensor([d['ret'] for d in data]).to(self.device)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            p = self.policies[i]
            with torch.no_grad():
                op = p(obs)
                olp = torch.log(op + 1e-8).gather(1, acts.unsqueeze(1)).squeeze(1)
            for _ in range(self.epochs):
                idx = np.random.permutation(len(data))
                for s in range(0, len(data), self.bs):
                    b = idx[s:s + self.bs]
                    pr = p(obs[b])
                    lp = torch.log(pr + 1e-8).gather(1, acts[b].unsqueeze(1)).squeeze(1)
                    ent = -(pr * torch.log(pr + 1e-8)).sum(1).mean()
                    rat = torch.exp(lp - olp[b])
                    s1 = rat * adv[b]
                    s2 = torch.clamp(rat, 1 - self.clip, 1 + self.clip) * adv[b]
                    loss = -torch.min(s1, s2).mean() - 0.01 * ent
                    self.p_optims[i].zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(p.parameters(), 0.5)
                    self.p_optims[i].step()
            for _ in range(self.epochs):
                vl = F.mse_loss(self.values[i](obs), ret)
                self.v_optims[i].zero_grad()
                vl.backward()
                torch.nn.utils.clip_grad_norm_(self.values[i].parameters(), 0.5)
                self.v_optims[i].step()