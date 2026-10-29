# GreenEmergence

This repository contains the GreenTown environment, the IPPO training pipeline, all experimental conditions, the rule-based baselines, and the analysis scripts used to generate the tables and figures in the paper.

---

## Overview

GreenTown is an abstract consumption environment in which 30 agents with heterogeneous need weights interact on a Watts–Strogatz small-world network. Each agent observes only its direct neighbors' recent purchase history and learns consumption strategies through Independent Proximal Policy Optimization (IPPO).

The environment operationalizes Maslow's five needs (physiological, safety, social, esteem, self-actualization) as dynamic need states that enter the reward through weighted need-satisfaction increments. The repository supports four research questions:

1. Does a higher safety weight increase green adoption relative to a lower safety weight?
2. Is social information (local network interaction and social-norm reward) necessary for sustaining high adoption?
3. How much of the apparent social-norm effect is attributable to reward scale?
4. Do subsidies and carbon taxes operate through different margins?

---

## Requirements

- Python 3.9+
- See `requirements.txt`

| Package    | Version |
| ---------- | ------- |
| numpy      | >= 1.24 |
| pandas     | >= 2.0  |
| scipy      | >= 1.10 |
| torch      | >= 2.0  |
| networkx   | >= 3.0  |
| matplotlib | >= 3.7  |

---

## Project Structure

```
GreenEmergence/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── train.py                     # main entry point
├── config.py                    # all experimental conditions and seeds
├── src/
│   ├── __init__.py
│   ├── utils.py                 # seed, logger, weight override
│   ├── agent.py                 # Agent definition
│   ├── environment.py           # GreenTown environment
│   ├── algorithm.py             # IPPO, actor/critic, GAE
│   ├── evaluation.py            # frozen evaluation, metrics, EI
│   ├── rules.py                 # rule-based baselines
│   ├── experiments.py           # run one experiment / rule baseline
│   └── summarize.py             # aggregate results
├── analysis/
│   ├── print_ch5_tables.py      # generate Tables 1-6
│   └── plot_ch5_figures_v2.py   # generate Figures 1-5
└── scripts/
    ├── run_core.sh
    ├── run_extended.sh
    ├── run_rule_based.sh
    └── run_all.sh
```

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/LLMdeveloper/GreenEmergence.git
cd GreenEmergence
pip install -r requirements.txt
```

Optional: create a virtual environment first.

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

---

## Reproducing the Results

All experiments use the seeds listed in `config.py`:

| Group          | Seeds                                     |
| -------------- | ----------------------------------------- |
| core           | 42, 43, 44, 45, 46                        |
| subsidy_scan   | 42, 43, 44                                |
| carbontax_scan | 42, 43, 44                                |
| single_dim     | 42, 43, 44                                |
| price_neutral  | 42, 43, 44                                |
| rule_based     | 42 (fixed config, 10 evaluation episodes) |

`train.py` reads the running mode from the environment variable `RUN_MODE`. The default is `core`.

### Quick self-test

```bash
RUN_MODE=quick_test python train.py
```

Runs one seed for 50 episodes to verify the environment and training loop. Use this before running the full suite.

### Core experiments

```bash
RUN_MODE=core python train.py
# or
bash scripts/run_core.sh
```

Runs Baseline, NoNetwork, NoSocial, FullSocial, and NoSeed for 200 episodes each (5 seeds per condition).

### Extended experiments

```bash
RUN_MODE=extended python train.py
# or
bash scripts/run_extended.sh
```

Runs:

- Subsidy scan (0%, 10%, 20%, 30%)
- Carbon-tax scan (10%, 20%, 30%)
- Single-dimension safety manipulation (high = 0.5, low = 0.1)
- Price-neutrality control (0% subsidy, 20% subsidy)

### Rule-based baselines

```bash
RUN_MODE=rule_based python train.py
# or
bash scripts/run_rule_based.sh
```

Runs four fixed rules: `always_green`, `always_regular`, `random`, `threshold`.

### All experiments

```bash
RUN_MODE=all python train.py
# or
bash scripts/run_all.sh
```

Runs core, extended, and rule-based groups in sequence.

### Summarize only

```bash
RUN_MODE=summary_only python train.py
```

Aggregates existing results without retraining.

---

## Outputs

All results are written to `runs_B_v3/`:

```
runs_B_v3/
├── core/
│   ├── Baseline/seed42/records.json
│   ├── Baseline/seed42/train.log
│   ├── NoNetwork/seed42/...
│   └── ...
├── subsidy_scan/
├── carbontax_scan/
├── single_dim/
├── price_neutral/
├── rule_based/
│   └── Rule_always_green/result.json
└── summary/
    ├── all_episodes.csv
    ├── seed_level.csv
    ├── group_summary.csv
    ├── stats.csv
    ├── emergence_verdict.csv
    ├── subsidy_scan.csv
    ├── carbontax_scan.csv
    ├── rule_based.csv
    └── ei_worked_example.txt
```

Each `records.json` contains:

- `records`: per-episode metrics (evaluation every 10 episodes)
- `train_shares`: per-episode green share during training
- `snapshots`: mean need states, mean weights, mean income
- `frozen`: 10-episode frozen evaluation metrics
- `frozen_traces`: step-level traces for the 30% subsidy fragmentation analysis
- `config`, `group`, `exp`, `seed`

---

## Generating Tables and Figures

After running the experiments:

```bash
python analysis/print_ch5_tables.py
python analysis/plot_ch5_figures_v2.py
```

- `print_ch5_tables.py` prints Tables 1–6 to the console.
- `plot_ch5_figures_v2.py` saves Figures 1–5 to `figures_ch5/` as PDF and SVG.

Both scripts change to the repository root automatically, so they can be run from any directory.

---

## Key Metrics

| Metric                       | Definition                                                   |
| ---------------------------- | ------------------------------------------------------------ |
| **R_overall**                | Mean green share over all training episodes                  |
| **Frozen green share**       | Mean green share over 10 post-training evaluation episodes with exploration disabled |
| **Frozen persistence**       | Proportion of the longest consecutive run of steps with green share above 0.6, averaged over evaluation episodes |
| **All-opportunity adoption** | Green purchases divided by all decision opportunities        |
| **Purchase rate**            | Purchases divided by all decision opportunities              |
| **Rejection rate**           | Budget-violating intentions divided by all decision opportunities |
| **Affordability ratio**      | Share of decision opportunities where the product is affordable |

`R_overall` reflects the learning trajectory; `frozen green share` and `frozen persistence` characterize the steady-state behavior of the trained policy.

---

## Experimental Conditions

| Group          | Condition           | Description                                        |
| -------------- | ------------------- | -------------------------------------------------- |
| core           | Baseline            | Full model                                         |
| core           | NoNetwork           | Local social observation removed                   |
| core           | NoSocial            | Social-norm reward disabled                        |
| core           | FullSocial          | Equal-total-magnitude control for NoSocial         |
| core           | NoSeed              | Initial green seeds set to zero                    |
| subsidy_scan   | Subsidy_00/10/20/30 | Subsidy rate 0% / 10% / 20% / 30%                  |
| carbontax_scan | CarbonTax_10/20/30  | Carbon tax 10% / 20% / 30%                         |
| single_dim     | Safety_High_1D      | Safety weight fixed at 0.5, other four rescaled    |
| single_dim     | Safety_Low_1D       | Safety weight fixed at 0.1, other four rescaled    |
| price_neutral  | PriceNeutral_0Sub   | Green base price = regular price, 0% subsidy       |
| price_neutral  | PriceNeutral_20Sub  | Green base price = regular price, 20% subsidy      |
| rule_based     | always_green        | Buy green whenever affordable                      |
| rule_based     | always_regular      | Buy regular whenever affordable                    |
| rule_based     | random              | Uniform over {0, 1, 2}                             |
| rule_based     | threshold           | Buy green when local green ratio exceeds threshold |

---

## Reproducibility Notes

- **Fixed seeds.** All seeds are specified in `config.py` and used throughout training, evaluation, and the rule-based baselines. Re-running `train.py` with the same `RUN_MODE` reproduces the same results.
- **Frozen evaluation.** Post-training metrics are computed on 10 evaluation episodes with exploration disabled, so they reflect the trained policy rather than the evolving training trajectory.
- **Seed-level statistics.** All primary tests are conducted at the seed level. No test treats the 200 training episodes as independent observations.
- **Rule-based baselines** are evaluated under the same frozen protocol as the trained policies, so their metrics are directly comparable.
- **Unit-testable need updates.** The five-dimensional need increments in `src/environment.py` can be verified against `src/agent.py` and `config.py`; the values are documented in the paper's Section 3.3.1 and Table A.1.

