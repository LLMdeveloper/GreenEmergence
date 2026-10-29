#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
echo "Running rule-based baselines"
RUN_MODE=rule_based python train.py