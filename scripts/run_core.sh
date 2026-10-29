#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
echo "Running core experiments (5 conditions x 5 seeds x 200 episodes)"
RUN_MODE=core python train.py