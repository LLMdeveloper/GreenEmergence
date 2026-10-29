#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
echo "Running all experiments"
RUN_MODE=all python train.py