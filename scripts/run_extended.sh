#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
echo "Running extended experiments (subsidy/carbontax/single_dim/price_neutral)"
RUN_MODE=extended python train.py