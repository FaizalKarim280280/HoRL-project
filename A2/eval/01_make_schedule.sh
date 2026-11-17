#!/usr/bin/env bash
# Sample the 30 configurations + randomised schedule from eval_plan.yaml (fixed seed) and the home pose.
# Run once, after filling in eval_plan.yaml; then commit eval_plan.yaml, configs.csv, schedule.csv BEFORE
# the first counted trial. Refuses to overwrite an existing schedule.
set -euo pipefail
cd "$(dirname "$0")"
python make_schedule.py "$@"
