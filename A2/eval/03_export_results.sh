#!/usr/bin/env bash
# Regenerate the Task 1 results folder (A2/results/task1): README with every reported result, tables, figures,
# manifest, plan and checkpoint configs, from plan/ and results/task1/logs/ (raw log + corrections.yaml).
set -euo pipefail
cd "$(dirname "$0")"
python export_results.py "$@"
