#!/usr/bin/env bash
# Episodes, frames and lengths of the raw recordings and of the converted dataset.
set -euo pipefail
cd "$(dirname "$0")" && source ./config.sh
python dataset_summary.py "$RAW_ROOT" "$DATASET_ROOT"
