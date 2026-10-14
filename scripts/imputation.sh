#!/bin/bash
# Run imputation evaluation on all datasets
cd "$(dirname "$0")/.."
python imputation_all.py "$@"
