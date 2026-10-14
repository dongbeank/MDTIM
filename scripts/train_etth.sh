#!/bin/bash
# Train MDTM on ETTh dataset
cd "$(dirname "$0")/.."
python main.py --data etth --mode train
