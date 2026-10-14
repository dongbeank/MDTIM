#!/bin/bash
# Train MDTM on all datasets
cd "$(dirname "$0")/.."

echo "===== Training ETTh ====="
python main.py --data etth --mode train

echo "===== Training Energy ====="
python main.py --data energy --mode train

echo "===== Training Sine ====="
python main.py --data sine --mode train

echo "===== Training Weather ====="
python main.py --data weather --mode train

echo "===== All training complete ====="
