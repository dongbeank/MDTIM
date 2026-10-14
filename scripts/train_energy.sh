#!/bin/bash
# Train MDTM on Energy dataset
cd "$(dirname "$0")/.."
python main.py --data energy --mode train
