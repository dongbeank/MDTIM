#!/bin/bash
# Train MDTM on Sine dataset
cd "$(dirname "$0")/.."
python main.py --data sine --mode train
