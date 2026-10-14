#!/bin/bash
# Train MDTM on Weather dataset
cd "$(dirname "$0")/.."
python main.py --data weather --mode train
