#!/bin/bash

# Run training
python training/train.py --data_path "./data/long_context_dataset.json"

# Run evaluation
python evaluation/evaluate.py --data_path "./data/long_context_dataset.json"
