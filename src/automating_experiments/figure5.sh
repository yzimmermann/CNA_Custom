#!/bin/bash

# Get the project root directory (src)
PROJECT_ROOT=$(dirname $(dirname $(realpath $0)))
# Baselines for the direct comparissons
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_5/ogbn-arxiv_baseline_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_5/ogbn-arxiv_baseline_gcnconv --num_seeds 5
# For performance comparssion CN vs CNA
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_5/ogbn-arxiv_cn_gcnconv --num_seeds 5
# CNA
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_5/ogbn-arxiv_cna_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_5/ogbn-arxiv_cna_sageconv --num_seeds 5
