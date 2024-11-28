#!/bin/bash

# Get the project root directory (src)
PROJECT_ROOT=$(dirname $(dirname $(realpath $0)))

python $PROJECT_ROOT/scripts/execute_experiments.py --config ablation_study_ogbn_arxiv/ogbn_arxiv_gcnconv --num_seeds 5      # 1
python $PROJECT_ROOT/scripts/execute_experiments.py --config ablation_study_ogbn_arxiv/ogbn_arxiv_c_gcnconv --num_seeds 5    # 2
python $PROJECT_ROOT/scripts/execute_experiments.py --config ablation_study_ogbn_arxiv/ogbn_arxiv_cn_gcnconv --num_seeds 5   # 3
python $PROJECT_ROOT/scripts/execute_experiments.py --config ablation_study_ogbn_arxiv/ogbn_arxiv_ca_gcnconv --num_seeds 5   # 4
python $PROJECT_ROOT/scripts/execute_experiments.py --config ablation_study_ogbn_arxiv/ogbn_arxiv_a_gcnconv --num_seeds 5    # 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config ablation_study_ogbn_arxiv/ogbn_arxiv_n_gcnconv --num_seeds 5    # 6
python $PROJECT_ROOT/scripts/execute_experiments.py --config ablation_study_ogbn_arxiv/ogbn_arxiv_na_gcnconv --num_seeds 5   # 7
python $PROJECT_ROOT/scripts/execute_experiments.py --config ablation_study_ogbn_arxiv/ogbn_arxiv_cna_gcnconv --num_seeds 5  # 8
