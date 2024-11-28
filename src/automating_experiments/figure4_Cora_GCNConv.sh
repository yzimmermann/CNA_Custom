#!/bin/bash

# Get the project root directory (src)
PROJECT_ROOT=$(dirname $(dirname $(realpath $0)))
# 
# CORA
#
#   CNA
# 
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/CNA/GCNConv/cora_1_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/CNA/GCNConv/cora_2_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/CNA/GCNConv/cora_4_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/CNA/GCNConv/cora_8_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/CNA/GCNConv/cora_16_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/CNA/GCNConv/cora_32_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/CNA/GCNConv/cora_64_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/CNA/GCNConv/cora_96_gcnconv --num_seeds 5
#
#   ReLU
#
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/ReLU/GCNConv/cora_1_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/ReLU/GCNConv/cora_2_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/ReLU/GCNConv/cora_4_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/ReLU/GCNConv/cora_8_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/ReLU/GCNConv/cora_16_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/ReLU/GCNConv/cora_32_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/ReLU/GCNConv/cora_64_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/ReLU/GCNConv/cora_96_gcnconv --num_seeds 5
#
#   Linear
#
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/Linear/GCNConv/cora_1_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/Linear/GCNConv/cora_2_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/Linear/GCNConv/cora_4_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/Linear/GCNConv/cora_8_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/Linear/GCNConv/cora_16_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/Linear/GCNConv/cora_32_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/Linear/GCNConv/cora_64_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/Linear/GCNConv/cora_96_gcnconv --num_seeds 5
