#!/bin/bash

# Get the project root directory (src)
PROJECT_ROOT=$(dirname $(dirname $(realpath $0)))
# 
# CORA
#
#   CNA
# 
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/CNA/GATConv/cora_1_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/CNA/GATConv/cora_2_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/CNA/GATConv/cora_4_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/CNA/GATConv/cora_8_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/CNA/GATConv/cora_16_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/CNA/GATConv/cora_32_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/CNA/GATConv/cora_64_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/CNA/GATConv/cora_96_gatconv --num_seeds 5
#
#   ReLU
#
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/ReLU/GATConv/cora_1_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/ReLU/GATConv/cora_2_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/ReLU/GATConv/cora_4_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/ReLU/GATConv/cora_8_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/ReLU/GATConv/cora_16_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/ReLU/GATConv/cora_32_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/ReLU/GATConv/cora_64_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/ReLU/GATConv/cora_96_gatconv --num_seeds 5
#
#   Linear
#
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/Linear/GATConv/cora_1_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/Linear/GATConv/cora_2_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/Linear/GATConv/cora_4_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/Linear/GATConv/cora_8_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/Linear/GATConv/cora_16_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/Linear/GATConv/cora_32_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/Linear/GATConv/cora_64_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/Linear/GATConv/cora_96_gatconv --num_seeds 5
