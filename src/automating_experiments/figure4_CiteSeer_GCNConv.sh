#!/bin/bash

# Get the project root directory (src)
PROJECT_ROOT=$(dirname $(dirname $(realpath $0)))
# 
# CITESEER
#
#   CNA
# 
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/CNA/GCNConv/citeseer_1_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/CNA/GCNConv/citeseer_2_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/CNA/GCNConv/citeseer_4_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/CNA/GCNConv/citeseer_8_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/CNA/GCNConv/citeseer_16_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/CNA/GCNConv/citeseer_32_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/CNA/GCNConv/citeseer_64_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/CNA/GCNConv/citeseer_96_gcnconv --num_seeds 5
#
#   ReLU
#
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/ReLU/GCNConv/citeseer_1_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/ReLU/GCNConv/citeseer_2_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/ReLU/GCNConv/citeseer_4_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/ReLU/GCNConv/citeseer_8_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/ReLU/GCNConv/citeseer_16_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/ReLU/GCNConv/citeseer_32_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/ReLU/GCNConv/citeseer_64_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/ReLU/GCNConv/citeseer_96_gcnconv --num_seeds 5
#
#   Linear
#
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/Linear/GCNConv/citeseer_1_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/Linear/GCNConv/citeseer_2_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/Linear/GCNConv/citeseer_4_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/Linear/GCNConv/citeseer_8_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/Linear/GCNConv/citeseer_16_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/Linear/GCNConv/citeseer_32_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/Linear/GCNConv/citeseer_64_gcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/Linear/GCNConv/citeseer_96_gcnconv --num_seeds 5
