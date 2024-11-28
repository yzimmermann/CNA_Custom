#!/bin/bash

# Get the project root directory (src)
PROJECT_ROOT=$(dirname $(dirname $(realpath $0)))
#
# CITESEER
#
#   CNA
#
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/CNA/GATConv/citeseer_1_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/CNA/GATConv/citeseer_2_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/CNA/GATConv/citeseer_4_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/CNA/GATConv/citeseer_8_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/CNA/GATConv/citeseer_16_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/CNA/GATConv/citeseer_32_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/CNA/GATConv/citeseer_64_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/CNA/GATConv/citeseer_96_gatconv --num_seeds 5
#
#   ReLU
#
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/ReLU/GATConv/citeseer_1_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/ReLU/GATConv/citeseer_2_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/ReLU/GATConv/citeseer_4_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/ReLU/GATConv/citeseer_8_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/ReLU/GATConv/citeseer_16_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/ReLU/GATConv/citeseer_32_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/ReLU/GATConv/citeseer_64_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/ReLU/GATConv/citeseer_96_gatconv --num_seeds 5
#
#   Linear
#
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/Linear/GATConv/citeseer_1_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/Linear/GATConv/citeseer_2_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/Linear/GATConv/citeseer_4_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/Linear/GATConv/citeseer_8_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/Linear/GATConv/citeseer_16_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/Linear/GATConv/citeseer_32_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/Linear/GATConv/citeseer_64_gatconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/Linear/GATConv/citeseer_96_gatconv --num_seeds 5
