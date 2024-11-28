#!/bin/bash

# Get the project root directory (src)
PROJECT_ROOT=$(dirname $(dirname $(realpath $0)))
# 
# CITESEER
#
#   CNA
# 
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/CNA/SAGEConv/citeseer_1_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/CNA/SAGEConv/citeseer_2_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/CNA/SAGEConv/citeseer_4_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/CNA/SAGEConv/citeseer_8_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/CNA/SAGEConv/citeseer_16_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/CNA/SAGEConv/citeseer_32_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/CNA/SAGEConv/citeseer_64_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/CNA/SAGEConv/citeseer_96_sageconv --num_seeds 5
#
#   ReLU
#
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/ReLU/SAGEConv/citeseer_1_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/ReLU/SAGEConv/citeseer_2_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/ReLU/SAGEConv/citeseer_4_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/ReLU/SAGEConv/citeseer_8_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/ReLU/SAGEConv/citeseer_16_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/ReLU/SAGEConv/citeseer_32_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/ReLU/SAGEConv/citeseer_64_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/ReLU/SAGEConv/citeseer_96_sageconv --num_seeds 5
#
#   Linear
#
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/Linear/SAGEConv/citeseer_1_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/Linear/SAGEConv/citeseer_2_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/Linear/SAGEConv/citeseer_4_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/Linear/SAGEConv/citeseer_8_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/Linear/SAGEConv/citeseer_16_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/Linear/SAGEConv/citeseer_32_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/Linear/SAGEConv/citeseer_64_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/Linear/SAGEConv/citeseer_96_sageconv --num_seeds 5
