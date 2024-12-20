#!/bin/bash

# Get the project root directory (src)
PROJECT_ROOT=$(dirname $(dirname $(realpath $0)))
# 
# CORA
#
#   CNA
# 
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/CNA/SAGEConv/cora_1_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/CNA/SAGEConv/cora_2_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/CNA/SAGEConv/cora_4_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/CNA/SAGEConv/cora_8_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/CNA/SAGEConv/cora_16_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/CNA/SAGEConv/cora_32_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/CNA/SAGEConv/cora_64_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/CNA/SAGEConv/cora_96_sageconv --num_seeds 5
#
#   ReLU
#
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/ReLU/SAGEConv/cora_1_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/ReLU/SAGEConv/cora_2_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/ReLU/SAGEConv/cora_4_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/ReLU/SAGEConv/cora_8_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/ReLU/SAGEConv/cora_16_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/ReLU/SAGEConv/cora_32_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/ReLU/SAGEConv/cora_64_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/ReLU/SAGEConv/cora_96_sageconv --num_seeds 5
#
#   Linear
#
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/Linear/SAGEConv/cora_1_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/Linear/SAGEConv/cora_2_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/Linear/SAGEConv/cora_4_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/Linear/SAGEConv/cora_8_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/Linear/SAGEConv/cora_16_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/Linear/SAGEConv/cora_32_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/Linear/SAGEConv/cora_64_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/Linear/SAGEConv/cora_96_sageconv --num_seeds 5
