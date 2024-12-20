#!/bin/bash

# Get the project root directory (src)
PROJECT_ROOT=$(dirname $(dirname $(realpath $0)))
# 
# CORA
#
#   CNA
# 
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/CNA/TransformerConv/cora_1_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/CNA/TransformerConv/cora_2_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/CNA/TransformerConv/cora_4_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/CNA/TransformerConv/cora_8_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/CNA/TransformerConv/cora_16_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/CNA/TransformerConv/cora_32_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/CNA/TransformerConv/cora_64_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/CNA/TransformerConv/cora_96_transformerconv --num_seeds 5
#
#   ReLU
#
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/ReLU/TransformerConv/cora_1_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/ReLU/TransformerConv/cora_2_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/ReLU/TransformerConv/cora_4_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/ReLU/TransformerConv/cora_8_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/ReLU/TransformerConv/cora_16_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/ReLU/TransformerConv/cora_32_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/ReLU/TransformerConv/cora_64_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/ReLU/TransformerConv/cora_96_transformerconv --num_seeds 5
#
#   Linear
#
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/Linear/TransformerConv/cora_1_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/Linear/TransformerConv/cora_2_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/Linear/TransformerConv/cora_4_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/Linear/TransformerConv/cora_8_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/Linear/TransformerConv/cora_16_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/Linear/TransformerConv/cora_32_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/Linear/TransformerConv/cora_64_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/Cora/Linear/TransformerConv/cora_96_transformerconv --num_seeds 5
    