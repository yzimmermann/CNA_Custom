#!/bin/bash

# Get the project root directory (src)
PROJECT_ROOT=$(dirname $(dirname $(realpath $0)))
# 
# CITESEER
#
#   CNA
# 
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/CNA/TransformerConv/citeseer_1_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/CNA/TransformerConv/citeseer_2_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/CNA/TransformerConv/citeseer_4_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/CNA/TransformerConv/citeseer_8_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/CNA/TransformerConv/citeseer_16_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/CNA/TransformerConv/citeseer_32_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/CNA/TransformerConv/citeseer_64_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/CNA/TransformerConv/citeseer_96_transformerconv --num_seeds 5
#
#   ReLU
#
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/ReLU/TransformerConv/citeseer_1_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/ReLU/TransformerConv/citeseer_2_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/ReLU/TransformerConv/citeseer_4_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/ReLU/TransformerConv/citeseer_8_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/ReLU/TransformerConv/citeseer_16_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/ReLU/TransformerConv/citeseer_32_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/ReLU/TransformerConv/citeseer_64_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/ReLU/TransformerConv/citeseer_96_transformerconv --num_seeds 5
#
#   Linear
#
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/Linear/TransformerConv/citeseer_1_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/Linear/TransformerConv/citeseer_2_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/Linear/TransformerConv/citeseer_4_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/Linear/TransformerConv/citeseer_8_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/Linear/TransformerConv/citeseer_16_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/Linear/TransformerConv/citeseer_32_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/Linear/TransformerConv/citeseer_64_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config figure_4/CiteSeer/Linear/TransformerConv/citeseer_96_transformerconv --num_seeds 5
