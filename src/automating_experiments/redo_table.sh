#!/bin/bash

# Get the project root directory (src)
PROJECT_ROOT=$(dirname $(dirname $(realpath $0)))

python $PROJECT_ROOT/scripts/execute_experiments.py --config chameleon_2_dirgcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config citeseer_2_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config computers_2_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config cora_4_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config corafull_2_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config dblp_4_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config photo_4_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config pubmed_2_transformerconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config squirrel_2_dirgcnconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config texas_2_sageconv --num_seeds 5
python $PROJECT_ROOT/scripts/execute_experiments.py --config wisconsin_2_transformerconv --num_seeds 5
