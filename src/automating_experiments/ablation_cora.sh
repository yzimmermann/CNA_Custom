#!/bin/bash

# Get the project root directory (src)
PROJECT_ROOT=$(dirname $(dirname $(realpath $0)))

python $PROJECT_ROOT/scripts/execute_experiments.py --config mod_cora_4_gcnconv  --num_seed 20                                      #1
python $PROJECT_ROOT/scripts/execute_experiments.py --config mod_cora_4_gcnconv_cluster --num_seed 20                               #2
python $PROJECT_ROOT/scripts/execute_experiments.py --config mod_cora_4_gcnconv_cluster_normalize --num_seed 20                     #3
python $PROJECT_ROOT/scripts/execute_experiments.py --config mod_cora_4_gcnconv_activate_cluster --num_seed 20                      #4
python $PROJECT_ROOT/scripts/execute_experiments.py --config mod_cora_4_gcnconv_activate --num_seed 20                              #5
python $PROJECT_ROOT/scripts/execute_experiments.py --config mod_cora_4_gcnconv_normalize --num_seed 20                             #6
python $PROJECT_ROOT/scripts/execute_experiments.py --config mod_cora_4_gcnconv_activate_normalize --num_seed 20                    #7
python $PROJECT_ROOT/scripts/execute_experiments.py --config mod_cora_4_gcnconv_activate_cluster_normalize --num_seed 20            #8
python $PROJECT_ROOT/scripts/execute_experiments.py --config mod_cora_4_gatconv_activate_cluster_normalize --num_seed 20            #9
python $PROJECT_ROOT/scripts/execute_experiments.py --config mod_cora_4_transformerconv_activate_cluster_normalize --num_seed 20    #10
python $PROJECT_ROOT/scripts/execute_experiments.py --config mod_cora_4_sageconv_activate_cluster_normalize --num_seed 20           #11
python $PROJECT_ROOT/scripts/execute_experiments.py --config mod_cora_4_gatconv --num_seed 20                                       #12
python $PROJECT_ROOT/scripts/execute_experiments.py --config mod_cora_4_transformerconv --num_seed 20                               #13
python $PROJECT_ROOT/scripts/execute_experiments.py --config mod_cora_4_sageconv --num_seed 20                                      #14
