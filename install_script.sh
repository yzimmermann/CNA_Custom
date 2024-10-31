#!/bin/bash

# Navigate back to projects folder
cd ..

conda activate base

# Create conda environment and activate
conda create -n cluster-normalize-activate python=3.8
conda activate cluster-normalize-activate

# Clone activation-functions project
git clone https://github.com/k4ntz/activation-functions.git
cd activation-functions

# Install airspeed
pip3 install airspeed==0.5.17

# Install pytorch
pip3 install torch torchvision torchaudio

# Install requirements
pip3 install -r requirements.txt --user

# Note about CUDA version
echo "if multi CUDA versions are installed then make sure to export the used one"
echo "export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:/usr/local/cuda-12.2/lib64"
echo "check \$LD_LIBRARY_PATH to verify"

# Install activation function
pip3 install -e .

# Navigate back to projects folder
cd ..

# bag_of_rationals project
cd bag_of_rationals

# Install dependencies
pip3 install setuptools==65.6.3 faiss_gpu==1.7.2 fast_pytorch_kmeans==0.1.9 matplotlib==3.7.1 networkx==3.1
pip3 install numpy pytorch_lightning==2.0.2 SciencePlots==2.1.0 scipy==1.9.1 seaborn==0.12.2
pip3 install torchviz==0.0.2 scikit-learn==1.1.2 tensorboard-plugin-customizable-plots dill
pip3 install ogb==1.3.6


# Install torch geometric
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.2.2+cu121.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-2.2.2+cu121.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-2.2.2+cu121.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-2.2.2+cu121.html
pip install torch-geometric==2.3.0

echo "All installations completed successfully."
