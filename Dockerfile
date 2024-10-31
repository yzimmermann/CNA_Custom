FROM nvcr.io/nvidia/pytorch:23.04-py3
WORKDIR /cna_modules 

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update

RUN echo "Upgrade pip" 
RUN python -m pip install -U pip
# RUN echo "Installing LaTeX"
# RUN apt-get update && apt-get install -y \
#     texlive-latex-base \
#     texlive-fonts-recommended \
#     texlive-fonts-extra \
#     texlive-latex-extra
RUN echo "Install torch==2.2.0 & dependencies..."
RUN python -m pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 torchtext==0.17.0 --index-url https://download.pytorch.org/whl/cu121
RUN echo "Installing Activation-Functions"
RUN git clone https://github.com/k4ntz/activation-functions.git
WORKDIR /cna_modules/activation-functions/
# RUN cd /bag_of_rationals/activation-functions/
RUN python -m pip install airspeed==0.5.17
RUN echo "Install dependencies..."
RUN python -m pip install -r requirements.txt
RUN echo "install the library itself..."
RUN python -m pip install -e .
# RUN cd /bag_of_rationals
WORKDIR /bag_of_rationals
RUN echo "Installing package dependencies..."
RUN python -m pip install setuptools==65.6.3 faiss_gpu==1.7.2 fast_pytorch_kmeans==0.1.9 matplotlib==3.7.1 networkx==3.1 numpy pytorch_lightning==2.0.2 SciencePlots==2.1.0 scipy==1.9.1 seaborn==0.12.2 torchviz==0.0.2 scikit-learn==1.3.2 tensorboard-plugin-customizable-plots dill ogb==1.3.6
# RUN python -m pip install numpy pytorch_lightning==2.0.2 SciencePlots==2.1.0 scipy==1.9.1 seaborn==0.12.2
# RUN python -m pip install torchviz==0.0.2 scikit-learn==1.3.2 tensorboard-plugin-customizable-plots dill
# RUN python -m pip install ogb==1.3.6
RUN echo "Install torch geometric and its dependencies"
RUN python -m pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.2.2+cu121.html
RUN python -m pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-2.2.2+cu121.html
RUN python -m pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-2.2.2+cu121.html
RUN python -m pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-2.2.2+cu121.html
RUN python -m pip install pyg_lib -f https://pytorch-geometric.com/whl/torch-2.2.2+cu121.html
RUN python -m pip install torch-geometric==2.3.0
WORKDIR /cna_modules
RUN echo "All installations completed successfully!"    
