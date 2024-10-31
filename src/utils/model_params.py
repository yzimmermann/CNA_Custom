import os
import random
from enum import Enum

import torch


class ActivationType(Enum):
    RELU = "relu"  # relu
    RATMODEL = "RationalModel"  # our rational model
    RAT = "Rational"  # Lab rational model
    RPM = "RationalPowerMean"  # RPM rational model with Lab rationals
    RPMMODEL = "RationalPowerMeanModel"  # RPM rational model with our rationals
    RWAM = "RationalWeightedArithmeticMean"  # RWAM rational model with Lab rationals


class DistanceMetrics(Enum):
    EUCLIDEAN = "euclidean"  # Euclidean distance for KMean clustering
    COSINE = "cosine"  # Cosine distance for KMean clustering


class ReclusterOption(Enum):
    TH = "threshold"
    EV = "every"
    ITR = "iterations"


class TaskType(Enum):
    NODECLASSIFICATION = "node_classification"
    NODEREGRESSION = "node_regression"
    NODEPROPPRED = "node_prop_pred"


class LayerType(Enum):
    GCNCONV = "GCNConv"
    SAGECONV = "SAGEConv"
    GATCONV = "GATConv"
    TRANSFORMERCONV = "TransformerConv"
    LINEAR = "Linear"
    CHEBCONV = "ChebConv"
    CUGRAPHSAGECONV = "CuGraphSAGEConv"
    GRAPHCONV = "GraphConv"
    ARMACONV = "ARMAConv"
    SGCONV = "SGConv"
    MFCONV = "MFConv"
    SSGCONV = "SSGConv"
    LECONV = "LEConv"
    DIRSAGECONV = "DirSageConv"
    DIRGCNCONV = "DirGCNConv"


class Planetoid(Enum):
    CORA = "Planetoid+Cora"
    CITESEER = "Planetoid+CiteSeer"
    PUBMED = "Planetoid+PubMed"


class CitationFull(Enum):
    CORA = "CitationFull+Cora"
    CITESEER = "CitationFull+CiteSeer"
    PUBMED = "CitationFull+PubMed"
    CORA_ML = "CitationFull+Cora_ML"
    DBLP = "CitationFull+DBLP"


class Amazon(Enum):
    PHOTO = "Amazon+Photo"
    COMPUTERS = "Amazon+Computers"


class WikipediaNetwork(Enum):
    CHAMELEON = "WikipediaNetwork+Chameleon"
    SQUIRREL = "WikipediaNetwork+Squirrel"


class WebKB(Enum):
    CORNELL = "WebKB+Cornell"
    TEXAS = "WebKB+Texas"
    WISCONSIN = "WebKB+Wisconsin"


class PygNodePropPredDataset(Enum):
    OGBNARXIV = "PygNodePropPredDataset+ogbn-arxiv"
    OGBNPROTEINS = "PygNodePropPredDataset+ogbn-proteins"


class ModelParams(object):
    """
    A class to represent the model parameters.
    """
    # 1 CORA
    # 2 CITESEER
    # !!!!!!!!!!!!!
    # TODO: DON'T FORGET TO PERFORM HP SEARCH FOR num_layers = [1] !!!!!!!
    # !!!!!!!!!!!!!
    print("Well, hello there!")
    experiment_number = 1  # number of experiment TODO: Think about an identifier for the experiments
    epochs = [300]  # number of epochs (list)
    model_type = LayerType.GCNCONV  # to define the model type TODO: GCNCONV SAGECONV GATCONV TRANSFORMERCONV
    num_hidden_features = [40, 80, 120, 160, 200, 240, 280, 320, 360, 400]  # [280]  # [60, 120, 160, 200, 240, 280, 320]  # number of hidden features (list)
    lr_model = [1e-03]  # [1e-4, 1e-5]  # learning rate for the model (list)
    lr_activation = [1e-05]  # learning rate for the activations (list)
    weight_decay = [1e-04]  # weight decay for both
    clusters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # number of clusters (list)
    num_layers = [4]  # [1,2,4,8,16]  # number of layers (list) [64, 96, 128]
    num_activation = [4]  # number of activations inside RPM (list)
    n = 4  # numerator
    m = 5  # denominator
    recluster_option = ReclusterOption.TH  # options are iterative, thresholding or iterative
    activation_type = [ActivationType.RAT]  # activation type (list)
    mode = [DistanceMetrics.COSINE]  # distance metric type (list)
    with_clusters = [True]  # flag for clustering
    use_coefficients = True  # flag for use of coefficients in our Rationals
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Types: Planetoid, CitationFull, Amazon, WikipediaNetwork, WebKB
    # name of the dataset (Cora, CiteSeer, PubMed, Cora_ML, chameleon, Photo etc.)
    set_dataset = PygNodePropPredDataset.OGBNARXIV # Planetoid.CORA # CitationFull.CITESEER  # here to set the dataset TODO: CiteFull.CITESEER, Planetoid.CORA
    task_type = TaskType.NODEPROPPRED # TaskType.NODECLASSIFICATION
    type_name = set_dataset.value.split("+")
    dataset_type = type_name[0]
    dataset_name = type_name[1]
    set_ = 0
    seed = 0  # random.randint(1, 9999999)
    log_path = (
        f"../log_files/sensitivity_analysis_thr_cos/{model_type}/experiment{experiment_number}_{task_type.value}_ds_{dataset_name}"
        f"_type_{dataset_type}_layers_{num_layers[0]}"
        f"_Model_{model_type.value}/seed{set_}"
    )  # "../log_files/example_dir"

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # for deep GNN
    compute_other_metrics = False  # to compute MAD, MADGap, DE, MADC, GMAD
    print(f"Computer other metrics? - {compute_other_metrics}")
    save_model = True  # to save trained model
    plot_centers = False  # to plot centers after training
    direct_visualization = False  # to visualize accuracy after training

    # ablation metrics
    normalize = True


    # Set cluster type
    use_gmm = False

    if task_type == TaskType.NODEREGRESSION:
        assert set_dataset in [WikipediaNetwork.SQUIRREL, WikipediaNetwork.CHAMELEON]