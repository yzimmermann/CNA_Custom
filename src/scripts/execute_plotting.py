import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import itertools

from utils.model_params import ActivationType, DistanceMetrics, ReclusterOption
from utils.visualizations import ConfigComparisonVisualizer


class ModelParams(object):
    """
    A class to represent the model parameters.
    """

    experiment_number = [5, 6]  # number of experiment
    epochs = [200]  # number of epochs (list)
    num_hidden_features = [140]  # number of hidden features (list)
    lr_model = [0.01, 0.001, 0.0001, 1e-05]  # learning rate for the model (list)
    lr_activation = [1e-05]  # learning rate for the activations (list)
    weight_decay = 5e-4  # weight decay for both
    clusters = [14]  # number of clusters (list)
    num_layers = [
        2,
        4,
        8,
        16,
        32,
        64,
        100,
    ]  # number of layers (list)
    num_activation = [4]  # number of activations inside RPM (list)
    n = 5  # numerator
    m = 5  # denominator
    recluster_option = ReclusterOption.ITR
    activation_type = [ActivationType.RELU]  # activation type (list)
    mode = [DistanceMetrics.EUCLIDEAN]  # distance metric type (list)
    with_clusters = [False, True]  # flag for clustering
    use_coefficients = True  # flag for use of coefficients in our Rationals


def execute() -> None:
    """
    Main function to run the model visualization.
    """
    params_dict = {
        "experiment_number": ModelParams.experiment_number,
        "epochs": ModelParams.epochs,
        "num_hidden_features": ModelParams.num_hidden_features,
        "lr_model": ModelParams.lr_model,
        "lr_activation": ModelParams.lr_activation,
        "weight_decay": ModelParams.weight_decay,
        "clusters": ModelParams.clusters,
        "num_layers": ModelParams.num_layers,
        "num_activation": ModelParams.num_activation,
        "n": ModelParams.n,
        "m": ModelParams.m,
        "recluster_option": ModelParams.recluster_option,
        "activation_type": ModelParams.activation_type,
        "mode": ModelParams.mode,
        "with_clusters": ModelParams.with_clusters,
    }

    # Get all possible combinations of the parameter values
    list_dict = {
        key: value for key, value in params_dict.items() if isinstance(value, list)
    }
    param_combinations = list(itertools.product(*list_dict.values()))
    # Loop through each combination of hyperparameters and perform model training and visualization.
    configs = []
    for combination in param_combinations:
        (
            experiment_number,
            epoch,
            num_hidden_features,
            lr_model,
            lr_activation,
            clusters,
            num_layers,
            num_activation,
            activation_type,
            mode,
            with_clusters,
        ) = combination
        config = {
            "experiment_number": ModelParams.experiment_number,
            "epochs": epoch,
            "num_hidden_features": num_hidden_features,
            "lr_model": lr_model,
            "lr_activation": lr_activation,
            "weight_decay": ModelParams.weight_decay,
            "clusters": clusters,
            "num_layers": num_layers,
            "num_activation": num_activation,
            "n": ModelParams.n,
            "m": ModelParams.m,
            "activation_type": activation_type,
            "recluster_option": ModelParams.recluster_option,
            "mode": mode,
            "with_clusters": with_clusters,
        }

        experiment_number = config["experiment_number"]
        epochs = config["epochs"]
        num_hidden_features = config["num_hidden_features"]
        lr_model = config["lr_model"]
        lr_activation = config["lr_activation"]
        weight_decay = config["weight_decay"]
        clusters = config["clusters"]
        num_layers = config["num_layers"]
        num_activation = config["num_activation"]
        n = config["n"]
        m = config["m"]
        activation_type = config["activation_type"]
        mode = config["mode"]
        with_clusters = config["with_clusters"]
        experiment_name = f"experiment_{experiment_number}"
        recluster_option = config["recluster_option"]

        log_file = (
            f"../log_files/without_activation/Cora/GAT/{experiment_name}_epc_{epochs}_mlr_"
            f"{lr_model}_alr_{lr_activation}_hf_"
            f"{num_hidden_features}_layers_{num_layers}_cl_"
            f"{with_clusters}_ncl_{clusters}_{activation_type}_rco_"
            f"{recluster_option}.txt"
        )

        if os.path.exists(log_file):
            configs.append(log_file)

    plot_name = "comparision_with_without_clustering_2_100"
    visualizer = ConfigComparisonVisualizer(plot_name)
    visualizer.plot_comparison(configs, save_flag=False)


if __name__ == "__main__":
    execute()
