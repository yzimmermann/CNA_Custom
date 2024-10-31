import json
import os
import random
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import argparse
import itertools
import logging

from scripts.model_trainer import ModelTrainer
from scripts.regression_trainer import RegressionTrainer
from utils.model_params import (
    ActivationType,
    LayerType,
    ModelParams,
    ReclusterOption,
    TaskType,
)
from utils.utils import set_seed, set_mode
from utils.visualizations import Visualizer


def execute(json_data=None) -> None:
    """
    Main function to run the model training and visualization.
    """
    if json_data is None:
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
            "dataset_name": ModelParams.dataset_name,
            "log_path": ModelParams.log_path,
            "dataset_type": ModelParams.dataset_type,
        }

        # Get all possible combinations of the parameter values
        list_dict = {
            key: value for key, value in params_dict.items() if isinstance(value, list)
        }
        param_combinations = list(itertools.product(*list_dict.values()))
        # Loop through each combination of hyperparameters and perform model training and visualization.
        for combination in param_combinations:
            (
                epoch,
                num_hidden_features,
                lr_model,
                lr_activation,
                weight_decay,
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
                "weight_decay": weight_decay,
                "clusters": clusters,
                "num_layers": num_layers,
                "num_activation": num_activation,
                "n": ModelParams.n,
                "m": ModelParams.m,
                "activation_type": activation_type,
                "recluster_option": ModelParams.recluster_option,
                "mode": mode,
                "with_clusters": with_clusters,
                "dataset_name": ModelParams.dataset_name,
                "log_path": ModelParams.log_path,
                "dataset_type": ModelParams.dataset_type,
                "model_type": ModelParams.model_type,
                "normalize": ModelParams.normalize,
            }

            set_mode(config["with_clusters"] and config["normalize"])

            set_seed(seed=ModelParams.seed)
            # execute the training process
            if ModelParams.task_type.value == "node_regression":
                all_test_losses = []
                for split in range(10):
                    trainer = RegressionTrainer(config)
                    min_test_loss = trainer.train(split)
                    all_test_losses.append(min_test_loss)
                mean_value = np.mean(all_test_losses)
                std_deviation = np.std(all_test_losses)
                print(f"Final test results -- mean: {mean_value:.4f}, std: {std_deviation:.4f}")
                logging.info(
                    f"Final test results -- mean: {mean_value:.4f}, std: {std_deviation:.4f}"
                )
            else:
                trainer = ModelTrainer(config)
                min_test_loss = trainer.train()

            # visualize the trained metrics directly
            if ModelParams.direct_visualization:
                visualizer = Visualizer(config)
                visualizer.plot_test_accuracy()
    else:
        config = {
            "experiment_number": json_data["experiment_number"],
            "epochs": json_data["epochs"],
            "num_hidden_features": json_data["num_hidden_features"],
            "lr_model": json_data["lr_model"],
            "lr_activation": json_data["lr_activation"],
            "weight_decay": json_data["weight_decay"],
            "clusters": json_data["clusters"],
            "num_layers": json_data["num_layers"],
            "num_activation": ModelParams.num_activation[0],
            "n": 5,
            "m": 5,
            "activation_type": get_activation_type(json_data["activation_type"]),
            "recluster_option": ReclusterOption.ITR,
            "mode": ModelParams.mode[0],
            "with_clusters": json_data["with_clusters"],
            "dataset_name": json_data["dataset_name"],
            "log_path": None,
            "dataset_type": json_data["dataset_type"],
            "task_type": json_data["task"],
            "model_type": get_layer_type(json_data["model"]),
            "normalize": json_data["normalize"],
        }
        for i in range(args.num_seeds):
            config["log_path"] = (
                f"../log_files/KK_experiment{config['experiment_number']}_"
                f"{config['task_type']}_ds_{config['dataset_name']}"
                f"_type_{config['dataset_type']}_layers_{config['num_layers']}"
                f"_Model_{config['model_type']}/seed{i}"
            )

            if not os.path.exists(config["log_path"]):
                os.makedirs(config["log_path"])

            set_mode(config["with_clusters"] and config["normalize"])

            set_seed(seed=i)  # random.randint(10000, 99999))
            # execute the training process
            if config["task_type"] == "node_regression":
                all_test_losses = []
                for split in range(10):
                    trainer = RegressionTrainer(config)
                    min_test_loss = trainer.train(split)
                    all_test_losses.append(min_test_loss)
                mean_value = np.mean(all_test_losses)
                std_deviation = np.std(all_test_losses)
                print(f"Final test results -- mean: {mean_value:.4f}, std: {std_deviation:.4f}")
                logging.info(
                    f"Final test results -- mean: {mean_value:.4f}, std: {std_deviation:.4f}"
                )
            else:
                trainer = ModelTrainer(config)
                trainer.train()
            # visualize the trained metrics directly
            if ModelParams.direct_visualization:
                visualizer = Visualizer(config)
                visualizer.plot_test_accuracy()


def get_activation_type(type_config):
    """Parameters:
        - type_config (str): A string representing the activation type.
    Returns:
        - enum_member (ActivationType): The corresponding enum value for the given activation type.
    Processing Logic:
        - Loop through each enum member.
        - Check if the string representation of the enum member matches the given activation type.
        - If there is a match, return the enum member.
        - If there is no match, raise a ValueError."""
    for enum_member in ActivationType:
        if str(enum_member) == type_config:
            return enum_member
    raise ValueError("No matching enum value found for the given activation type")


def get_layer_type(layer_type):
    for enum_member in LayerType:
        if enum_member.value == layer_type:
            return enum_member
    raise ValueError("No matching enum value found for the given layer type")


def get_available_configs(directory):
    """Parameters:
        - directory (str): Path to the directory containing config files.
    Returns:
        - config_files (list): List of config file names.
    Processing Logic:
        - Get all files in directory.
        - Filter for only JSON files.
        - Extract file names without extension."""
    config_files = [
        filename.split(".")[0]
        for filename in os.listdir(directory)
        if filename.endswith(".json")
    ]
    return config_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process config files in executing.")
    parser.add_argument(
        "--config",
        type=str,
        default="model_params",
        help="Name of the configuration to load as a dictionary. "
        "Available options:\n" + "\n".join(get_available_configs("utils/configs/")),
    )
    parser.add_argument(
        "--num_seeds", type=int, default=5, help="The number of seeds to be executed"
    )
    parser.add_argument(
        "--n", type=int, default=4, help="The degreee of denominator"
    )
    parser.add_argument(
        "--m", type=int, default=5, help="The degreee of nominator"
    )

    args = parser.parse_args()
    if args.config != "model_params":
        try:
            with open(f"utils/configs/{args.config.lower()}.json", "r") as file:
                json_data = json.load(file)
                #print(json_data)
                execute(
                    json_data
                )
        except FileNotFoundError:
            print(f"Error: File {args.config.lower()}.json not found.")
    else:
        execute()
