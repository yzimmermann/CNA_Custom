import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import seaborn as sns
import torch
from matplotlib.ticker import AutoMinorLocator

sns.set_context("paper")
sns.set_style("whitegrid")
sns.set_palette("colorblind")
matplotlib.rcParams.update({"font.size": 12})


class Visualizer:
    """
    Initializes a Visualizer object with the given configuration.

    Parameters:
        config (dict): A dictionary containing configuration parameters for the Visualizer.

    Returns:
        Visualizer: The initialized Visualizer object.
    """

    def __init__(self, config):
        self.config = config
        self.experiment_number = config["experiment_number"]
        self.epochs = config["epochs"]
        self.num_hidden_features = config["num_hidden_features"]
        self.lr_model = config["lr_model"]
        self.lr_activation = config["lr_activation"]
        self.weight_decay = config["weight_decay"]
        self.clusters = config["clusters"]
        self.num_layers = config["num_layers"]
        self.num_activation = config["num_activation"]
        self.n = config["n"]
        self.m = config["m"]
        self.activation_type = config["activation_type"]
        self.recluster_option = config["recluster_option"]
        self.mode = config["mode"]
        self.with_clusters = config["with_clusters"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.experiment_name = f"experiment_{self.experiment_number}"
        self.dataset_name = config["dataset_name"]
        self.log_path = config["log_path"]
        self.dataset_type = config["dataset_type"]
        self.test_accuracy_values = []
        self.model_name = (
            f"{self.log_path}/{self.experiment_name}_epc_{self.epochs}_mlr_{self.lr_model}_alr_"
            f"{self.lr_activation}_wd_{self.weight_decay}_hf_{self.num_hidden_features}_layers_"
            f"{self.num_layers}_cl_{self.with_clusters}_ncl_{self.clusters}_"
            f"{self.activation_type}_rco_{self.recluster_option}_"
            f"dst_{self.dataset_type}_dsn_{self.dataset_name}.txt"
        )

        self.read_test_accuracy_from_log(self.model_name)

    def read_test_accuracy_from_log(self, log_file):
        """
        Reads the test accuracy values from the specified log file.

        Parameters:
            log_file (str): The path to the log file.
        """
        with open(log_file, "r") as file:
            for line in file:
                if "Test Accuracy" in line:
                    test_accuracy = float(line.split("Test Accuracy: ")[-1].strip()[:6])
                    self.test_accuracy_values.append(test_accuracy)

    def plot_test_accuracy(self, save_flag=False, file_format="pdf", dpi=600):
        """
        Plots the test accuracy over epochs using SciencePlots.

        Parameters:
            save_flag (bool): Flag to control whether to save the plot or not.
            file_format (str): File format to save the plot. Possible values: "jpg", "pdf", "svg", etc.
            dpi (int): DPI (dots per inch) for the saved image.
        """
        with plt.style.context(["science"]):
            fig, ax = plt.subplots()
            ax.plot(
                np.arange(1, len(self.test_accuracy_values) + 1),
                self.test_accuracy_values,
                # marker="o",
                # markersize=4,
            )
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Test Accuracy")
            ax.set_title("Test Accuracy over Epochs")
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            plt.tight_layout()

            if save_flag:
                save_path = f"figures/test_accuracy.{file_format}"
                fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
                print(f"Plot saved as '{save_path}'.")
            else:
                plt.show()
            plt.close()

    def plot_std_test_accuracy(
        self, experiment_data, labels, save_flag=False, file_format="pdf", dpi=600
    ):
        """
        Plots the test accuracy with a 95% confidence interval (standard deviation) over epochs for multiple experiments using SciencePlots.

        Parameters:
            experiment_data (list): List of test accuracy values for each experiment.
            labels (list): List of labels for each experiment.
            save_flag (bool): Flag to control whether to save the plot or not.
            file_format (str): File format to save the plot. Possible values: "jpg", "pdf", "svg", etc.
            dpi (int): DPI (dots per inch) for the saved image.
        """
        with plt.style.context(["science"]):
            fig, ax = plt.subplots()
            for i, data in enumerate(experiment_data):
                mean_accuracy = np.mean(data)
                std_accuracy = np.std(data)
                confidence_interval = 1.96 * (std_accuracy / np.sqrt(len(data)))
                epochs = np.arange(1, len(data) + 1)
                ax.plot(
                    epochs,
                    mean_accuracy,
                    label=labels[i],
                )
                ax.fill_between(
                    epochs,
                    mean_accuracy - confidence_interval,
                    mean_accuracy + confidence_interval,
                    alpha=0.3,
                )
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Test Accuracy")
            ax.set_title("Test Accuracy over Epochs with 95% Confidence Interval")
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.legend(title="Experiment")
            plt.tight_layout()

            if save_flag:
                save_path = f"figures/std_test_accuracy.{file_format}"
                fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
                print(f"Plot saved as '{save_path}'.")
            else:
                plt.show()
            plt.close()


class ConfigComparisonVisualizer:
    def __init__(self, plot_name):
        self.plot_name = plot_name

    def get_max_test_accuracy_from_log(self, log_file):
        """
        Get the maximum test accuracy value from a log file.

        This function reads the specified log file, extracts the test accuracy values,
        and returns the maximum test accuracy found in the log file.

        Parameters:
            log_file (str): The path to the log file containing test accuracy values.

        Returns:
            float: The maximum test accuracy value found in the log file.
        """
        test_accuracy_values = []
        with open(log_file, "r") as file:
            for line in file:
                if "Test Accuracy" in line:
                    test_accuracy = float(line.split("Test Accuracy: ")[-1].strip()[:6])
                    test_accuracy_values.append(test_accuracy)
        return max(test_accuracy_values)

    def read_value_from_log(self, log_file, value_name):
        """
        Reads a specific value from the specified log file.

        Parameters:
            log_file (str): The path to the log file.
            value_name (str): The name of the value to read from the log.

        Returns:
            float: The value extracted from the log.
        """
        with open(log_file, "r") as file:
            for line in file:
                if value_name in line:
                    word = line.split(f"{value_name}: ")[-1].strip()[:6]
                    if word == "True":
                        value = True
                    elif word == "False":
                        value = False
                    else:
                        value = float(line.split(f"{value_name}: ")[-1].strip()[:6])
                    return value

    def initialize_dicts(self, configs):
        """
        Initialize a list of dictionaries containing configuration details.

        This function takes a list of configuration strings, extracts relevant information
        from each configuration, and creates dictionaries for each configuration with keys
        representing configuration details. The dictionaries are sorted based on the number
        of layers and returned as a sorted list.

        Parameters:
            configs (list): A list of configuration strings.

        Returns:
            list: A sorted list of dictionaries containing configuration details.
        """
        with_clustering = []
        for c in configs:
            all_configs_with = {}
            start_index = c.index("ActivationType.") + len("ActivationType.")
            end_index = c.index("_", start_index)
            all_configs_with["num_layers"] = self.read_value_from_log(
                c, "Number of Layers"
            )
            all_configs_with["with_clustering"] = self.read_value_from_log(
                c, "With Clusters"
            )
            all_configs_with["num_clusters"] = self.read_value_from_log(
                c, "Number of Clusters"
            )
            all_configs_with["accuracy"] = self.get_max_test_accuracy_from_log(c)
            all_configs_with["activation_type"] = c[start_index:end_index]
            with_clustering.append(all_configs_with)

        sorted_data = sorted(with_clustering, key=lambda x: x["num_layers"])
        return sorted_data

    def get_the_cases(self, configs):
        """
        Split a list of configurations into two lists based on a specific keyword.

        This function takes a list of configurations and splits it into two separate lists:
        one list containing configurations that contain a specific keyword and another list
        containing configurations that do not contain that keyword.

        Parameters:
            configs (list): A list of configuration strings.

        Returns:
            tuple: A tuple containing two lists. The first list contains configurations
                   containing the specified keyword, and the second list contains
                   configurations not containing the keyword.
        """
        configs = list(dict.fromkeys(configs))
        key_word = "cl_True"
        configs_with_clustering = [s for s in configs if key_word in s]
        configs_without_clustering = [s for s in configs if key_word not in s]
        return configs_with_clustering, configs_without_clustering

    def plot_comparison(self, configs, save_flag=False, file_format="pdf", dpi=600):
        """
        Plot a comparison of test accuracy values based on different configurations.

        This function takes a list of configurations, separates them into cases with and without clustering,
        and plots a comparison of test accuracy values against the number of layers for each case and
        activation type.

        Parameters:
            configs (list): A list of dictionaries containing configuration details.

        Returns:
            None
        """
        configs_with_clustering, configs_without_clustering = self.get_the_cases(
            configs
        )
        sorted_with_clustering = self.initialize_dicts(configs_with_clustering)
        sorted_without_clustering = self.initialize_dicts(configs_without_clustering)
        activation_types = set(
            entry["activation_type"] for entry in sorted_with_clustering
        )
        with_clustering_values = set(
            entry["with_clustering"] for entry in sorted_with_clustering
        )
        num_clusters_values = set(
            int(entry["num_clusters"]) for entry in sorted_with_clustering
        )

        fig, ax = plt.subplots()
        layers_to_compare = [2, 4, 8, 16, 32, 64, 100]

        for activation_type in activation_types:
            for with_clustering in with_clustering_values:
                filtered_data = [
                    entry
                    for entry in sorted_with_clustering
                    if entry["activation_type"] == activation_type
                    and entry["with_clustering"] == with_clustering
                ]
                x_values = [entry["num_layers"] for entry in filtered_data]
                y_values = [entry["accuracy"] for entry in filtered_data]
                if with_clustering:
                    text = "without activations"
                label = f"{text}"
                ax.plot(x_values, y_values, label=label)

        activation_types = set(
            entry["activation_type"] for entry in sorted_without_clustering
        )
        with_clustering_values = set(
            entry["with_clustering"] for entry in sorted_without_clustering
        )
        for activation_type in activation_types:
            for with_clustering in with_clustering_values:
                filtered_data = [
                    entry
                    for entry in sorted_without_clustering
                    if entry["activation_type"] == activation_type
                    and entry["with_clustering"] == with_clustering
                ]
                x_values = [entry["num_layers"] for entry in filtered_data]
                y_values = [entry["accuracy"] for entry in filtered_data]
                if not with_clustering:
                    text = "with activations"
                label = f"{text}"
                ax.plot(x_values, y_values, label=label)

        ax.set_xlabel("Number of Layers")
        ax.set_ylabel("Test Accuracy")
        ax.set_title("Cora: Test Accuracy vs. Number of Layers")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_ylim((0.1, 0.9))
        # Set the x-axis tick values and labels
        ax.set_xticks(layers_to_compare)
        ax.set_xticklabels(layers_to_compare)
        ax.legend()
        plt.tight_layout()

        if save_flag:
            save_path = f"../plots/{self.plot_name}.{file_format}"
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"Plot saved as '{save_path}'.")
        else:
            plt.show()
        plt.close()
