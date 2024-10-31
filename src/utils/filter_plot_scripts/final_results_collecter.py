import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import itertools
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

cora_table_with = defaultdict(dict)

sns.set(style="whitegrid")


def extract_and_convert_max_test_accuracy(file_path):
    """
    Extracts and converts the maximum test accuracy and related metrics from a text file.

    Parameters:
    - file_path (str): The path to the text file containing the test accuracy information.

    Returns:
    - dict: A dictionary containing the extracted metrics, including epochs, max test accuracy,
            Mean Absolute Deviation (MAD), MADGap, DE, MADC, GMAD, and the number of layers.
    """
    result_dict = {}
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            if "Max Test Accuracy:" in line:
                parts = line.split(" - ")
                # print(parts)
                result_dict["epochs"] = int(parts[2].split(" ")[1])
                result_dict["max_test_accuracy"] = float(parts[3].split(": ")[1])
                #result_dict["MAD"] = float(parts[4].split(": ")[1])
                #result_dict["MADGap"] = float(parts[5].split(": ")[1])
                #result_dict["DE"] = float(parts[6].split(": ")[1])
                #result_dict["MADC"] = float(parts[7].split(": ")[1])
                #result_dict["GMAD"] = float(parts[8].split(": ")[1])
                result_dict["layers"] = int(parts[4].split(":")[1])
    return result_dict


def extract_max_test_accuracy(file_path):
    """
    Extracts the maximum test accuracy from a text file.

    Parameters:
    - file_path (str): The path to the text file containing test accuracy information.

    Returns:
    - float: The maximum test accuracy found in the file.
    """
    max_test_accuracy = 0.0
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            if "Test Accuracy" in line:
                test_accuracy = float(line.split("Test Accuracy: ")[1].split(" ")[0])
                max_test_accuracy = max(max_test_accuracy, test_accuracy)
    return max_test_accuracy


def plot_std_95(list_of_data, name):
    """
    Plots the mean accuracy with a 95% confidence interval based on a list of data dictionaries.

    Parameters:
    - list_of_data (list): A list of dictionaries containing data for different experiments.
    - name (str): A name identifier for the plot.

    Returns:
    - None: The function displays the plot and saves it as a PDF file.
    """
    if not list_of_data:
        print("Empty list of data.")
        return

    layers = list(list_of_data[0].keys())
    mean_values = np.mean([list(d.values()) for d in list_of_data], axis=0)
    std_values = np.std([list(d.values()) for d in list_of_data], axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))

    plot_mean = sns.lineplot(
        x=layers,
        y=mean_values,
        label="Mean Accuracy",
        palette="rocket",
        dashes=False,
        markers=True,
        ci="sd",
        ax=ax,
    )

    ax.fill_between(
        layers,
        mean_values - 1.96 * std_values,
        mean_values + 1.96 * std_values,
        alpha=0.2,
    )

    ax.set_xticks(layers)
    ax.set_xlabel("Number of Layers", fontsize=18)
    ax.set_ylabel("Test Accuracy", fontsize=18)
    ax.set_title("Mean Accuracy with 95% Confidence Interval", fontsize=20)
    ax.legend()

    ax.set_ylim([0.2, 0.95])

    plt.tight_layout()
    fig = plot_mean.get_figure()
    fig.savefig(f"std_95_plot_{name}_{subset}.pdf", dpi=600)

    plt.show()


def process_value(value):
    """
    Processes a value, converting it to a suitable format.

    Parameters:
    - value: The value to be processed.

    Returns:
    - float or None: If the value is a float, it is returned as is. If it's a dictionary, 0 is returned.
                    Otherwise, None is returned.
    """
    if isinstance(value, float):
        return value
    elif isinstance(value, dict):
        return 0
    else:
        return None


def process_seed_directories(seed_directories, directory_path, name="Cora"):
    """
    Processes directories containing results from different seed experiments.

    Parameters:
    - seed_directories (list): List of directories containing experiment results.
    - directory_path (str): The base path to the experiment result directories.
    - name (str): A name identifier for the experiment (default is "Cora").

    Returns:
    - None: The function prints result tables, config data, and saves the result data as a JSON file.
            It also generates and saves a plot of mean accuracy with a 95% confidence interval.
    """
    list_dicts = []
    list_results = []
    for seed_dir in seed_directories:
        directory = f"{directory_path}/{seed_dir}"
        result_table = defaultdict(dict)
        config_data = defaultdict(dict)
        # print(directory)
        for root, dirs, files in os.walk(directory):
            for file in files:
                # print(file)
                if file.endswith(".txt"):
                    num_layers = int(file.split("_layers_")[1].split("_")[0])
                    accuracy = extract_max_test_accuracy(os.path.join(root, file))
                    if process_value(result_table[num_layers]) < accuracy:
                        config_data[num_layers] = os.path.join(seed_dir, file)
                    result_dict = extract_and_convert_max_test_accuracy(
                        os.path.join(root, file)
                    )
                    list_dicts.append(result_dict)
                    result_table[num_layers] = max(
                        process_value(result_table[num_layers]), accuracy
                    )

        result_table = dict(sorted(result_table.items()))
        config_data = dict(sorted(config_data.items()))

        # Print the tables
        # print(f"Result Table for Seed Directories ({directory}):")
        #  print(result_table)
        # print(f"Config Data for Seed Directories ({directory}):")
        # for key, value in config_data.items():
        #    print(f'{key} : {value}')
        list_results.append(result_table)
    # print(list_results)
    tmp = np.array([next(iter(elm.items()))[1] for elm in list_results])
    # print(tmp.__len__())
    std = 100*np.std(tmp, axis=0)
    mean = 100*np.mean(tmp, axis=0)
    experiment = directory_path.split("/")[2].split("_")[1]
    print(f"{experiment} {mean}+/-{std}")
    # print(100*np.std(tmp, axis=0))
    # print(100*np.mean(tmp, axis=0))
    # Save the list of dictionaries as JSON
    # with open(f"result_data_citatoinfull_{name}_cluster_{subset}.json", "w") as json_file:
    #     json.dump(list_dicts, json_file, indent=2)

    # plot_std_95(list_results, name)


if __name__ == "__main__":
    subset = "80_10_10"
    # List of seed directories to process
    # seed_directories = ["seedA", "seedB", "seedC", "seedD", "seedE"]
    seed_directories = ['seed'+str(i) for i in range(5)]
    # directory_path = f"experiment8({subset})/{seed_dir}"
    # directory_path = f"experiment_CiteSeer_cluster_{subset}/"
    # directory_path = f"experiment_citeseer_citatoinfull_{subset}/"
    # directory_path = "log_files/KK_experiment70_node_classification_ds_CiteSeer_type_Planetoid_layers_8_Model_LayerType.GCNCONV/"
    # process_seed_directories(seed_directories, directory_path, name="CiteSeer")
    # directory_path = "log_files/KK_experiment71_node_classification_ds_CiteSeer_type_Planetoid_layers_8_Model_LayerType.GCNCONV/"
    # process_seed_directories(seed_directories, directory_path, name="CiteSeer")
    # directory_path = "log_files/KK_experiment72_node_classification_ds_CiteSeer_type_Planetoid_layers_8_Model_LayerType.GCNCONV/"
    # process_seed_directories(seed_directories, directory_path, name="CiteSeer")
    # directory_path = "log_files/KK_experiment73_node_classification_ds_CiteSeer_type_Planetoid_layers_8_Model_LayerType.GCNCONV/"
    # process_seed_directories(seed_directories, directory_path, name="CiteSeer")
    # directory_path = "log_files/KK_experiment74_node_classification_ds_CiteSeer_type_Planetoid_layers_8_Model_LayerType.GCNCONV/"
    # process_seed_directories(seed_directories, directory_path, name="CiteSeer")
    # directory_path = "log_files/KK_experiment75_node_classification_ds_CiteSeer_type_Planetoid_layers_8_Model_LayerType.GCNCONV/"
    # process_seed_directories(seed_directories, directory_path, name="CiteSeer")
    # directory_path = "log_files/KK_experiment76_node_classification_ds_CiteSeer_type_Planetoid_layers_8_Model_LayerType.GCNCONV/"
    # process_seed_directories(seed_directories, directory_path, name="CiteSeer")
    # directory_path = "log_files/KK_experiment77_node_classification_ds_CiteSeer_type_CitationFull_layers_8_Model_LayerType.GCNCONV/"
    # process_seed_directories(seed_directories, directory_path, name="CiteSeer")
    # directory_path = "log_files/KK_experiment78_node_classification_ds_CiteSeer_type_CitationFull_layers_8_Model_LayerType.SAGECONV/"
    # process_seed_directories(seed_directories, directory_path, name="CiteSeer")
    # directory_path = "log_files/KK_experiment79_node_classification_ds_CiteSeer_type_CitationFull_layers_8_Model_LayerType.TRANSFORMERCONV/"
    # process_seed_directories(seed_directories, directory_path, name="CiteSeer")
    # directory_path = "log_files/KK_experiment80_node_classification_ds_CiteSeer_type_CitationFull_layers_8_Model_LayerType.GATCONV/"
    # # Process the seed directories
    # process_seed_directories(seed_directories, directory_path, name="CiteSeer")

    # directory_path = "log_files/KK_experiment82_node_classification_ds_CiteSeer_type_CitationFull_layers_8_Model_LayerType.GCNCONV/"
    # process_seed_directories(seed_directories, directory_path, name="CiteSeer")
    # directory_path = "log_files/KK_experiment83_node_classification_ds_CiteSeer_type_CitationFull_layers_8_Model_LayerType.GCNCONV/"
    # process_seed_directories(seed_directories, directory_path, name="CiteSeer")
    # directory_path = "log_files/KK_experiment84_node_classification_ds_CiteSeer_type_CitationFull_layers_8_Model_LayerType.GCNCONV/"
    # process_seed_directories(seed_directories, directory_path, name="CiteSeer")
    # directory_path = "log_files/KK_experiment85_node_classification_ds_CiteSeer_type_CitationFull_layers_8_Model_LayerType.GCNCONV/"
    # process_seed_directories(seed_directories, directory_path, name="CiteSeer")
    # directory_path = "log_files/KK_experiment86_node_classification_ds_CiteSeer_type_CitationFull_layers_8_Model_LayerType.GCNCONV/"
    # process_seed_directories(seed_directories, directory_path, name="CiteSeer")
    # directory_path = "log_files/KK_experiment87_node_classification_ds_CiteSeer_type_CitationFull_layers_8_Model_LayerType.GCNCONV/"
    # process_seed_directories(seed_directories, directory_path, name="CiteSeer")
    # directory_path = "log_files/KK_experiment88_node_classification_ds_CiteSeer_type_CitationFull_layers_8_Model_LayerType.GCNCONV/"
    # process_seed_directories(seed_directories, directory_path, name="CiteSeer")
    # directory_path = "log_files/KK_experiment90_node_classification_ds_Cora_type_Planetoid_layers_4_Model_LayerType.GCNCONV/"
    # process_seed_directories(seed_directories, directory_path, name="Cora")
    # directory_path = "log_files/KK_experiment91_node_classification_ds_Cora_type_Planetoid_layers_4_Model_LayerType.GCNCONV/"
    # process_seed_directories(seed_directories, directory_path, name="Cora")
    # directory_path = "log_files/KK_experiment92_node_classification_ds_Cora_type_Planetoid_layers_4_Model_LayerType.GCNCONV/"
    # process_seed_directories(seed_directories, directory_path, name="Cora")
    # directory_path = "log_files/KK_experiment93_node_classification_ds_Cora_type_Planetoid_layers_4_Model_LayerType.GCNCONV/"
    # process_seed_directories(seed_directories, directory_path, name="Cora")
    # directory_path = "log_files/KK_experiment94_node_classification_ds_Cora_type_Planetoid_layers_4_Model_LayerType.GCNCONV/"
    # process_seed_directories(seed_directories, directory_path, name="Cora")
    # directory_path = "log_files/KK_experiment95_node_classification_ds_Cora_type_Planetoid_layers_4_Model_LayerType.GCNCONV/"
    # process_seed_directories(seed_directories, directory_path, name="Cora")
    # directory_path = "log_files/KK_experiment96_node_classification_ds_Cora_type_Planetoid_layers_4_Model_LayerType.GCNCONV/"
    # process_seed_directories(seed_directories, directory_path, name="Cora")
    # directory_path = "log_files/KK_experiment97_node_classification_ds_Cora_type_Planetoid_layers_4_Model_LayerType.GCNCONV/"
    # process_seed_directories(seed_directories, directory_path, name="Cora")
    # directory_path = "log_files/KK_experiment98_node_classification_ds_Cora_type_Planetoid_layers_4_Model_LayerType.GATCONV/"
    # process_seed_directories(seed_directories, directory_path, name="Cora")
    # directory_path = "log_files/KK_experiment99_node_classification_ds_Cora_type_Planetoid_layers_4_Model_LayerType.TRANSFORMERCONV/"
    # process_seed_directories(seed_directories, directory_path, name="Cora")
    # directory_path = "log_files/KK_experiment100_node_classification_ds_Cora_type_Planetoid_layers_4_Model_LayerType.SAGECONV/"
    # process_seed_directories(seed_directories, directory_path, name="Cora")
    # directory_path = "log_files/KK_experiment101_node_classification_ds_Cora_type_Planetoid_layers_4_Model_LayerType.GATCONV/"
    # process_seed_directories(seed_directories, directory_path, name="Cora")
    # directory_path = "log_files/KK_experiment102_node_classification_ds_Cora_type_Planetoid_layers_4_Model_LayerType.TRANSFORMERCONV/"
    # process_seed_directories(seed_directories, directory_path, name="Cora")
    # directory_path = "log_files/KK_experiment103_node_classification_ds_Cora_type_Planetoid_layers_4_Model_LayerType.SAGECONV/"
    # process_seed_directories(seed_directories, directory_path, name="Cora")
    # 
    #    Sensitivity Analysis on OGBN-Arxiv
    print("Sensitivity Analysis on OGBN-Arxiv")
    for i in range(1,9):
        directory_path = f"log_files/ablation_ogbn_arxiv/KK_experiment500{i}_node_prop_pred_ds_ogbn-arxiv_type_PygNodePropPredDataset_layers_4_Model_LayerType.GCNCONV"
        process_seed_directories(seed_directories, directory_path, name="OGBN-Arxiv")