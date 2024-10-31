import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import json
from collections import defaultdict

import seaborn as sns

cora_table_with = defaultdict(dict)

sns.set(style="whitegrid")


def extract_and_convert_max_test_accuracy(file_path):
    result_dict = {}
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            if "Max Test Accuracy:" in line:
                parts = line.split(" - ")
                # print(parts)
                result_dict["epochs"] = int(parts[2].split(" ")[1])
                result_dict["max_test_accuracy"] = float(parts[3].split(": ")[1])
                result_dict["layers"] = int(parts[4].split(":")[1])
                result_dict["file_path"] = file_path
    return result_dict


def extract_max_test_accuracy(file_path):
    max_test_accuracy = 0.0
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            if "Test Accuracy" in line:
                test_accuracy = float(line.split("Test Accuracy: ")[1].split(" ")[0])
                max_test_accuracy = max(max_test_accuracy, test_accuracy)
    return max_test_accuracy


def process_value(value):
    if isinstance(value, float):
        return value
    elif isinstance(value, dict):
        return 0
    else:
        return None


def get_highest_accuracy(data):
    if not data:
        return None

    max_accuracy = float("-inf")
    max_accuracy_dict = None
    for d in data:
        if "max_test_accuracy" in d:
            if d["max_test_accuracy"] > max_accuracy:
                max_accuracy = d["max_test_accuracy"]
                max_accuracy_dict = d

    return max_accuracy_dict


def process_seed_directories(seed_directories, directory_path, name="Cora"):
    list_dicts = []
    list_results = []
    for seed_dir in seed_directories:
        directory = f"{directory_path}/{seed_dir}"
        # print(directory)
        result_table = defaultdict(dict)
        config_data = defaultdict(dict)

        for root, dirs, files in os.walk(directory):
            for file in files:
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
        # print(result_table)
        # print(f"Config Data for Seed Directories ({directory}):")
        # print(config_data)
        list_results.append(result_table)
        for i in result_table.keys():
            print(f"{result_table[i]}\t{config_data[i]}")

    # Save the list of dictionaries as JSON
    dict_of_highest_acc = get_highest_accuracy(list_dicts)
    # print(dict_of_highest_acc)
    # with open(f"results_{name}.json", "w") as json_file:
    #     json.dump(dict_of_highest_acc, json_file, indent=2)


if __name__ == "__main__":
    seed_directories = ["seed0"] #, "seed1", "seed2", "seed3", "seed4"]
    # directory_path = f"PATH_TO_DIRECTORY"
    num = ["02", "04", "08", "16", "32", "64"]
    layers = ["2", "4", "8", "16", "32", "64"]  
    # layer_type = "GATCONV"
    layer_type = "GCNCONV"
    # layer_type = "SAGECONV"
    # layer_type = "TRANSFORMERCONV"

    # for l,n in enumerate(num):
    #     # directory_path = os.getcwd()+f"/log_files/KK_experiment50{n}_node_classification_ds_CiteSeer_type_CitationFull_layers_{layers[l]}_Model_LayerType.GCNCONV"
    #     directory_path = os.getcwd()+f"/log_file/ablation_citeseer/LayerType.{layer_type}/experiment2_node_classification_ds_CiteSeer_type_CitationFull_layer_1_Model_TransformerCOnv/seed0/"
    #     process_seed_directories(
    #         seed_directories, directory_path, name=f"{layer_type}_citeseer_layers_{n}"
    #     )

    # CiteSeer
    # 
    # GATConv
    # directory_path = os.getcwd()+f"/log_files/ablation_citeseer/LayerType.{layer_type}/experiment2_node_classification_ds_CiteSeer_type_CitationFull_layers_1_Model_GATConv"
    # directory_path = os.getcwd()+f"/log_files/ablation_citeseer/LayerType.{layer_type}/experiment2_node_classification_ds_CiteSeer_type_CitationFull_layers_32_Model_GATConv"
    # directory_path = os.getcwd()+f"/log_files/ablation_citeseer/LayerType.{layer_type}/experiment2_node_classification_ds_CiteSeer_type_CitationFull_layers_64_Model_GATConv"
    # directory_path = os.getcwd()+f"/log_files/ablation_citeseer/LayerType.{layer_type}/experiment2_node_classification_ds_CiteSeer_type_CitationFull_layers_96_Model_GATConv"
    # /workspaces/bag_of_rationals/log_files/ablation_citeseer/LayerType.GATCONV/experiment2_node_classification_ds_CiteSeer_type_CitationFull_layers_1_Model_GATConv
    #
    # GCNConv
    # directory_path = os.getcwd()+f"/log_files/ablation_citeseer/LayerType.{layer_type}/experiment2_node_classification_ds_CiteSeer_type_CitationFull_layers_1_Model_GCNConv"
    # directory_path = os.getcwd()+f"/log_files/ablation_citeseer/LayerType.{layer_type}/experiment2_node_classification_ds_CiteSeer_type_CitationFull_layers_32_Model_GCNConv"
    # directory_path = os.getcwd()+f"/log_files/ablation_citeseer/LayerType.{layer_type}/experiment2_node_classification_ds_CiteSeer_type_CitationFull_layers_64_Model_GCNConv"
    # directory_path = os.getcwd()+f"/log_files/ablation_citeseer/LayerType.{layer_type}/experiment2_node_classification_ds_CiteSeer_type_CitationFull_layers_96_Model_GCNConv"
    # /workspaces/bag_of_rationals/log_files/ablation_citeseer/LayerType.GCNCONV/experiment2_node_classification_ds_CiteSeer_type_CitationFull_layers_1_Model_GCNConv
    # 
    # SAGEConv
    # /workspaces/bag_of_rationals/log_files/ablation_citeseer/LayerType.SAGECONV/experiment2_node_classification_ds_CiteSeer_type_CitationFull_layers_1_Model_SAGEConv
    # directory_path = os.getcwd()+f"/log_files/ablation_citeseer/LayerType.{layer_type}/experiment2_node_classification_ds_CiteSeer_type_CitationFull_layers_1_Model_SAGEConv"
    # directory_path = os.getcwd()+f"/log_files/ablation_citeseer/LayerType.{layer_type}/experiment2_node_classification_ds_CiteSeer_type_CitationFull_layers_32_Model_SAGEConv"
    # directory_path = os.getcwd()+f"/log_files/ablation_citeseer/LayerType.{layer_type}/experiment2_node_classification_ds_CiteSeer_type_CitationFull_layers_64_Model_SAGEConv"
    # directory_path = os.getcwd()+f"/log_files/ablation_citeseer/LayerType.{layer_type}/experiment2_node_classification_ds_CiteSeer_type_CitationFull_layers_96_Model_SAGEConv"
    # /workspaces/bag_of_rationals/log_files/ablation_citeseer/LayerType.SAGECONV/experiment2_node_classification_ds_CiteSeer_type_CitationFull_layers_32_Model_SAGEConv
    # directory_path = os.getcwd()+f"/log_files/ablation_citeseer/sageconv/experiment2_node_classification_ds_CiteSeer_type_CitationFull_layers_1_Model_SAGEConv"
    # 
    # TransformerConv
    # /workspaces/bag_of_rationals/log_files/ablation_citeseer/LayerType.TRANSFORMERCONV/experiment2_node_classification_ds_CiteSeer_type_CitationFull_layers_1_Model_TransformerConv
    # directory_path = os.getcwd()+f"/log_files/ablation_citeseer/LayerType.{layer_type}/experiment2_node_classification_ds_CiteSeer_type_CitationFull_layers_1_Model_TransformerConv"
    # directory_path = os.getcwd()+f"/log_files/ablation_citeseer/LayerType.{layer_type}/experiment2_node_classification_ds_CiteSeer_type_CitationFull_layers_32_Model_TransformerConv"
    # directory_path = os.getcwd()+f"/log_files/ablation_citeseer/LayerType.{layer_type}/experiment2_node_classification_ds_CiteSeer_type_CitationFull_layers_64_Model_TransformerConv"
    # directory_path = os.getcwd()+f"/log_files/ablation_citeseer/LayerType.{layer_type}/experiment2_node_classification_ds_CiteSeer_type_CitationFull_layers_96_Model_TransformerConv"

    # Cora
    # 
    # GATConv
    # /workspaces/bag_of_rationals/log_files/ablation_cora/LayerType.GATCONV/experiment1_node_classification_ds_Cora_type_Planetoid_layers_1_Model_GATConv
    # directory_path = os.getcwd()+f"/log_files/ablation_cora/LayerType.{layer_type}/experiment1_node_classification_ds_Cora_type_Planetoid_layers_1_Model_GATConv"
    # directory_path = os.getcwd()+f"/log_files/ablation_cora/LayerType.{layer_type}/experiment1_node_classification_ds_Cora_type_Planetoid_layers_32_Model_GATConv"
    # directory_path = os.getcwd()+f"/log_files/ablation_cora/LayerType.{layer_type}/experiment1_node_classification_ds_Cora_type_Planetoid_layers_64_Model_GATConv"
    # directory_path = os.getcwd()+f"/log_files/ablation_cora/LayerType.{layer_type}/experiment1_node_classification_ds_Cora_type_Planetoid_layers_96_Model_GATConv"
    #
    # GCNConv
    # /workspaces/bag_of_rationals/log_files/ablation_cora/LayerType.GCNCONV/experiment1_node_classification_ds_Cora_type_Planetoid_layers_1_Model_GCNConv
    # directory_path = os.getcwd()+f"/log_files/ablation_cora/LayerType.{layer_type}/experiment1_node_classification_ds_Cora_type_Planetoid_layers_1_Model_GCNConv"
    # directory_path = os.getcwd()+f"/log_files/ablation_cora/LayerType.{layer_type}/experiment1_node_classification_ds_Cora_type_Planetoid_layers_16_Model_GCNConv"
    # directory_path = os.getcwd()+f"/log_files/ablation_cora/LayerType.{layer_type}/experiment1_node_classification_ds_Cora_type_Planetoid_layers_32_Model_GCNConv"
    # directory_path = os.getcwd()+f"/log_files/ablation_cora/LayerType.{layer_type}/experiment1_node_classification_ds_Cora_type_Planetoid_layers_64_Model_GCNConv"
    directory_path = os.getcwd()+f"/log_files/ablation_cora/LayerType.{layer_type}/experiment1_node_classification_ds_Cora_type_Planetoid_layers_96_Model_GCNConv"
    # print("FICKEN! ICH HASSE MICH FOKKUSSIEREN ZU MÜSSEN, WEIL MIR PEINLICH IST, WIE WENIG ICH HINKRIEGE!")
    #
    # SAGECONV
    # /workspaces/bag_of_rationals/log_files/ablation_cora/LayerType.SAGECONV/experiment1_node_classification_ds_Cora_type_Planetoid_layers_1_Model_SAGEConv
    # directory_path = os.getcwd()+f"/log_files/ablation_cora/LayerType.{layer_type}/experiment1_node_classification_ds_Cora_type_Planetoid_layers_1_Model_SAGEConv"
    # directory_path = os.getcwd()+f"/log_files/ablation_cora/LayerType.{layer_type}/experiment1_node_classification_ds_Cora_type_Planetoid_layers_16_Model_SAGEConv"
    # directory_path = os.getcwd()+f"/log_files/ablation_cora/LayerType.{layer_type}/experiment1_node_classification_ds_Cora_type_Planetoid_layers_32_Model_SAGEConv"
    # directory_path = os.getcwd()+f"/log_files/ablation_cora/LayerType.{layer_type}/experiment1_node_classification_ds_Cora_type_Planetoid_layers_64_Model_SAGEConv"
    # directory_path = os.getcwd()+f"/log_files/ablation_cora/LayerType.{layer_type}/experiment1_node_classification_ds_Cora_type_Planetoid_layers_96_Model_SAGEConv"
    # 
    # TRANSFORMERCONV
    # /workspaces/bag_of_rationals/log_files/ablation_cora/LayerType.TRANSFORMERCONV/experiment1_node_classification_ds_Cora_type_Planetoid_layers_1_Model_TransformerConv
    # directory_path = os.getcwd()+f"/log_files/ablation_cora/LayerType.{layer_type}/experiment1_node_classification_ds_Cora_type_Planetoid_layers_1_Model_TransformerConv"
    # directory_path = os.getcwd()+f"/log_files/ablation_cora/LayerType.{layer_type}/experiment1_node_classification_ds_Cora_type_Planetoid_layers_32_Model_TransformerConv"
    # directory_path = os.getcwd()+f"/log_files/ablation_cora/LayerType.{layer_type}/experiment1_node_classification_ds_Cora_type_Planetoid_layers_64_Model_TransformerConv"
    # directory_path = os.getcwd()+f"/log_files/ablation_cora/LayerType.{layer_type}/experiment1_node_classification_ds_Cora_type_Planetoid_layers_96_Model_TransformerConv"
    print("Ich bin geliebt und schlau! Ich kriege auch das mit links hin!")
    print(os.path.isdir(directory_path))
    process_seed_directories(
        seed_directories, directory_path, name=f"{layer_type}_citeseer"
    )
    # directory_path = os.getcwd()+"/log_files/experiment(40,)_node_prop_pred_ds_ogbn-arxiv_type_PygNodePropPredDataset_layers_4_Model_SAGEConv/"
    # directory_path = os.getcwd()+"/log_files/experiment(41,)_node_prop_pred_ds_ogbn-arxiv_type_PygNodePropPredDataset_layers_4_Model_GCNConv/"
    # process_seed_directories(
    #     seed_directories, directory_path, name="GCNConv_ogbn_arxiv"
    # )
    # directory_path = os.getcwd()+"/log_files/experiment(42,)_node_prop_pred_ds_ogbn-arxiv_type_PygNodePropPredDataset_layers_4_Model_TransformerConv/"
    # process_seed_directories(
    #     seed_directories, directory_path, name="TransformerConv_ogbn_arxiv"
    # )
    # directory_path = os.getcwd()+"/log_files/experiment(43,)_node_prop_pred_ds_ogbn-arxiv_type_PygNodePropPredDataset_layers_4_Model_GATConv/"
    # process_seed_directories(
    #     seed_directories, directory_path, name="GATConv_ogbn_arxiv"
    # )