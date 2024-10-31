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
                print(parts)
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
        print(os.path.isdir(directory))
        print(directory)
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
        print(f"Result Table for Seed Directories ({directory}):")
        print(result_table)
        print(f"Config Data for Seed Directories ({directory}):")

        list_results.append(result_table)

    # Save the list of dictionaries as JSON
    dict_of_highest_acc = get_highest_accuracy(list_dicts)
    with open(f"results_{name}.json", "w") as json_file:
        json.dump(dict_of_highest_acc, json_file, indent=2)


if __name__ == "__main__":
    seed_directories = ["seed0"]
    directory_path = f"PATH_TO_DIRECTORY"
    process_seed_directories(
        seed_directories, directory_path, name="node_prop_pred_ds_ogbn_arxiv"
    )
