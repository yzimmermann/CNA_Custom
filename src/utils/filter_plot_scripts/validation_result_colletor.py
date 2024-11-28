import os
import sys
import json
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def extract_max_validation_and_test_accuracy(file_path):
    """
    Extracts the maximum validation accuracy and the corresponding test accuracy from a text file.

    Parameters:
    - file_path (str): The path to the text file.

    Returns:
    - dict: A dictionary containing max validation accuracy and its corresponding test accuracy.
    """
    max_val_acc = 0.0
    corresponding_test_acc = 0.0

    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            if "Validation Accuracy" in line and "Test Accuracy" in line and not 'Max Validation Accuracy' in line:
                print(line)
                parts = line.split(" - ")
                print(parts)
                val_acc = float(parts[-1].split(": ")[1])
                print(val_acc)
                test_acc = float(parts[-2].split(": ")[1])
                print(test_acc)

                if val_acc > max_val_acc:
                    max_val_acc = val_acc
                    corresponding_test_acc = test_acc

    return {'max_validation_acc': max_val_acc, 'test_acc': corresponding_test_acc}


def process_directory_for_max_accuracies(base_directory):
    """
    Processes a directory recursively, looking for .txt files to extract accuracy information.

    Parameters:
    - base_directory (str): The path to the base directory to search.

    Returns:
    - list: A list of dictionaries containing max validation accuracy and corresponding test accuracy for each file.
    """
    results = []

    for root, _, files in os.walk(base_directory):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                accuracy_dict = extract_max_validation_and_test_accuracy(file_path)
                accuracy_dict['file_path'] = file_path  # Add the file path for reference
                results.append(accuracy_dict)

    return results


def save_results_to_json(data, output_file):
    """
    Saves the results to a JSON file.

    Parameters:
    - data (list): A list of dictionaries to save.
    - output_file (str): The path to the output JSON file.

    Returns:
    - None
    """
    with open(output_file, "w") as json_file:
        json.dump(data, json_file, indent=2)


if __name__ == "__main__":
    base_directory = "log_files"
    output_json = "results_max_accuracies.json"

    max_accuracies = process_directory_for_max_accuracies(base_directory)

    for result in max_accuracies:
        print(result)

    save_results_to_json(max_accuracies, output_json)
    print(f"Results saved to {output_json}")
