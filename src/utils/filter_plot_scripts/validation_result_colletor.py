import os
import sys
import json
import numpy as np
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def extract_max_validation_and_test_accuracy(file_path):
    """
    Extracts the maximum validation accuracy, corresponding test accuracy, and DE if present
    """
    max_val_acc = 0.0
    corresponding_test_acc = 0.0
    corresponding_de = None
    has_de = False

    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            # Try to match both formats
            if "Max Validation Accuracy" in line and "Test Accuracy" in line:
                parts = line.split(" - ")
                
                # Extract validation and test accuracies
                for part in parts:
                    if "Max Validation Accuracy" in part:
                        val_acc = float(part.split(": ")[1])
                    elif "Test Accuracy" in part:
                        test_acc = float(part.split(": ")[1])
                    elif "DE:" in part:
                        has_de = True
                        de = float(part.split(": ")[1])
                
                if val_acc > max_val_acc:
                    max_val_acc = val_acc
                    corresponding_test_acc = test_acc
                    if has_de:
                        corresponding_de = de

    result = {
        'max_validation_acc': max_val_acc,
        'test_acc': corresponding_test_acc
    }
    if has_de:
        result['de'] = corresponding_de
    
    return result

def process_directory_for_max_accuracies(base_directory):
    """
    Processes directory recursively for accuracy information
    """
    results = []
    for root, _, files in os.walk(base_directory):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                accuracy_dict = extract_max_validation_and_test_accuracy(file_path)
                accuracy_dict['file_path'] = file_path
                results.append(accuracy_dict)
    return results

def process_accuracies(data):
    """
    Process and print statistics from the collected data
    """
    # Group results by experiment
    experiments = {}
    has_de = 'de' in data[0] if data else False
    
    for entry in data:
        file_path = entry['file_path']
        exp_num = file_path.split('experiment')[1].split('_')[0]
        
        if exp_num not in experiments:
            experiments[exp_num] = {
                'val_acc': [],
                'test_acc': [],
                'de': [] if has_de else None,
                'seeds': 0
            }
        
        experiments[exp_num]['val_acc'].append(entry['max_validation_acc'])
        experiments[exp_num]['test_acc'].append(entry['test_acc'])
        if has_de:
            experiments[exp_num]['de'].append(entry['de'])
        experiments[exp_num]['seeds'] += 1

    # Print header
    if has_de:
        print(f"{'experiment':<12}{'seeds':<8}{'mean±std validation acc':<24}"
              f"{'mean±std test acc':<24}{'mean±std Dirichlet Energy (DE)':<24}")
        print("-" * 92)
    else:
        print(f"{'experiment':<12}{'seeds':<8}{'mean±std validation acc':<24}{'mean±std test acc':<24}")
        print("-" * 68)
    
    # Calculate and print statistics
    for exp_num in sorted(experiments.keys()):
        val_accs = np.array(experiments[exp_num]['val_acc'])
        test_accs = np.array(experiments[exp_num]['test_acc'])
        seeds = experiments[exp_num]['seeds']
        
        if has_de:
            de_values = np.array(experiments[exp_num]['de'])
            print(f"{exp_num:<12}{seeds:<8}{val_accs.mean()*100:.2f}±{val_accs.std()*100:.2f}%{' '*13}"
                  f"{test_accs.mean()*100:.2f}±{test_accs.std()*100:.2f}%"
                  f"{de_values.mean():.4f}±{de_values.std():.4f}")
        else:
            print(f"{exp_num:<12}{seeds:<8}{val_accs.mean()*100:.2f}±{val_accs.std()*100:.2f}%{' '*13}"
                  f"{test_accs.mean()*100:.2f}±{test_accs.std()*100:.2f}%")

    return experiments

def save_results_to_json(data, output_file):
    """
    Saves results to JSON file
    """
    with open(output_file, "w") as json_file:
        json.dump(data, json_file, indent=2)

if __name__ == "__main__":
    base_directory = "log_files/ablation_cora/"
    output_json = "results_max_accuracies.json"
    
    # Process directory and collect results
    max_accuracies = process_directory_for_max_accuracies(base_directory)
    
    # Save raw results to JSON
    save_results_to_json(max_accuracies, output_json)
    print(f"\nRaw results saved to {output_json}")
    
    # Process and print statistics
    print("\nCalculated Statistics:")
    statistics = process_accuracies(max_accuracies)
    
    # Save processed statistics to JSON
    save_results_to_json(statistics, "extracted_statistics.json")
    print(f"\nProcessed statistics saved to extracted_statistics.json")
