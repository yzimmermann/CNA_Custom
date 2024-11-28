import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import json
import re
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
# Set font family and serif
plt.rcParams['font.serif'] = 'Times New Roman'

def extract_test_accuracy(base_path):
    test_acc = {'GCNConv': {}, 'GCNConv_CN': {}, 'GCNConv_CNA': {}}

    # Iterate through each folder in base_path
    for folder in os.listdir(base_path):
        if folder.startswith('KK_experiment'):
            # Map the experiment codes to model types
            experiment_code = folder.split('_')[1][10:]
            if experiment_code == '4402':
                model_type = 'GCNConv'
            elif experiment_code == '4604':
                model_type = 'GCNConv_CN'
            elif experiment_code == '4503':
                model_type = 'GCNConv_CNA'
            else:
                continue  # Skip if not a recognized code

            for seed_folder in os.listdir(os.path.join(base_path, folder)):
                seed_folder_path = os.path.join(base_path, folder, seed_folder)
                if seed_folder not in test_acc[model_type]:
                    test_acc[model_type][seed_folder] = []

                for file_name in os.listdir(seed_folder_path):
                    if file_name.endswith('.txt'):
                        txt_file_path = os.path.join(seed_folder_path, file_name)
                        with open(txt_file_path, 'r') as f:
                            for line in f:
                                # Extract Test Accuracy from the line
                                match = re.search(r'Test Accuracy: ([0-9]+\.[0-9]+)', line)
                                if match:
                                    test_accuracy = float(match.group(1))
                                    test_acc[model_type][seed_folder].append(test_accuracy)

    # Save to JSON file
    with open('val_accuracy.json', 'w') as f:
        json.dump(test_acc, f, indent=4)

    return test_acc

def plot_test_accuracy(test_acc):
    # Define the styles for each model type
    model_styles = {
        "GCNConv_CNA": ("-", "CNA"),
        "GCNConv_CN": ("-.", "CN"),
    }

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot test accuracy for each model type
    for model_type, seeds in test_acc.items():
        if model_type == "GCNConv":
            continue  # Skip plotting for GCNConv

        all_accuracies = []
        for seed, accuracies in seeds.items():
            all_accuracies.append(accuracies)
        
        # Calculate mean and standard deviation across seeds
        all_accuracies = np.array(all_accuracies)
        mean_accuracies = np.mean(all_accuracies, axis=0)
        std_accuracies = np.std(all_accuracies, axis=0)
        
        linestyle, label = model_styles[model_type]
        epochs = range(len(mean_accuracies))
        ax.plot(epochs, mean_accuracies, label=label, linestyle=linestyle)
        ax.fill_between(epochs, mean_accuracies - std_accuracies, mean_accuracies + std_accuracies, alpha=0.2)

    # Set x-axis and y-axis labels
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Test Accuracy')

    # Add legend
    ax.legend()

    # Save the figure
    plt.savefig('convergence_analysis.pdf', dpi=600, bbox_inches='tight')

# Main Function
def main():
    base_path = os.path.join('..', '..', 'log_files/figure5/')
    test_acc = extract_test_accuracy(base_path)
    plot_test_accuracy(test_acc)

if __name__ == '__main__':
    main()