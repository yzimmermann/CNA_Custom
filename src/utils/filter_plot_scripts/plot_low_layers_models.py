import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set Seaborn style
sns.set(style="whitegrid")

def read_accuracy_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Use regular expression to extract test and validation accuracy
    pattern = r'Test Accuracy: (\d+\.\d+) - Validation Accuracy: (\d+\.\d+)'
    matches = re.findall(pattern, content)

    # Extracted lists
    test_accuracy_list = [float(match[0]) for match in matches]
    validation_accuracy_list = [float(match[1]) for match in matches]

    return test_accuracy_list, validation_accuracy_list

def plot_accuracies(directory):
    test_accuracies = []
    validation_accuracies = []
    test_acc_max_list = []
    val_acc_max_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                test_accuracy, validation_accuracy = read_accuracy_file(file_path)
                test_accuracies.append(test_accuracy)
                validation_accuracies.append(validation_accuracy)
                test_acc_max_list.append(max(test_accuracy))
                val_acc_max_list.append(max(validation_accuracy))


    # Convert the list of lists to a NumPy array
    data_np = np.array(test_accuracies)
    data2_np = np.array(validation_accuracies)

    # Calculate mean and std for each epoch
    mean_values = np.mean(data_np, axis=0)
    std_values = np.std(data_np, axis=0)

    # Calculate mean and std for each epoch
    mean_values2 = np.mean(data2_np, axis=0)
    std_values2 = np.std(data2_np, axis=0)

    # Calculate mean of maximum test accuracies and std
    #max_test_accuracies = np.max(data_np, axis=1)
    print(len(mean_values))
    mean_max_test_accuracy = np.mean(test_acc_max_list)
    std_max_test_accuracy = np.std(test_acc_max_list)

    # Plotting
    epochs = range(1, len(mean_values) + 1)

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=epochs, y=mean_values, palette="rocket", label="Test Accuracy")
    sns.lineplot(x=epochs, y=mean_values2, palette="rocket", label="Validation Accuracy")

    plt.fill_between(epochs, mean_values - std_values, mean_values + std_values, alpha=0.2)
    plt.fill_between(epochs, mean_values2 - std_values2, mean_values2 + std_values2, alpha=0.2)

    max_annotation_text = f'{(mean_max_test_accuracy)*100:.2f}±{(std_max_test_accuracy)*100:.2f}%'
    plt.annotate(max_annotation_text, xy=(len(mean_values) - 12, 0.45),
                 fontsize=10, color='black')

    # Print mean of maximum test accuracies and std
    print(f'Mean of Maximum Test Accuracies: {(mean_max_test_accuracy)*100:.2f}% ± {(std_max_test_accuracy)*100:.2f}%')

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Dataset: {dataset}, Type: {data_type}, Layers: {layers}, Model: {conv_type}")
    plt.grid(True)
    plt.legend()

    plt.savefig(f"{directory}.pdf", dpi=600, bbox_inches='tight')
    plt.show()

# Example usage
# Amazon_2_Computers_layers_TRANSFORMERCONV
# Amazon_4_Photo_layers_TRANSFORMERCONV
# CiteSeer_4_layers_gatconv_citationfull
# cora_4_layers_SAGEConv
# DBLP_4_layers_TransformerConv
# PubMed_2_layers_transformerconv
#directory_path = 'experiment_corafull_citatoinfull_TRANSFORMERCONV_80_10_10'
# experiment_ds_Squirrel_type_WikipediaNetwork_layers_2_Model_DirGCNConv_split_80_10_10
# experiment_ds_Chameleon_type_WikipediaNetwork_layers_2_Model_DirGCNConv_split_80_10_10
#experiment_ds_Wisconsin_type_WebKB_layers_2_Model_DirSageConv_split_80_10_10
# experiment_ds_Wisconsin_type_WebKB_layers_4_Model_TransformerConv_split_80_10_10
# experiment_ds_Wisconsin_type_WebKB_layers_2_Model_TransformerConv_split_80_10_10
# experiment_ds_Texas_type_WebKB_layers_2_Model_SAGEConv_split_70_15_15
directory_path = "experiment_ds_Texas_type_WebKB_layers_2_Model_SAGEConv_split_70_15_15"

data_type = "WebKB"
dataset = "Texas"
conv_type = "SAGEConv"
layers = 2
plot_accuracies(directory_path)
