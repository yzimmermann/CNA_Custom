import os
import json
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from matplotlib.lines import Line2D
import numpy as np
# Set font family and serif
# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = 'Times New Roman'
fontsize_ = 35
# sns.set_context('paper')

def extract_values_from_line(line):
    match = re.search(r'Epochs 400 - Max Test Accuracy: ([0-9]+\.[0-9]+) - .* - DE: ([0-9]+\.[0-9]+) -.*', line)
    if match:
        return float(match.group(1)), float(match.group(2))
    else:
        print(f"Line did not match pattern: {line}")
    return None, None

def parse_experiment_code(code):
    mappings_model_type = {
        '42': 'CNA', '41': 'ReLU', '40': 'Linear', '39': 'CNA', '38': 'ReLU', '37': 'Linear',
    }
    mappings_dataset = {'02': 'CiteSeer', '01': 'Cora'}
    mappings_architecture = {
        '01': 'GATConv', '02': 'GCNConv', '03': 'SAGEConv', '04': 'TransformerConv'
    }
    layer_mappings = {'01': '1', '02': '2', '04': '4', '08': '8', '16': '16', '32': '32', '64': '64', '96': '96'}

    model_type = mappings_model_type.get(code[:2], 'Unknown')
    dataset = mappings_dataset.get(code[2:4], 'Unknown')
    architecture = mappings_architecture.get(code[4:6], 'Unknown')
    layers = layer_mappings.get(code[6:], 'Unknown')
    
    return dataset, f"{architecture}_{model_type}", layers

def find_txt_file(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            return os.path.join(folder_path, file_name)
    return None

def process_json_files(base_path):
    test_acc = {'Cora': {}, 'CiteSeer': {}}
    de = {'Cora': {}, 'CiteSeer': {}}
    
    for folder in os.listdir(base_path):
        if folder.startswith('KK_experiment'):
            experiment_code = folder.split('_')[1][10:]
            dataset, model_type, layers = parse_experiment_code(experiment_code)
            
            if model_type not in test_acc[dataset]:
                test_acc[dataset][model_type] = {}
                de[dataset][model_type] = {}
            
            if layers not in test_acc[dataset][model_type]:
                test_acc[dataset][model_type][layers] = []
                de[dataset][model_type][layers] = []
            
            for seed_folder in os.listdir(os.path.join(base_path, folder)):
                seed_folder_path = os.path.join(base_path, folder, seed_folder)
                txt_file_path = find_txt_file(seed_folder_path)

                if txt_file_path and os.path.isfile(txt_file_path):
                    with open(txt_file_path, 'r') as f:
                        for line in f:
                            if 'Max Test Accuracy' in line or 'Epochs 400' in line:
                                max_test_acc, de_value = extract_values_from_line(line)
                                if max_test_acc is not None and de_value is not None:
                                    test_acc[dataset][model_type][layers].append(max_test_acc)
                                    de[dataset][model_type][layers].append(de_value)
                                else:
                                    print(f"Failed to extract values from line: {line}")
                else:
                    print(f'{txt_file_path} does not exists!')

    return test_acc, de

def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def plot_and_save_figure(test_acc, de):
    # Define the datasets
    datasets = ['Cora', 'CiteSeer']

    # Create a figure with 4 subplots
    # fig, axs = plt.subplots(4, 2, figsize=(12, 12))
    # fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))

    # Define the legend lines
    legend_lines = [
        Line2D([], [], color="orange", linestyle="-", label="GATConv + CNA"),
        Line2D([], [], color="orange", linestyle="-.", label="GATConv (linear)"),
        Line2D([], [], color="orange", linestyle="dotted", label="GATConv"),
        Line2D([], [], color="blue", linestyle="-", label="GCNConv + CNA"),
        Line2D([], [], color="blue", linestyle="-.", label="GCNConv (linear)"),
        Line2D([], [], color="blue", linestyle="dotted", label="GCNConv"),
        Line2D([], [], color="green", linestyle="-", label="SAGEConv + CNA"),
        Line2D([], [], color="green", linestyle="-.", label="SAGEConv (linear)"),
        Line2D([], [], color="green", linestyle="dotted", label="SAGEConv"),
        Line2D([], [], color="brown", linestyle="-", label="TransformerConv + CNA"),
        Line2D([], [], color="brown", linestyle="-.", label="TransformerConv (linear)"),
        Line2D([], [], color="brown", linestyle="dotted", label="TransformerConv")
    ]

    # Define the colors and styles for each model type
    model_styles = {
        "GATConv_CNA": ("orange", "-"),
        "GATConv_Linear": ("orange", "-."),
        "GATConv_ReLU": ("orange", "dotted"),
        "GCNConv_CNA": ("blue", "-"),
        "GCNConv_Linear": ("blue", "-."),
        "GCNConv_ReLU": ("blue", "dotted"),
        "SAGEConv_CNA": ("green", "-"),
        "SAGEConv_Linear": ("green", "-."),
        "SAGEConv_ReLU": ("green", "dotted"),
        "TransformerConv_CNA": ("brown", "-"),
        "TransformerConv_Linear": ("brown", "-."),
        "TransformerConv_ReLU": ("brown", "dotted"),
    }

    # Plot test_acc for each dataset
    # for i, dataset in enumerate(datasets):
    #     # ax = axs[0, i]
    #     ax = axs[i]
    #     for model_type in test_acc[dataset]:
    #         layers = sorted([int(layer) for layer in test_acc[dataset][model_type] if layer != '1'])
    #         means = np.array([np.mean(test_acc[dataset][model_type][str(layer)]) for layer in layers])
    #         stds = np.array([np.std(test_acc[dataset][model_type][str(layer)]) for layer in layers])
    #         color, linestyle = model_styles[model_type]
    #         ax.plot(layers, means, label=model_type, color=color, linestyle=linestyle)
    #         ax.fill_between(layers, means - stds, means + stds, alpha=0.15, color=color)
    #     ax.set_xticks(layers)
    #     ax.set_xticklabels(layers)
    #     ax.set_title(dataset)
    #     ax.set_xlim(2, 96)  # Set x-axis limits
    #     ax.set_ylim(0.0, 1.0)  # Set y-axis limits
    #     ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    #     # Add grid lines
    #     ax.grid(True, which='both', axis='y', linestyle='-', color='gray', alpha=0.2)  
    #     if i == 0:
    #         ax.set_ylabel('Accuracy')

    # Plot Dirichlet Energy (DE) for each dataset
    for i, dataset in enumerate(datasets):
        cna_model_types = [model_type for model_type in de[dataset] if 'CNA' in model_type]
        linear_model_types = [model_type for model_type in de[dataset] if 'Linear' in model_type]
        relu_model_types = [model_type for model_type in de[dataset] if 'ReLU' in model_type]

        for j, model_types in enumerate([cna_model_types, linear_model_types, relu_model_types]):
            # ax = axs[j+1, i]  # Plot in the last three rows
            ax = axs[j, i]  # Plot in the last three rows
            for model_type in model_types:
                layers = sorted([int(layer) for layer in de[dataset][model_type] if layer != '1'])
                means = np.array([np.mean(de[dataset][model_type][str(layer)]) for layer in layers])
                stds = np.array([np.std(de[dataset][model_type][str(layer)]) for layer in layers])
                color, linestyle = model_styles[model_type]
                ax.plot(layers, means, label=model_type, color=color, linestyle=linestyle)
                # ax.fill_between(layers, means - stds, means + stds, alpha=0.15, color=color)
            ax.set_xticks(layers)
            ax.set_xticklabels(layers)
            ax.set_xlim(2, 96)
            ax.set_yscale('log')
            # Add grid horizontal lines
            ax.grid(True, which='both', axis='y', linestyle='-', color='gray', alpha=0.2)  
            if i == 0:
                if j==0:
                    ax.set_ylabel('CNA Dirichlet Energy')
                if j==1:
                    ax.set_ylabel('Linear Dirichlet Energy')
                if j==2:
                    ax.set_ylabel('ReLU Dirichlet Energy')
            if (j==0 and i==0) or (j==0 and i==1):
                ax.set_ylim(1e4,3e5)
            if j==1 and i==0:
                ax.set_ylim(1e2,1e8)
            if j==1 and i==1:
                ax.set_ylim(1e2,1e8)
            if j==2 and i==0:
                ax.set_ylim(1e-5,1e8)
            if j==2 and i==1:
                ax.set_ylim(1e-5,1e8)
                                      

    # Add legend
    # fig.legend(handles=legend_lines, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4)
    fig.legend(handles=legend_lines, loc='upper center', bbox_to_anchor=(0, 0.95, 1, 0.1), 
               ncol=4, mode="expand", bbox_transform=fig.transFigure)
    
    # Layout so plots do not overlap
    fig.tight_layout(pad=2.0)

    # Save the figure
    plt.savefig('dirichlet_energy.pdf', dpi=600, bbox_inches='tight')
    print("Done ploting!")

def main():
    # Specify the base path to look for folders
    base_path = os.path.join('..', '..', 'log_files/figure4/')
    test_acc, de = process_json_files(base_path)
    save_to_json(test_acc, 'test_acc.json')
    save_to_json(de, 'de.json')
    plot_and_save_figure(test_acc, de)
    return test_acc, de

# Extract the data and generate plots
if __name__ == '__main__':
    test_acc, de = main()
    plot_and_save_figure(test_acc, de)
