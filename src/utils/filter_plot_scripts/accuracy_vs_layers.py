import json

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

sns.set(style="whitegrid")


def plot_accuracy_vs_layers(datasets, save_filename=None):
    num_layers = [2, 4, 8, 16, 32]

    sns.set(style="whitegrid")
    fig, axs = plt.subplots(2, 2, figsize=(22, 15))

    color_palette = sns.color_palette("rocket", n_colors=len(datasets["Cora"]))
    k = 0
    for i, (title, data) in enumerate(datasets.items()):
        if i == 2:
            k = 1
            i = 0
        ax = axs[i][k]
        ax.set_title(title, fontsize=22)
        for j, (layer_type, values) in enumerate(data.items()):
            linestyle = "--" if "with activations" in layer_type else "-"
            plot = sns.lineplot(
                x=num_layers,
                y=values,
                linestyle=linestyle,
                color=color_palette[j],
                linewidth=2.0,
                ax=ax,
            )
        plot.set_xticks(num_layers)
        plot.set_xlabel("Number of Layers", fontsize=18)
        plot.set_ylabel("Test Accuracy", fontsize=18)

    axs[1, 1].axis("off")

    # Set legends using Seaborn
    a = Line2D(
        [], [], color="#35193e", linestyle="-", label="GATConv without activations"
    )
    b = Line2D(
        [], [], color="#35193e", linestyle="--", label="GATConv with activations"
    )
    c = Line2D(
        [], [], color="#701f57", linestyle="-", label="GCNConv without activations"
    )
    d = Line2D(
        [], [], color="#701f57", linestyle="--", label="GCNConv with activations"
    )
    e = Line2D(
        [], [], color="#e13342", linestyle="-", label="SAGEConv without activations"
    )
    f = Line2D(
        [], [], color="#e13342", linestyle="--", label="SAGEConv with activations"
    )
    g = Line2D(
        [],
        [],
        color="#f37651",
        linestyle="-",
        label="TransformerConv without activations",
    )
    h = Line2D(
        [],
        [],
        color="#f37651",
        linestyle="--",
        label="TransformerConv with activations",
    )
    axs[1, 1].legend(
        handles=[a, b, c, d, e, f, g, h], bbox_to_anchor=(0.5, 0.5), fontsize=14
    )
    fig.suptitle(f"Test Accuracy vs. Number of Layers", fontsize=24)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_filename:
        plt.savefig(save_filename, dpi=600)

    # Show the plot
    plt.show()


if __name__ == "__main__":
    # Specify the file path to save the data
    json_file_path = "results_datasets_with_without_activation.json"

    with open(json_file_path, "r") as json_file:
        loaded_datasets = json.load(json_file)

    print("Datasets loaded successfully:")
    print(loaded_datasets)
    print(f"Datasets saved to {json_file_path}")

    plot_accuracy_vs_layers(
        loaded_datasets, "with_without_activation_accuracy_plot.pdf"
    )
