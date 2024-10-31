import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(style="whitegrid")


def read_and_plot_data_cora_all_metrics_subsets():
    """Reads and plots the data for Cora with GCNConvolution for all metrics and subsets.

    The data is read from the JSON files and plotted using Seaborn.
    The mean and standard deviation of the data is also plotted.

    Returns:
        None
    """
    with open(
        "results_json/data_default_solution_without_cluster_gcn_cora.json", "r"
    ) as json_file:
        data1 = json.load(json_file)
    with open(
        "results_json/result_data_our_solution_default_subsets_gcn_cora.json", "r"
    ) as json_file:
        data_default_our = json.load(json_file)
    with open("results_json/result_data_50_25_25.json", "r") as json_file:
        data2 = json.load(json_file)
    with open("results_json/result_data_70_15_15.json", "r") as json_file:
        data3 = json.load(json_file)
    with open("results_json/result_data_80_10_10.json", "r") as json_file:
        data4 = json.load(json_file)

    value_to_filter = 128
    data_default = [entry for entry in data1 if entry.get("layers") != value_to_filter]
    data0 = [
        entry for entry in data_default_our if entry.get("layers") != value_to_filter
    ]

    df = pd.DataFrame(data_default)
    df0 = pd.DataFrame(data0)
    df2 = pd.DataFrame(data2)
    df3 = pd.DataFrame(data3)
    df4 = pd.DataFrame(data4)

    metrics = ["max_test_accuracy", "MAD", "MADC", "DE"]

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    for i, metric in enumerate(metrics):
        ax = axs[i // 2, i % 2]
        if metric == "max_test_accuracy":
            metric_ = "Test Accuracy"
        else:
            metric_ = metric

        plot = sns.lineplot(
            x="layers", y=metric, data=df, palette="rocket", label="ReLU", ax=ax
        )
        plot = sns.lineplot(
            x="layers",
            y=metric,
            data=df2,
            palette="rocket",
            label="NCA 50-25-25",
            ax=ax,
        )
        plot = sns.lineplot(
            x="layers",
            y=metric,
            data=df3,
            palette="rocket",
            label="NCA 70-15-15",
            ax=ax,
        )
        plot = sns.lineplot(
            x="layers",
            y=metric,
            data=df4,
            palette="rocket",
            label="NCA 80-10-10",
            ax=ax,
        )
        plot = sns.lineplot(
            x="layers", y=metric, data=df0, palette="rocket", label="NCA default", ax=ax
        )

        # mean and standard deviation
        mean_values = df.groupby("layers")[metric].mean()
        std_values = df.groupby("layers")[metric].std()
        mean_values2 = df2.groupby("layers")[metric].mean()
        std_values2 = df2.groupby("layers")[metric].std()
        mean_values3 = df3.groupby("layers")[metric].mean()
        std_values3 = df3.groupby("layers")[metric].std()
        mean_values4 = df4.groupby("layers")[metric].mean()
        std_values4 = df4.groupby("layers")[metric].std()
        mean_values0 = df0.groupby("layers")[metric].mean()
        std_values0 = df0.groupby("layers")[metric].std()
        # Plot the standard deviation
        ax.fill_between(
            mean_values.index,
            mean_values - 1.96 * std_values,
            mean_values + 1.96 * std_values,
            alpha=0.2,
        )
        ax.fill_between(
            mean_values2.index,
            mean_values2 - 1.96 * std_values2,
            mean_values2 + 1.96 * std_values2,
            alpha=0.2,
        )
        ax.fill_between(
            mean_values3.index,
            mean_values3 - 1.96 * std_values3,
            mean_values3 + 1.96 * std_values3,
            alpha=0.2,
        )
        ax.fill_between(
            mean_values4.index,
            mean_values4 - 1.96 * std_values4,
            mean_values4 + 1.96 * std_values4,
            alpha=0.2,
        )
        ax.fill_between(
            mean_values0.index,
            mean_values0 - 1.96 * std_values0,
            mean_values0 + 1.96 * std_values0,
            alpha=0.2,
        )

        ax.set_title(metric_, fontsize=20)
        ax.set_xlabel("Layers", fontsize=18)
        ax.set_ylabel("Values", fontsize=18)
        ax.legend()
        # plot.set_xticks([2, 4, 8, 16, 32, 64, 96, 128])
        plot.set_xticks([2, 4, 8, 16, 32, 64, 96])

    plt.suptitle("(Metrics vs. Layers) for Cora with GCNConvolution", fontsize=22)

    plt.tight_layout()

    fig = plot.get_figure()
    fig.savefig(
        "Metrics_vs_Layers_for_Cora_with_GCNConvolution_all_solution.pdf", dpi=600
    )
    plt.show()


def read_and_plot_data_citeseer():
    """Reads and plots the data for CiteSeer with GCNConvolution for two metrics and one subsets.

    The data is read from the JSON files and plotted using Seaborn. The mean and standard deviation of the data is also plotted.

    Returns:
        None
    """
    with open("results_json/result_data_relu_default_citeseer.json", "r") as json_file:
        data1 = json.load(json_file)
    data1 = [entry for entry in data1 if entry.get("layers") != 96]
    with open(
        "result_data_citatoinfull_CiteSeer_cluster_80_10_10.json", "r"
    ) as json_file:
        data2 = json.load(json_file)

    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    metrics = ["max_test_accuracy", "MADC"]
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    for i, metric in enumerate(metrics):
        ax = axs[i]
        if metric == "max_test_accuracy":
            metric_ = "Test Accuracy"
        else:
            metric_ = metric
        plot = sns.lineplot(
            x="layers", y=metric, data=df1, palette="rocket", label="ReLU", ax=ax
        )

        plot = sns.lineplot(
            x="layers",
            y=metric,
            data=df2,
            palette="rocket",
            label="NCA 80-10-10",
            ax=ax,
        )

        mean_values = df1.groupby("layers")[metric].mean()
        std_values = df1.groupby("layers")[metric].std()

        # Calculate mean and standard deviation for each layer
        mean_values2 = df2.groupby("layers")[metric].mean()
        std_values2 = df2.groupby("layers")[metric].std()

        # Plot the standard deviation
        ax.fill_between(
            mean_values.index,
            mean_values - 1.96 * std_values,
            mean_values + 1.96 * std_values,
            alpha=0.2,
        )

        # Plot the standard deviation
        ax.fill_between(
            mean_values2.index,
            mean_values2 - 1.96 * std_values2,
            mean_values2 + 1.96 * std_values2,
            alpha=0.2,
        )

        ax.set_title(metric_, fontsize=20)
        ax.set_xlabel("Layers", fontsize=18)
        ax.set_ylabel("Values", fontsize=18)
        ax.legend()
        plot.set_xticks([2, 4, 8, 16, 32, 64])

    plt.suptitle("(Metrics vs. Layers) for CiteSeer with GATConvolution", fontsize=22)
    plt.tight_layout()

    fig = plot.get_figure()
    fig.savefig(
        "Metrics_vs_Layers_for_CiteSeer_with_GATCONV_solution.pdf", dpi=600
    )
    plt.show()


if __name__ == "__main__":
    #read_and_plot_data_cora_all_metrics_subsets()
    read_and_plot_data_citeseer()
