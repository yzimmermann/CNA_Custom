import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(style="whitegrid")


def plot_data_comparison(rpm_data, relu_data, rwam_data):
    df_prm = pd.DataFrame(rpm_data)
    df_relu = pd.DataFrame(relu_data)
    df_rwam = pd.DataFrame(rwam_data)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

    # Plot for PRM vs. RELU
    ax1 = axes[0]
    plot = sns.lineplot(
        x="Layer",
        y="PRM",
        data=df_prm,
        label="PRM",
        palette="rocket",
        dashes=False,
        markers=True,
        ax=ax1,
    )
    plot = sns.lineplot(
        x="Layer",
        y="RELU",
        data=df_relu,
        label="RELU",
        palette="rocket",
        dashes=False,
        markers=True,
        ax=ax1,
    )
    ax1.set_xticks([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    ax1.set_xlabel("Number of Layers", fontsize=18)
    ax1.set_ylabel("Test Accuracy", fontsize=18)
    ax1.set_title("PRM vs. RELU, Cora, GCNConvolution", fontsize=20)
    ax1.legend()

    # Plot for RWAM vs. RELU
    ax2 = axes[1]
    plot = sns.lineplot(
        x="Layer",
        y="RWAM",
        data=df_rwam,
        label="RWAM",
        palette="rocket",
        dashes=False,
        markers=True,
        ax=ax2,
    )

    plot = sns.lineplot(
        x="Layer",
        y="RELU",
        data=df_relu,
        label="RELU",
        palette="rocket",
        dashes=False,
        markers=True,
        ax=ax2,
    )
    ax2.set_xticks([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    ax2.set_xlabel("Number of Layers", fontsize=18)
    ax2.set_ylabel("Test Accuracy", fontsize=18)
    ax2.set_title("RWAM vs. RELU, Cora, GCNConvolution", fontsize=20)
    ax2.legend()

    # Adjust layout
    plt.tight_layout()
    fig = plot.get_figure()
    fig.savefig("RPM_RWAM_RELU_comparison.pdf", dpi=600)

    plt.show()


if __name__ == "__main__":
    relu_data_path = "relu_data_2_to_20.json"
    rpm_data_path = "rpm_data_2_to_20.json"
    rwam_data_path = "rwam_data_2_to_20.json"

    with open(relu_data_path, "r") as json_file:
        loaded_relu_data = json.load(json_file)
    with open(rpm_data_path, "r") as json_file:
        loaded_rpm_data = json.load(json_file)

    with open(rwam_data_path, "r") as json_file:
        loaded_rwam_data = json.load(json_file)

    plot_data_comparison(loaded_rpm_data, loaded_relu_data, loaded_rwam_data)
