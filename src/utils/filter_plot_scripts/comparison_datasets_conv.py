import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(style="whitegrid")


def comparison_test_acc_datasets_conv(json_file_path):
    with open(json_file_path, "r") as json_file:
        loaded_datasets = json.load(json_file)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
    j = 0
    for i, (dataset, conv_types) in enumerate(loaded_datasets.items()):
        data = pd.DataFrame(conv_types)
        # data.index = [2, 4, 8, 16, 32, 64]
        data.index = [1, 2, 4, 8, 16, 32, 64, 96, 128]
        if i == 2:
            j = 1
            i = 0
        plot = sns.lineplot(data=data, palette="rocket", linewidth=2.5, ax=axes[i][j])
        plot.set_xlabel("Layer Number", fontsize=20)
        plot.set_ylabel("Test Accuracy", fontsize=20)
        plot.set_title(dataset, fontsize=22)
        plot.set_xticks(data.index)

    axes[1, 1].axis("off")
    plt.tight_layout()
    fig = plot.get_figure()
    fig.savefig("comparison_test_acc_datasets_conv.pdf", dpi=600)
    plt.show()


if __name__ == "__main__":
    json_file_path = "data_comparison_test_acc_datasets_conv.json"
    comparison_test_acc_datasets_conv(json_file_path)
