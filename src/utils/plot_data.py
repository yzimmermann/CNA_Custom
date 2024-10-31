import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE

from .metrics import match_ratio


def plot_heatmap(path, save=False, name_to_store="stored_heatmap"):
    """
    Plot a heatmap from a CSV file and optionally save it.

    Parameters:
    - path (str): The path to the CSV file containing the data.
    - save (bool): Whether to save the plot (default is False).
    - name_to_store (str): The name to use when saving the plot (default is
    "stored_heatmap").

    """
    # Set Seaborn theme and style
    sns.set_theme(style="whitegrid")

    # Read data from CSV file
    loaded_df = pd.read_csv(path, index_col=0)

    # Plot the heatmap
    hm = sns.heatmap(
        loaded_df,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        vmin=-0.2,
        vmax=1,
        cmap="Blues",
    )

    # Add x and y-axis titles
    plt.xlabel("Model Layer")
    plt.ylabel("Model")

    # Adjust plot layout
    figure = hm.get_figure()
    figure.subplots_adjust(left=0.3)

    # Save the plot if requested
    if save:
        plt.savefig(f"{name_to_store}.pdf", dpi=600)

    # Display the plot
    plt.show()


def representaion_plot(model: torch.nn.Module, G, title: str, palette=None):
    """
    Plot representations with predicted and true labels.

    Parameters:
    - model (torch.nn.Module): The neural network model.
    - G: The graph or data structure containing features and labels.
    - title (str): The title for the plot.
    - palette: The color palette for labels (default is None, uses default Seaborn
    palette).

    Returns:
    - Tuple: The predicted logits and t-SNE representations.

    """
    # Set Seaborn theme
    sns.set_theme(style="darkgrid")

    # Representations with t-SNE algorithm
    with torch.no_grad():
        h, logits = model(G.x, G.edge_index)
    representations = TSNE().fit_transform(h.cpu().numpy())

    # Set default palette if not provided
    if palette is None:
        palette = sns.color_palette("hls", len(set(G.y)))

    # Plot 2-D representations with true labels and predictions
    fig, ax = plt.subplots(1, 2, figsize=(24, 8))
    fig.suptitle(title, fontsize=20)

    # Subplot for predicted labels
    sns.scatterplot(
        x=representations[:, 0],
        y=representations[:, 1],
        hue=logits.argmax(dim=1).cpu(),
        legend="full",
        palette=palette,
        ax=ax[0],
    ).set(title="Predicted labels")

    # Subplot for true labels
    sns.scatterplot(
        x=representations[:, 0],
        y=representations[:, 1],
        hue=G.y,
        legend="full",
        palette=palette,
        ax=ax[1],
    ).set(title="True labels")

    plt.show()
    return logits, representations


def tsne_plot_with_centers(
    model: torch.nn.Module, layer, dataset, data, conv, num_classes
):
    fontsize_ = 38
    # Representing the representations with t-SNE algorithm
    G = data
    sns.set(style="whitegrid")
    palette = sns.color_palette("hls", num_classes)

    with torch.no_grad():
        h, logits = model(G.x, G.edge_index)
    representations = TSNE().fit_transform(h.cpu().numpy())

    match_ratio_ = match_ratio(logits, G.y)

    print(f"Layer: {layer}, Dataset: {dataset}, Conv Type: {conv}")
    print(f"Match Ratio: {match_ratio_:.4f}")

    cluster_centers_predicted = [
        np.mean(representations[logits.argmax(dim=1).cpu() == i], axis=0)
        for i in range(num_classes)
    ]

    # Plot the 2-D representations, both with true labels and predictions
    fig, ax = plt.subplots(1, 2, figsize=(24, 8))
    title_ = f"Layer: {layer}, Dataset: {dataset}, Conv Type: {conv}, Match Ratio: {match_ratio_*100:.2f}%"
    fig.suptitle(title_, fontsize=fontsize_)

    plot = sns.scatterplot(
        x=representations[:, 0],
        y=representations[:, 1],
        hue=logits.argmax(dim=1).cpu(),
        legend="full",
        palette=palette,
        ax=ax[0],
    )
    plot.set_title("Predicted Labels", fontsize=fontsize_)

    # Add markers for cluster centers
    for center, color in zip(cluster_centers_predicted, palette):
        ax[0].scatter(
            center[0],
            center[1],
            marker="X",
            s=200,
            c=[color],
            edgecolors="black",
            linewidths=2,
        )

    cluster_centers_true = [
        np.mean(representations[G.y.cpu() == i], axis=0) for i in range(num_classes)
    ]

    plot = sns.scatterplot(
        x=representations[:, 0],
        y=representations[:, 1],
        hue=G.y.cpu(),
        legend="full",
        palette=palette,
        ax=ax[1],
    )
    plot.set_title("True Labels", fontsize=fontsize_)

    # Add markers for cluster centers
    for center, color in zip(cluster_centers_true, palette):
        ax[1].scatter(
            center[0],
            center[1],
            marker="X",
            s=200,
            c=[color],
            edgecolors="black",
            linewidths=2,
        )

    print(plot)
    # Save the plot as a PDF with appropriate name
    plt.tight_layout()
    fig = plot.get_figure()
    pdf_filename = f"our_solution_{layer}_{dataset}_{conv}_plot.pdf"
    fig.savefig(pdf_filename, dpi=600)
    # plt.show()

    plt.close()

    return logits, representations
