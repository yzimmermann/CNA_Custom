import warnings

import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from torch_geometric.utils import to_dense_adj

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)


def compute_mad(h, mask):
    """
    The compute_mad function computes the mean average distance (MAD) of a graph.
    The MAD is defined as the average distance between all pairs of nodes in a graph,
    where each node's degree is used to weight its contribution to the overall MAD.
    This function takes two inputs: h and mask. The input h should be an n x d matrix,
    where n is the number of nodes in your graph and d is their dimensionality
    (i.e., how many features they have).
    The input mask should be an n x n matrix that defines which edges are present in
    your network; if there exists an edge between node.

    :param h: Compute the pairwise distances between the nodes in a graph
    :param mask: Mask out the distances between nodes that are not connected
    :return: The mean average distance (mad) of the graph

    """
    dist_arr = pairwise_distances(h, metric="cosine")
    mask_dist = np.multiply(dist_arr, mask)
    divide_arr = (mask_dist != 0).sum(1) + 1e-8
    node_dist = mask_dist.sum(1) / divide_arr
    mad = np.sum(node_dist) / ((node_dist != 0).sum() + 1e-8)

    try:
        mad = round(mad, 3)
    except Exception:
        pass

    return mad


def compute_mad_for_centers(h, G, num_classes):
    """
    Compute the Mean Average Distance (MAD) for cluster centers in a t-SNE representation.

    Parameters:
    - h (torch.Tensor): Hidden features tensor.
    - G (torch_geometric.data.Data): PyTorch Geometric data object.
    - num_classes (int): Number of classes in the dataset

    Returns:
    - float: Mean Average Distance (MAD) for cluster centers.
    """
    representations = TSNE().fit_transform(h.cpu().numpy())
    cluster_centers = [
        np.mean(representations[G.y == i], axis=0) for i in range(num_classes)
    ]
    distances = cdist(cluster_centers, cluster_centers, metric="euclidean")
    mad_for_centers = np.mean(distances)

    try:
        mad_for_centers = round(mad_for_centers, 3)
    except Exception:
        pass

    return mad_for_centers


def compute_GMAD(h):
    """
    Compute the General Mean Average Distance (GMAD) for hidden features.

    Parameters:
    - h (torch.Tensor): Hidden features tensor.

    Returns:
    - float: General Mean Average Distance (GMAD) for hidden features.
    """
    dist_arr = pairwise_distances(h, metric="cosine")
    mean_gen = np.mean(dist_arr)

    try:
        mean_gen = round(mean_gen, 3)
    except Exception:
        pass

    return mean_gen


def compute_mad_gap(in_arr, adj):
    """
    The compute_mad_gap function computes the difference between the median
    absolute deviation of a given array and its adjacency matrix. The function
    takes in an array and an adjacency matrix, then uses these to compute
    the median absolute deviation for both the neighborhood and remote regions
    of each node. It returns a single value, which is equal to
    (median_absolute_deviation(remote) - median_absolute_deviation(neighborhood)).
    This value can be used as a measure of how much more variable one region is
    than another.

    :param in_arr: Pass in the array of values that we want to compute the mad for
    :param adj: Determine whether the mad is computed for
    :return: The difference between the median absolute deviation

    Example:
    >>> madgap = compute_mad_gap(data.x, A)
    """

    mad_neb = compute_mad(in_arr, adj)
    mad_rmt = compute_mad(in_arr, 1 - adj)

    return (mad_rmt - mad_neb).item()


def compute_dirichlet_energy(L, h):
    """
    The dirichlet_energy function computes the Dirichlet energy of a given
    harmonic function. The Dirichlet energy is defined as:

    :param L: Calculate the laplacian matrix
    :param h: Compute the energy of a given configuration
    :return: The dirichlet energy

    """
    return torch.matmul(torch.matmul(h.transpose(0, 1), L), h).squeeze(0).trace().item()


def match_ratio(logits, true_labels):
    # Get the predicted labels
    predicted_labels = logits.argmax(dim=1)

    # Compute the number of correct predictions
    correct_predictions = torch.sum(predicted_labels == true_labels)

    # Compute the total number of samples
    total_samples = true_labels.size(0)

    # Compute the match ratio
    ratio = correct_predictions.item() / total_samples

    # Round the ratio to four decimal places
    rounded_ratio = round(ratio, 4)

    return rounded_ratio
