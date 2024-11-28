import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import json
from typing import Optional
import fast_pytorch_kmeans as fpk
import numpy as np
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
from utils.model_params import DistanceMetrics, ModelParams, ReclusterOption


class ClusterActivation(nn.Module):
    def __init__(
        self,
        num_clusters: int,
        activation: list,
        mode=DistanceMetrics.EUCLIDEAN,
        recluster_option=ReclusterOption.TH,
        normalize=True,
    ):
        super(ClusterActivation, self).__init__()

        self.num_clusters = num_clusters
        self.activation = activation
        self.normalize = normalize
        if mode == DistanceMetrics.EUCLIDEAN:
            self.mode = "euclidean"
        else:
            self.mode = "cosine"

        self.recluster_option = recluster_option
        self.recluster_threshold = 0.5
        self.recluster_iterations = 20
        self.iteration_count = 0

        self.use_gmm = ModelParams.use_gmm  # Flag to choose between KMeans and GMM
        if self.use_gmm:
            self.gmm = None
        else:
            self.kmeans = fpk.KMeans(n_clusters=self.num_clusters, mode=self.mode)

        self.prev_centroids = None
        self.init_check = None
        self.eps = 1e-5
        self.weight = None
        self.bias = None
        self.reset_params(ModelParams.num_hidden_features[0])

        self.cluster_indices_history = []
        self.cnt = 0

    def reset_params(self, width: int) -> None:
        self.weight = nn.Parameter(torch.empty(width, device=ModelParams.device))
        self.bias = nn.Parameter(torch.empty(width, device=ModelParams.device))
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.init_check is None:
            if self.use_gmm:
                _, features = x.size()
                self.gmm = GaussianMixture(self.num_clusters, covariance_type='full')
                self._initialize_gmm(x)
            else:
                self.init_check = self.kmeans.fit_predict(x)
            if x.size()[1] != ModelParams.num_hidden_features[0]:
                self.reset_params(x.size()[1])

        if self.recluster_option == ReclusterOption.EV:
            self._recluster(x)
        elif self.recluster_option == ReclusterOption.TH:
            if self._check_recluster_threshold(x):
                self._recluster(x)
        elif self.recluster_option == ReclusterOption.ITR:
            self.iteration_count += 1
            if self.iteration_count >= self.recluster_iterations:
                self.iteration_count = 0
                self._recluster(x)
        else:
            raise ValueError(f"Unsupported recluster type: {self.recluster_option}")

        cluster_indices = self.get_cluster_indices(x)
        """temp_m, temp_n = x.size()
        assert temp_n % self.num_clusters == 0
        x = x.view(temp_m, self.num_clusters, -1)
        x = (x - x.mean(-1, keepdim=True)) / (x.var(-1, keepdim=True) + self.eps).sqrt()
        x = x.view(temp_m, temp_n)
        x = x * self.weight + self.bias"""
        for c in range(self.num_clusters):
            if cluster_indices[c].numel() != 0:
                cluster = x[cluster_indices[c]].clone()
                if self.normalize:
                    cluster = (cluster - cluster.mean(-1, keepdim=True)) / (cluster.var(-1, keepdim=True) + self.eps).sqrt()
                # cluster = cluster * self.weight[c] + self.bias[c]
                x[cluster_indices[c]] = self.activation[c](cluster)

        return x

    def _initialize_gmm(self, x: torch.Tensor) -> None:
        """
        Initialize the Gaussian Mixture Model with the input data.

        Parameters:
            x : Input data for GMM initialization.
        """
        num_samples, num_features = x.size()
        self.gmm.fit(x.detach().cpu().numpy())

    def _recluster(self, x: torch.Tensor) -> None:
        if self.use_gmm:
            self.gmm.fit(x.detach().cpu().numpy())
            self.prev_centroids = torch.tensor(self.gmm.means_)
        else:
            self.kmeans.fit(x)
            self.prev_centroids = self.kmeans.centroids.clone()

    def _check_recluster_threshold(self, x: torch.Tensor) -> bool:
        if self.prev_centroids is None:
            return True

        if self.use_gmm:
            current_centroids = torch.tensor(self.gmm.means_)
        else:
            # Fit the KMeans model to get the current cluster centroids
            self.kmeans.fit(x)
            current_centroids = torch.tensor(self.kmeans.centroids)

        centroid_distances = torch.cdist(current_centroids, self.prev_centroids)
        mean_distances = centroid_distances.min(dim=1).values
        cluster_means = [
            x[torch.nonzero(self.kmeans.predict(x) == i)].mean(dim=0)
            for i in range(self.num_clusters)
        ]

        for i in range(self.num_clusters):
            distance_limit = mean_distances[i] * self.recluster_threshold
            if torch.norm(cluster_means[i] - current_centroids[i]) > distance_limit:
                return True

        return False

    def get_cluster_indices(self, x: torch.Tensor) -> torch.Tensor:
        if self.init_check is None:
            if self.use_gmm:
                self.init_check = True
                self._initialize_gmm(x)
            else:
                self.init_check = self.kmeans.fit_predict(x)
        if self.use_gmm:
            labels = torch.tensor(self.gmm.predict(x.detach().cpu().numpy()))
        else:
            labels = self.kmeans.predict(x)
        cluster_indices = [
            torch.nonzero(labels == index).squeeze().cpu()
            for index in range(self.num_clusters)
        ]
        return cluster_indices
