import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch
from activations.torch import Rational
from utils.model_params import ActivationType, DistanceMetrics, ReclusterOption
from activation_functions.power_mean import RationalPowerMeanModel
from activation_functions.rational_power_mean import RationalPowerMean
from activation_functions.rationals import RationalsModel
from .tensor_cluster import ClusterActivation
from activation_functions.weighted_arithmetic_mean import RationalWeightedArithmeticMean
from utils.model_params import ModelParams as mp


class RationalOnCluster(torch.nn.Module):
    def __init__(
        self,
        clusters=None,
        with_clusters=False,
        normalize=True,
        num_activation=4,
        n=5,
        m=5,
        activation_type=ActivationType.RAT,
        mode=DistanceMetrics.EUCLIDEAN,
        recluster_option=ReclusterOption.TH
    ):
        """
        Initializes a RationalOnCluster activation module.

        Parameters:
            clusters (int): Number of clusters for tensor clustering. Default is None.
            with_clusters (bool): If True, the activation uses tensor clustering. Default is False.
            normalize (bool): If True, the hidden features will be normalized.
            num_activation (int): Number of individual activation_functions in the rational function. Default is 4.
            n (int): Numerator for the rational function. Default is 5.
            m (int): Denominator for the rational function. Default is 5.
            activation_type (ActivationType): Type of activation function to use. Default is "RAT".
            mode (DistanceMetrics): Mode for tensor clustering distance metric. Default is "Euclidean".
        """
        super(RationalOnCluster, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.clusters = clusters
        self.with_clusters = with_clusters
        self.normalize = normalize
        self.type = activation_type
        self.n = n
        self.m = m
        self.mode = mode
        self.num_activation = num_activation
        self.recluster_option = recluster_option
        self.parameters = []
        self.individual = []
        [self.individual.append("relu") for _ in range(self.num_activation)]
        self.activation = self._get_activation()
        if self.with_clusters:
            self.cluster_activation = ClusterActivation(
                num_clusters=self.clusters, activation=self.activation, mode=self.mode,
                recluster_option=self.recluster_option, normalize=self.normalize
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the activation.

        Parameters:
            x : Input tensor.

        Returns:
            Output tensor after applying the activation.
        """
        if self.with_clusters:
            return self.cluster_activation(x)
        else:
            return self.activation[0](x)

    def _get_activation(self) -> list:
        """
        Returns the activation functions based on the specified activation type.

        Returns:
            A list of activation functions.
        """
        activation = []
        if self.type == ActivationType.RATMODEL:
            if self.with_clusters:
                [
                    activation.append(
                        RationalsModel(
                            n=self.n, m=self.m, function="relu", use_coefficients=mp.use_coefficients
                        )
                    )
                    for _ in range(self.clusters)
                ]
            else:
                activation.append(
                    RationalsModel(
                        n=self.n, m=self.m, function="relu", use_coefficients=mp.use_coefficients
                    )
                )
            [self.parameters.extend(act.parameters()) for act in activation]
            return activation

        elif self.type == ActivationType.RAT:
            if self.with_clusters:
                [activation.append(Rational("relu")) for _ in range(self.clusters)]
            else:
                activation.append(Rational("relu"))
            [self.parameters.extend(act.parameters()) for act in activation]
            return activation

        elif self.type == ActivationType.RPM:
            if self.with_clusters:
                [
                    activation.append(RationalPowerMean(self.individual).to(self.device))
                    for _ in range(self.clusters)
                ]
            else:
                activation.append(RationalPowerMean(self.individual).to(self.device))
            for act_list in activation:
                [self.parameters.extend(act.parameters()) for act in act_list.rationals]
            return activation
        elif self.type == ActivationType.RPMMODEL:
            if self.with_clusters:
                [
                    activation.append(RationalPowerMeanModel(self.individual).to(self.device))
                    for _ in range(self.clusters)
                ]
            else:
                activation.append(RationalPowerMeanModel(self.individual).to(self.device))
            for act_list in activation:
                [self.parameters.extend(act.parameters()) for act in act_list.rationals]
            return activation
        elif self.type == ActivationType.RWAM:
            if self.with_clusters:
                [
                    activation.append(
                        RationalWeightedArithmeticMean(self.individual).to(self.device)
                    )
                    for _ in range(self.clusters)
                ]
            else:
                activation.append(
                    RationalWeightedArithmeticMean(self.individual).to(self.device)
                )
            for act_list in activation:
                [self.parameters.extend(act.parameters()) for act in act_list.rationals]
            return activation
        else:
            raise ValueError(f"Unsupported rationals: {self.type}")
