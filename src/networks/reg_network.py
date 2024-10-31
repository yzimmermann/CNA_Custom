import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch
import torch.nn as nn
from torch_geometric.nn import (
    ARMAConv,
    ChebConv,
    CuGraphSAGEConv,
    GATConv,
    GCNConv,
    GraphConv,
    LEConv,
    MFConv,
    SAGEConv,
    Sequential,
    SGConv,
    SSGConv,
    TransformerConv,
)

from utils.model_params import LayerType
from utils.model_params import ModelParams as mp


class RegNet(torch.nn.Module):
    """
    A neural network model for graph data with customizable layers and activations.

    Parameters:
        activation (callable): The activation function to use in intermediate layers.
        input_features (int): Number of input features (node features) in the graph.
        output_features (int): Number of output features (node classes) in the graph.
        hidden_features (int): Number of hidden features in the intermediate layers.
        num_layer (int): Number of layers in the model.
        layer_type (str, optional): The type of graph convolutional layer to use.
            Defaults to "GCNConv".

    Attributes:
        layer_type (str): The type of graph convolutional layer used in the model.
        activation (callable): The activation function used in intermediate layers.
        num_layer (int): Number of layers in the model.
        model (torch_geometric.nn.Sequential): The PyTorch Geometric Sequential model
            containing the layers of the network.
    """

    def __init__(
        self,
        activation,
        input_features,
        output_features,
        hidden_features,
        num_layer,
        layer_type=mp.model_type,
    ):
        super(RegNet, self).__init__()
        self.layer_type = layer_type
        self.activation = activation
        self.num_layer = num_layer
        self.encoder = nn.Linear(input_features, hidden_features)
        self.model_ = self._build_sequential_container(hidden_features, hidden_features)
        self.last_layer = self._get_conv_layer(hidden_features, hidden_features)
        self.decoder = nn.Linear(hidden_features, 1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(self, x, edge_index=None):
        """
        Perform a forward pass through the model.

        Parameters:
            x (torch.Tensor): Input node features in the graph.
            edge_index (torch.Tensor): Graph edge indices (COO format).

        Returns:
            torch.Tensor: Log softmax output of the model.
        """
        x = self.encoder(x)
        h = self.model_(x, edge_index)
        out = self.activation(h)
        out = self.last_layer(out, edge_index)
        out = self.activation(out)

        return self.decoder(out)

    def _build_sequential_container(self, input_features, hidden_features):
        """
        Build the PyTorch Geometric Sequential container with the specified layers.

        Parameters:
            input_features (int): Number of input features (node features) in the graph.
            hidden_features (int): Number of hidden features in the intermediate layers.

        Returns:
            torch_geometric.nn.Sequential: The PyTorch Geometric Sequential container
            containing the specified layers.
        """
        if self.num_layer == 2:
            conv_list = [
                (
                    self._get_conv_layer(input_features, hidden_features),
                    "x, edge_index -> x0",
                )
            ]
            return Sequential("x, edge_index", conv_list)

        conv_list = [
            (
                self._get_conv_layer(input_features, hidden_features),
                "x, edge_index -> x0",
            ),
        ]
        [
            conv_list.extend(
                [
                    (self.activation, f"x{i-1} -> x{i-1}a"),
                    (
                        self._get_conv_layer(hidden_features, hidden_features),
                        f"x{i - 1}a, edge_index -> x{i}",
                    ),
                ]
            )
            for i in range(1, self.num_layer - 1)
        ]

        return Sequential("x, edge_index", conv_list)

    def _get_conv_layer(self, input_features, output_features):
        """
        Get the specified graph convolutional layer based on the layer_type.

        Parameters:
            input_features (int): Number of input features (node features) in the graph.
            output_features (int): Number of output features (node classes) in the graph.

        Returns:
            torch_geometric.nn.MessagePassing: The graph convolutional layer based on
            the layer_type.

        Raises:
            ValueError: If the specified layer_type is not supported.
        """
        if self.layer_type == LayerType.GCNCONV:
            return GCNConv(input_features, output_features)
        elif self.layer_type == LayerType.SAGECONV:
            return SAGEConv(input_features, output_features)
        elif self.layer_type == LayerType.GATCONV:
            return GATConv(input_features, output_features)
        elif self.layer_type == LayerType.TRANSFORMERCONV:
            return TransformerConv(input_features, output_features)
        elif self.layer_type == LayerType.CHEBCONV:
            return ChebConv(input_features, output_features, K=3)
        elif self.layer_type == LayerType.CUGRAPHSAGECONV:
            return CuGraphSAGEConv(input_features, output_features)
        elif self.layer_type == LayerType.GRAPHCONV:
            return GraphConv(input_features, output_features)
        elif self.layer_type == LayerType.ARMACONV:
            return ARMAConv(input_features, output_features)
        elif self.layer_type == LayerType.SGCONV:
            return SGConv(input_features, output_features)
        elif self.layer_type == LayerType.MFCONV:
            return MFConv(input_features, output_features)
        elif self.layer_type == LayerType.SSGCONV:
            return SSGConv(input_features, output_features, alpha=0.5)
        elif self.layer_type == LayerType.LECONV:
            return LEConv(input_features, output_features)
        else:
            raise ValueError(f"Unsupported layer_type: {self.layer_type}")
