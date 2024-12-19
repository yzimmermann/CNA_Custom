import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


import torch
from torch_geometric import data
from torch_geometric.datasets import LRGBDataset
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from torch.nn import Linear, Dropout
import torch.nn.functional as F
from torch_geometric.nn import (
    ARMAConv, ChebConv, CuGraphSAGEConv, GATConv, GCNConv, GraphConv, LEConv, Linear, MFConv, SAGEConv, Sequential,
    SGConv, SSGConv, TransformerConv, BatchNorm
)

from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.model_params import LayerType, ModelParams as mp, ActivationType, ReclusterOption
from clustering.rationals_on_clusters import RationalOnCluster

import numpy as np
from torchmetrics.classification import MultilabelAveragePrecision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset laden
train_dataset = LRGBDataset(root='data/LRGBDataset', name='Peptides-func', split='train')
val_dataset = LRGBDataset(root='data/LRGBDataset', name='Peptides-func', split='val')
test_dataset = LRGBDataset(root='data/LRGBDataset', name='Peptides-func', split='test')

# Print dataset sizes for verification
print(f'Dataset: {train_dataset}:')
print(f"Number of training graphs: {len(train_dataset)}")
print(f"Number of test graphs: {len(test_dataset)}")
print('====================')


class BatchData:
    def __init__(self, x, batch):
        self.x = x
        self.batch = batch


class MLPGraphHead(torch.nn.Module):
    """
    MLP prediction head for graph prediction tasks.

    Args:
        hidden_channels (int): Input dimension.
        out_channels (int): Output dimension. For binary prediction, dim_out=1.
        L (int): Number of hidden layers.
    """

    def __init__(self, hidden_channels, out_channels):
        super().__init__()

        self.pooling_fun = global_mean_pool
        dropout = 0.1
        L = 3

        layers = []
        for _ in range(L - 1):
            layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.Linear(hidden_channels, hidden_channels, bias=True))
            layers.append(torch.nn.GELU())

        layers.append(torch.nn.Dropout(dropout))
        layers.append(torch.nn.Linear(hidden_channels, out_channels, bias=True))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, batch):
        x = self.pooling_fun(batch.x, batch.batch)
        return self.mlp(x)

def multilabel_weighted_bce_loss(output, target, weights=None):
    """
    Weighted Binary Cross Entropy for Multilabel Classification

    Args:
    - output: Model predictions (logits)
    - target: Ground truth labels (multilabel)
    - weights: Optional per-class weight tensor
    """
    if weights is None:
        # Default: compute class frequencies and invert
        pos_freq = target.float().mean(dim=0)
        weights = 1.0 / (pos_freq + 1e-8)

    # Compute BCE loss with per-class weights
    bce_loss = F.binary_cross_entropy_with_logits(
        output,
        target,
        pos_weight=weights.to(output.device),
        reduction='none'
    )

    return bce_loss.mean()

# Model Definition
class Net(torch.nn.Module):
    def __init__(self, activation, hidden_features, num_layer, layer_type, out_channels):
        super(Net, self).__init__()
        self.layer_type = layer_type
        self.activation = activation
        self.num_layer = num_layer
        self.model_ = self._build_sequential_container(train_dataset.num_features, hidden_features)
        self.head = MLPGraphHead(hidden_channels=hidden_features, out_channels=out_channels)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """
    def reset_parameters(self):
        self.model_.reset_parameters()
        for layer in self.head.mlp:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    """

    def forward(self, x, edge_index, batch):
        # Compute node-level embeddings
        h = self.model_(x, edge_index)

        # Global pooling and MLP head processing
        batch_data = BatchData(x=h, batch=batch)
        return self.head(batch_data)

    def _build_sequential_container(self, input_features, hidden_features):
        dropout_rate = 0.1  # You can adjust this value as needed
        conv_list = [
            (self._get_conv_layer(input_features, hidden_features), "x, edge_index -> x0"),
            (self.activation, "x0 -> x0_act"),
            (BatchNorm(hidden_features), "x0_act -> x0_bn"),
            (Dropout(p=dropout_rate), "x0_bn -> x0_do"),
        ]
        for i in range(1, self.num_layer - 1):
            conv_list.extend([
                (self._get_conv_layer(hidden_features, hidden_features), f"x{i - 1}_do, edge_index -> x{i}"),
                (self.activation, f"x{i} -> x{i}_act"),
                (BatchNorm(hidden_features), f"x{i}_act -> x{i}_bn"),
                (Dropout(p=dropout_rate), f"x{i}_bn -> x{i}_do"),
            ])
        return Sequential("x, edge_index", conv_list)

    def _get_conv_layer(self, input_features, output_features):
        if self.layer_type == LayerType.LINEAR:
            return Linear(input_features, output_features)
        elif self.layer_type == LayerType.GCNCONV:
            return GCNConv(input_features, output_features)
        elif self.layer_type == LayerType.SAGECONV:
            return SAGEConv(input_features, output_features)
        elif self.layer_type == LayerType.GATCONV:
            return GATConv(input_features, output_features, heads=3, edge_dim=train_dataset.num_edge_features,
                           concat=False)
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

torch.manual_seed(3)  # 0, 1, 2, 3, 4
#dataset = dataset.shuffle()
# MUTAG
# train_dataset, test_dataset = dataset[:150], dataset[150:]
# Protein-func
#train_dataset, test_dataset = dataset[:12428], dataset[12428:]
print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

activation = RationalOnCluster(
    clusters=20,
    with_clusters=True,
    # num_activation=8,
    n=5,
    m=5,
    activation_type=ActivationType.RAT,
    mode=True,
    normalize=True,
    recluster_option=ReclusterOption.ITR,
)


#activation = torch.nn.GELU()

model = Net(activation, 235, 6, LayerType.GCNCONV, train_dataset.num_classes).to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# criterion = multilabel_weighted_bce_loss
criterion = torch.nn.BCEWithLogitsLoss()

scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=20,
    min_lr=1e-5,
    verbose=True
)

def train_and_evaluate(model, train_loader, val_loader, test_loader, optimizer, criterion, scheduler, num_epochs=20):
    # Initialize Average Precision metric
    AP = MultilabelAveragePrecision(num_labels=train_loader.dataset.num_classes, average='macro')
    all_loss = []
    all_train_ap = []
    all_val_ap = []
    all_test_ap = []

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch.to(device)
            optimizer.zero_grad()
            batch.x = batch.x.float()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluation phase
        model.eval()

        # Helper function to compute AP for a given loader
        def compute_ap(loader, set_name):
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for batch in loader:
                    batch.to(device)
                    batch.x = batch.x.float()
                    out = model(batch.x, batch.edge_index, batch.batch)
                    preds = torch.sigmoid(out).cpu()
                    targets = batch.y.cpu()

                    all_preds.append(preds)
                    all_targets.append(targets)

            all_preds = torch.cat(all_preds)
            all_targets = torch.cat(all_targets)

            # Reset and update the AP metric
            AP.reset()
            AP.update(all_preds, all_targets.long())
            ap_score = AP.compute().item()
            return ap_score

        # Compute AP for train, validation, and test sets
        train_ap = compute_ap(train_loader, "Train")
        val_ap = compute_ap(val_loader, "Validation")
        test_ap = compute_ap(test_loader, "Test")

        # Save results
        all_loss.append(total_loss / len(train_loader))
        all_train_ap.append(train_ap)
        all_val_ap.append(val_ap)
        all_test_ap.append(test_ap)

        scheduler.step(val_ap)

        # Print epoch results
        print(f"Epoch {epoch + 1}:")
        print(f"  Loss: {total_loss:.4f}")
        print(f"  Train AP: {train_ap:.4f}")
        print(f"  Validation AP: {val_ap:.4f}")
        print(f"  Test AP: {test_ap:.4f}")
        print(f"  Current Learning Rate: {optimizer.param_groups[0]['lr']}")
        print("-" * 40)

    return model, all_loss, all_train_ap, all_val_ap, all_test_ap

trained_model, all_loss, all_train_ap, all_val_ap, all_test_ap = train_and_evaluate(model, train_loader, val_loader, test_loader, optimizer, criterion, scheduler, num_epochs=250)


