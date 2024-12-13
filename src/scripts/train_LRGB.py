import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
from torch_geometric.datasets import LRGBDataset
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import (
    ARMAConv, ChebConv, CuGraphSAGEConv, GATConv, GCNConv, GraphConv, LEConv, Linear, MFConv, SAGEConv, Sequential,
    SGConv, SSGConv, TransformerConv
)
from utils.model_params import LayerType, ModelParams as mp, ActivationType, ReclusterOption
from clustering.rationals_on_clusters import RationalOnCluster

import numpy as np
from torchmetrics.classification import MultilabelAveragePrecision

# Dataset laden
train_dataset = LRGBDataset(root='data/LRGBDataset', name='Peptides-func', split='train')
val_dataset = LRGBDataset(root='data/LRGBDataset', name='Peptides-func', split='val')
test_dataset = LRGBDataset(root='data/LRGBDataset', name='Peptides-func', split='test')

# Print dataset sizes for verification
print(f'Dataset: {train_dataset}:')
print(f"Number of training graphs: {len(train_dataset)}")
print(f"Number of test graphs: {len(test_dataset)}")
print('====================')


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
    def __init__(self, activation, hidden_features, num_layer, layer_type):
        super(Net, self).__init__()
        self.layer_type = layer_type
        self.activation = activation
        self.num_layer = num_layer
        self.model_ = self._build_sequential_container(train_dataset.num_features, hidden_features)
        self.out_layer = Linear(hidden_features, train_dataset.num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def reset_parameters(self):
        self.model_.reset_parameters()

    def forward(self, x, edge_index, batch):
        h = self.model_(x, edge_index)
        out = h.relu()  # self.activation(h)
        out = global_mean_pool(out, batch)
        out = F.dropout(out, p=0.1, training=self.training)
        out = self.out_layer(out)
        return out

    def _build_sequential_container(self, input_features, hidden_features):
        conv_list = [
            (self._get_conv_layer(input_features, hidden_features), "x, edge_index -> x0"),
        ]
        for i in range(1, self.num_layer - 1):
            conv_list.extend([
                (self.activation, f"x{i - 1} -> x{i - 1}a"),
                (self._get_conv_layer(hidden_features, hidden_features), f"x{i - 1}a, edge_index -> x{i}"),
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
            return GATConv(input_features, output_features, heads=3, edge_dim=train_dataset.num_edge_features, concat=False)
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

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()

activation = RationalOnCluster(
    clusters=1,
    with_clusters=False,
    # num_activation=8,
    n=5,
    m=5,
    activation_type=ActivationType.RAT,
    mode=True,
    normalize=False,
    recluster_option=ReclusterOption.ITR,
)


activation = torch.nn.ReLU

model = Net(activation, 300, 5, LayerType.GATCONV).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
# criterion = multilabel_weighted_bce_loss
criterion = torch.nn.BCEWithLogitsLoss()

def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data.x = data.x.float()  # Convert to float for integer features
        data = data.to(model.device)
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        return total_loss / len(train_loader)

AP = MultilabelAveragePrecision(num_labels=train_dataset.num_classes, average='macro')
def test(loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            data.x = data.x.float()
            data = data.to(model.device)
            out = model(data.x, data.edge_index, data.batch)
            pred = torch.sigmoid(out).cpu().numpy()  # Convert to probabilities
            labels = data.y.cpu().numpy()
            all_preds.append(pred)
            all_labels.append(labels)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    print(all_preds.shape, all_labels.shape)
    uap = AP(torch.tensor(all_preds), torch.tensor(all_labels, dtype=torch.long))
    return uap


# In the main training loop
for epoch in range(1, 1001):
    train_loss = train()
    train_uap = test(train_loader)
    test_uap = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.10f}, Train UAP: {train_uap:.4f}, Test UAP: {test_uap:.4f}')