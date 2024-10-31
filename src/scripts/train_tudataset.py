import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import (
    ARMAConv, ChebConv, CuGraphSAGEConv, GATConv, GCNConv, GraphConv, LEConv, Linear, MFConv, SAGEConv, Sequential, SGConv, SSGConv, TransformerConv
)
from utils.model_params import LayerType, ModelParams as mp, ActivationType, ReclusterOption
from clustering.rationals_on_clusters import RationalOnCluster

# Dataset laden
# dataset = TUDataset(root='data/TUDataset', name='MUTAG')
dataset = TUDataset(root='data/TUDataset', name='PROTEINS')
print(f'Dataset: {dataset}:')
print('====================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
print(data)
print('=============================================================')
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

# Modelldefinition
class Net(torch.nn.Module):
    def __init__(self, activation, hidden_features, num_layer, layer_type):
        super(Net, self).__init__()
        self.layer_type = layer_type
        self.activation = activation
        self.num_layer = num_layer
        self.model_ = self._build_sequential_container(dataset.num_features, hidden_features)
        self.out_layer = Linear(hidden_features, dataset.num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def reset_parameters(self):
        self.model_.reset_parameters()

    def forward(self, x, edge_index, batch):
        h = self.model_(x, edge_index)
        out = h.relu()  # self.activation(h)
        out = global_mean_pool(out, batch)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.out_layer(out)
        return out

    def _build_sequential_container(self, input_features, hidden_features):
        conv_list = [
            (self._get_conv_layer(input_features, hidden_features), "x, edge_index -> x0"),
        ]
        for i in range(1, self.num_layer - 1):
            conv_list.extend([
                (self.activation, f"x{i-1} -> x{i-1}a"),
                (self._get_conv_layer(hidden_features, hidden_features), f"x{i-1}a, edge_index -> x{i}"),
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

torch.manual_seed(3) # 0, 1, 2, 3, 4
dataset = dataset.shuffle()
# MUTAG 
# train_dataset, test_dataset = dataset[:150], dataset[150:]
# ENZYMES
train_dataset, test_dataset = dataset[:951], dataset[951:]
print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()

activation = RationalOnCluster(
    clusters=8,
    with_clusters=True,
    num_activation=8,
    n=5,
    m=5,
    activation_type=ActivationType.RAT,
    mode=True,
    recluster_option=ReclusterOption.ITR,
)

# activation = torch.nn.ReLU

model = Net(activation, 128, 4, LayerType.GCNCONV).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
print(model)

# class GCN(torch.nn.Module):
#     def __init__(self, hidden_channels):
#         super(GCN, self).__init__()
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, hidden_channels)
#         self.conv3 = GCNConv(hidden_channels, hidden_channels)
#         self.lin = Linear(hidden_channels, dataset.num_classes)

#     def forward(self, x, edge_index, batch):
#         # 1. Obtain node embeddings 
#         x = self.conv1(x, edge_index)
#         x = x.relu()
#         x = self.conv2(x, edge_index)
#         x = x.relu()
#         x = self.conv3(x, edge_index)

#         # 2. Readout layer
#         x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

#         # 3. Apply a final classifier
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.lin(x)
        
#         return x

# model = GCN(hidden_channels=64).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
# print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    for data in train_loader:
        data = data.to(model.device)
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(model.device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)

max_text_acc = 0.0
for epoch in range(1, 1001):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    if max_text_acc < test_acc:
        max_text_acc = test_acc
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
print(f'Maximal Test Acc: {max_text_acc}')
