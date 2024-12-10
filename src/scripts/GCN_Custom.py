import torch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import LRGBDataset
from torch.nn import BCEWithLogitsLoss
from torch_geometric.nn import GCN, global_mean_pool

from torchmetrics.classification import MultilabelAveragePrecision

import numpy as np

# Load the Peptide-Func dataset
dataset = LRGBDataset(root="data/LRGBDataset", name="Peptides-func")
val_dataset = LRGBDataset(root="data/LRGBDataset", name="Peptides-func", split="val")
test_dataset = LRGBDataset(root="data/LRGBDataset", name="Peptides-func", split="test")

class GraphGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.gcn = GCN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=hidden_channels, num_layers=3)
        self.linear = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.gcn(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = global_mean_pool(x, batch)
        x = self.linear(x)
        return x

# Define the GCN model
model = GraphGCN(
    in_channels=dataset.num_node_features,
    hidden_channels=300,
    out_channels=dataset.num_classes,
    num_layers=5
)

print(model)

# Define optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = BCEWithLogitsLoss()

# Create a DataLoader for batching multiple graphs
train_loader = DataLoader(dataset, batch_size=200, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

def train_and_evaluate(model, train_loader, val_loader, test_loader, optimizer, criterion, num_epochs=20):
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
            optimizer.zero_grad()
            batch.x = batch.x.float()
            out = model(x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch)
            loss = criterion(out, batch.y.float())
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
                    batch.x = batch.x.float()
                    out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
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


        # Print epoch results
        print(f"Epoch {epoch + 1}:")
        print(f"  Loss: {total_loss:.4f}")
        print(f"  Train AP: {train_ap:.4f}")
        print(f"  Validation AP: {val_ap:.4f}")
        print(f"  Test AP: {test_ap:.4f}")
        print("-" * 40)


    return model, all_loss, all_train_ap, all_val_ap, all_test_ap

trained_model, all_loss, all_train_ap, all_val_ap, all_test_ap = train_and_evaluate(model, train_loader, val_loader, test_loader, optimizer, criterion, num_epochs=1000)

# Save arrays to disk
np.save("all_loss.npy", all_loss)
np.save("all_train_ap.npy", all_train_ap)
np.save("all_val_ap.npy", all_val_ap)
np.save("all_test_ap.npy", all_test_ap)

# Save the trained model
torch.save(trained_model.state_dict(), "trained_model.pth")
