import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch_geometric.data as geom_data
from pytorch_lightning.loggers import TensorBoardLogger
from activations.torch import Rational
from torch_geometric.nn import GCNConv
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(pl.LightningModule):
    def __init__(
        self,
        activation,
        input_features,
        output_features,
        hidden_features,
        num_layer,
        learning_rate,
    ):
        """
        The __init__ function is called when the class is instantiated.
        It defines all the layers of our network, self.conv_list and self.activation.

        :param self: Represent the instance of the class
        :param activation: Specify the activation function used in each layer
        :param input_features: Specify the number of input features
        :param output_features: Define the number of output features
        :param hidden_features: Set the number of hidden features in each layer
        :param num_layer: Determine the number of layers in the network
        :return: The instance of the class
        """
        super(Net, self).__init__()

        self.conv_list = torch.nn.ModuleList([])
        self.num_layer = num_layer
        self.conv_list.append(GCNConv(input_features, hidden_features))
        for i in range(self.num_layer - 2):
            self.conv_list.append(GCNConv(hidden_features, hidden_features))

        self.conv_list.append(GCNConv(hidden_features, output_features))
        self.activation = activation
        self.lr = learning_rate
        self.weight_decay = 5e-4

    def forward(self, data):
        """
        The forward function of the GCN class.

        :param self: Represent the instance of the class
        :param x: Store the features of each node
        :param edge_index: Pass the graph structure to the gcn layer
        :return: The log_softmax of the output
        """
        x, edge_index = data.x, data.edge_index

        for i in range(self.num_layer - 2):
            x = self.conv_list[i](x, edge_index)
            x = self.activation(x)
            x = F.dropout(x, training=self.training)
        x = self.conv_list[self.num_layer - 1](x, edge_index)

        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        pred = out.argmax(dim=1)
        correct = (pred[batch.train_mask] == batch.y[batch.train_mask]).sum()
        acc = correct.float() / batch.train_mask.sum()
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        loss = F.nll_loss(out[batch.val_mask], batch.y[batch.val_mask])
        self.log("val_loss", loss)
        pred = out.argmax(dim=1)
        correct = (pred[batch.val_mask] == batch.y[batch.val_mask]).sum()
        acc = correct.float() / batch.val_mask.sum()
        self.log("val_acc", acc)


    def test_step(self, batch, batch_idx):
        out = self(batch)
        loss = F.nll_loss(out[batch.test_mask], batch.y[batch.test_mask])
        self.log("test_loss", loss)
        pred = out.argmax(dim=1)
        correct = (pred[batch.test_mask] == batch.y[batch.test_mask]).sum()
        acc = correct.float() / batch.test_mask.sum()

        self.log('test_acc', acc)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer


def run_experiment(
    activation,
    dataset,
    hidden_features=50,
    num_layers=3,
    lr=5e-3,
    epochs=1000,
    name=None,
):
    """
    The run_experiment function trains a model with the given activation
    function, hidden features, number of layers, learning rate and weight
    decay. It returns a list of test accuracies for each epoch.

    :param activation: Determine the activation function used in each layer
    :param hidden_features: Specify the number of hidden features in each layer
    :param num_layers: Specify the number of hidden layers in the network
    :param lr: Set the learning rate of the optimizer
    :param weight_decay: Control the l2 regularization strength
    :param epochs: Determine how many times the model will be trained on the data
    :return: A list of accuracies
    """

    data_loader = geom_data.DataLoader(dataset, batch_size=1)

    logger = TensorBoardLogger("tb_logs_activations", name=name)

    # Create PyTorch Lightning trainer
    trainer = pl.Trainer(accelerator="gpu", max_epochs=epochs, logger=logger, min_epochs=epochs)

    model = Net(
        activation,
        dataset.num_features,
        dataset.num_classes,
        hidden_features,
        num_layers,
        lr,
    ).to(device)

    trainer.logger._default_hp_metric = None

    trainer.fit(model, data_loader, data_loader)
    trainer.test(model, data_loader)


if __name__ == "__main__":
    """
    The main function runs the experiment.
    """

    # fixed seed
    set_manual_seed()

    # Load Cora dataset
    dataset = load_dataset("Cora")

    hidden_features = 400
    num_layers = [3, 5, 10]
    lrs = [5e-4, 1e-4]
    epochs = 1000


    for lyrs in num_layers:
        for lr in lrs:
            run_experiment(
                torch.nn.ReLU(),
                dataset,
                hidden_features=hidden_features,
                lr=lr,
                num_layers=lyrs,
                name=f"ReLU_nlyer_{lyrs}_hf_{hidden_features}_epochs_{epochs}_lr_{lr}",
            )
            run_experiment(
                Rational(approx_func="relu"),
                dataset,
                hidden_features=hidden_features,
                lr=lr,
                num_layers=lyrs,
                name=f"Rational_nlyer_{lyrs}_hf_{hidden_features}_epochs_{epochs}_lr_{lr}",
            )
