import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from tqdm import tqdm
import logging
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import add_self_loops, to_dense_adj, to_undirected

from clustering.rationals_on_clusters import RationalOnCluster
from networks.network import Net
from utils.metrics import (
    compute_dirichlet_energy,
    compute_GMAD,
    compute_mad,
    compute_mad_for_centers,
    compute_mad_gap,
)
from utils.model_params import ActivationType
from utils.model_params import ModelParams as mp
from utils.plot_data import tsne_plot_with_centers
from utils.utils import load_dataset


class ModelTrainer:
    """
    Initializes a ModelTrainer object with the given configuration.

    Parameters:
        config (dict): A dictionary containing configuration parameters for the
        ModelTrainer.

    Returns:
        ModelTrainer: The initialized ModelTrainer object.
    """

    def __init__(self, config):
        self.config = config
        self.experiment_number = config["experiment_number"]
        self.epochs = config["epochs"]
        self.num_hidden_features = config["num_hidden_features"]
        self.lr_model = config["lr_model"]
        self.lr_activation = config["lr_activation"]
        self.weight_decay = config["weight_decay"]
        self.clusters = config["clusters"]
        self.num_layers = config["num_layers"]
        self.n = config["n"]
        self.m = config["m"]
        self.activation_type = config["activation_type"]
        self.recluster_option = config["recluster_option"]
        self.mode = config["mode"]
        self.with_clusters = config["with_clusters"]
        self.normalize = config["normalize"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.experiment_name = f"experiment_{self.experiment_number}"
        self.dataset_name = config["dataset_name"]
        self.log_path = config["log_path"]
        self.dataset_type = config["dataset_type"]
        self.layer_type = config["model_type"]
        self.batch_train = False
        self.number_classes = None

        print(config)

        self.activation = None

        self.setup_logger()
        self.setup_model()

    def setup_logger(self) -> None:
        """
        Sets up the logging for the ModelTrainer and creates a log file based on the experiment parameters.
        """
        date = datetime.now().strftime("%Y-%m-%d")
        filename = (
            f"{self.log_path}/{self.experiment_name}_epc_{self.epochs}_mlr_{self.lr_model}_alr_"
            f"{self.lr_activation}_wd_{self.weight_decay}_hf_{self.num_hidden_features}_layers_"
            f"{self.num_layers}_cl_{self.with_clusters}_ncl_{self.clusters}_"
            f"{self.activation_type}_rco_{self.recluster_option}_"
            f"dst_{self.dataset_type}_dsn_{self.dataset_name}.txt"
        )

        if os.path.exists(filename):
            print("File already exists!")
            # return
            os.remove(filename)

        # Remove all handlers associated with the root logger object.
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            filename=filename,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

        logging.info(f"Experiment: {self.experiment_name}")
        logging.info(f"Date: {date}")
        logging.info(f"Number of Epochs: {self.epochs}")
        logging.info(f"Model Learning Rate: {self.lr_model}")
        logging.info(f"Activation Learning Rate: {self.lr_activation}")
        logging.info(f"Weight Decay: {self.weight_decay}")
        logging.info(f"Number of Hidden Features: {self.num_hidden_features}")
        logging.info(f"Number of Layers: {self.num_layers}")
        logging.info(f"With Clusters: {self.with_clusters}")
        logging.info(f"Number of Clusters: {self.clusters}")
        logging.info(f"Recluster Option: {self.recluster_option}")

    def setup_model(self) -> None:
        """
        Sets up the model and optimizer based on the provided activation type and other configuration parameters.
        """
        if self.activation_type == ActivationType.RELU:
            self.activation = torch.relu
        else:
            self.activation = RationalOnCluster(
                clusters=self.clusters,
                with_clusters=self.with_clusters,
                normalize=self.normalize,
                n=self.n,
                m=self.m,
                activation_type=self.activation_type,
                mode=self.mode,
                recluster_option=self.recluster_option,
            )

        if self.dataset_type == "Planetoid":
            self.dataset = load_dataset(
                self.dataset_name,
                dataset_type=self.dataset_type,
            )
            self.number_classes = self.dataset.num_classes
            self.model = Net(
                activation=self.activation,
                input_features=self.dataset.num_features,
                output_features=self.dataset.num_classes,
                hidden_features=self.num_hidden_features,
                num_layer=self.num_layers,
                layer_type=self.layer_type,
            ).to(self.device)
        elif self.dataset_type in [
            "CitationFull",
            "Amazon",
            "WikipediaNetwork",
            "WebKB",
            "PygNodePropPredDataset",
        ]:  
            loaded_dataset = load_dataset(
                self.dataset_name,
                dataset_type=self.dataset_type,
            )
            self.dataset, self.number_classes = loaded_dataset[0], loaded_dataset[1]
            self.model = Net(
                activation=self.activation,
                input_features=torch.Tensor(self.dataset.x).shape[1],
                output_features=self.number_classes,
                hidden_features=self.num_hidden_features,
                num_layer=self.num_layers,
                layer_type=self.layer_type,
            ).to(self.device)

        else:
            raise ValueError("Please check the dataset type!")

        self.model.reset_parameters()
        if self.activation_type == ActivationType.RELU:
            self.optimizer = torch.optim.Adam(
                [{"params": self.model.parameters()}],
                lr=self.lr_model,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimizer = torch.optim.Adam(
                [
                    {"params": self.model.parameters()},
                    {"params": self.activation.parameters, "lr": self.lr_activation},
                ],
                lr=self.lr_model,
                weight_decay=self.weight_decay,
            )

    def train(self) -> None:
        """
        Trains the model using the provided dataset for the specified number of epochs.
        """
        print(self.model)
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"The number of required params : {num_params}")
        stats = {'val_acc': list(), 'test_acc': list()}
        if self.dataset_type in [
            "CitationFull",
            "Amazon",
            "WikipediaNetwork",
            "WebKB",
            "PygNodePropPredDataset",
        ]:
            loaded_dataset = load_dataset(
                self.dataset_name,
                dataset_type=self.dataset_type,
            )
            data = loaded_dataset[0].to(self.device)
            self.num_classes = loaded_dataset[1]
            print(self.dataset_name, data)
        else:
            data = self.dataset[0].to(self.device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if self.dataset_type == "PygNodePropPredDataset":
            evaluator = Evaluator(self.dataset_name)
            x = data.x.to(self.device)
            y_true = data.y.to(self.device)
            edge_index = data.edge_index.to(self.device)
            edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)
        else:
            evaluator = None
            x = None
            y_true = None
            edge_index = None

        for epoch in range(self.epochs):
            if self.batch_train:
                train_loader = NeighborLoader(
                    data,
                    input_nodes=data.train_mask,
                    num_neighbors=[25, 10],
                    shuffle=True,
                )
                train_acc, train_loss = self._train_batches(train_loader, epoch)
                val_acc, val_loss = self._valid(data)
                test_acc, test_loss = self._test(data)
            else:
                if self.dataset_type == "PygNodePropPredDataset":
                    train_loss = self.train_ogb(edge_index, data)
                    result = self.test_ogb(edge_index, data, evaluator)
                    train_acc, val_acc, test_acc = result
                else:
                    train_acc, train_loss = self._train(data)
                    val_acc, val_loss = self._valid(data)
                    test_acc, test_loss = self._test(data)

            stats['val_acc'].append(val_acc)
            stats['test_acc'].append(test_acc)
            print(
                f"Epoch: {epoch:03d}, Loss: {train_loss:.4f}, "
                f"Test Accuracy: {test_acc:.4f}, "
                f"Validation Accuracy: {val_acc:.4f}"
            )
            self.log_metrics(epoch, train_loss, test_acc, val_acc)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self.model.eval()
        with torch.no_grad():
            h, _ = self.model(data.x, data.edge_index)
        data = data.to(torch.device("cpu"))
        if mp.compute_other_metrics:
            h = h.to(torch.device("cpu"))
            A = to_dense_adj(data.edge_index).squeeze(0)
            D = torch.diag(A.sum(dim=1).view(-1))
            L = D - A
            d = torch.diag(A.sum(dim=1).view(-1) ** (-1 / 2))
            normalized_laplacian = torch.matmul(d, torch.matmul(L, d))
            mad = compute_mad(h, np.ones((data.num_nodes, data.num_nodes)))
            mad_gap = compute_mad_gap(h, A.numpy())
            if self.dataset_type in [
                "CitationFull",
                "Amazon",
                "WikipediaNetwork",
                "WebKB",
                "PygNodePropPredDataset",
            ]:
                mad_centers = compute_mad_for_centers(h, data, self.number_classes)
            else:
                mad_centers = compute_mad_for_centers(h, data, self.dataset.num_classes)
            mad_gen = compute_GMAD(h)
            dirichlet_energy = compute_dirichlet_energy(normalized_laplacian, h)
            result = max(zip(stats['val_acc'], stats['test_acc']), key=lambda x: x[0])
            info = (
                f"Epochs {self.epochs} - "
                f"Max Validation Accuracy: {result[0]:.4f} - "
                f"Test Accuracy: {result[1]:.4f} - "
                f"MAD: {mad:.4f} - "
                f"MADGap: {mad_gap:.4f} - "
                f"DE: {dirichlet_energy:.4f} - "
                f"MADC: {mad_centers:.4f} - "
                f"GMAD: {mad_gen:.4f} - "
                f"layers:{self.num_layers}"
            )
            logging.info(info)
            print(info)
        else:
            result = max(zip(stats['val_acc'], stats['test_acc']), key=lambda x: x[0])
            info = (
                f"Epochs {self.epochs} - "
                f"Max Validation Accuracy: {result[0]:.4f} - "
                f"Test Accuracy: {result[1]:.4f} - "
                f"layers:{self.num_layers}"
            )
            logging.info(info)
            print(info)

        # store the trained model
        if mp.save_model:
            modelname = (
                f"{self.log_path}/{self.experiment_name}_epc_{self.epochs}_mlr_{self.lr_model}_alr_"
                f"{self.lr_activation}_wd_{self.weight_decay}_hf_{self.num_hidden_features}_layers_"
                f"{self.num_layers}_cl_{self.with_clusters}_ncl_{self.clusters}_"
                f"{self.activation_type}_rco_{self.recluster_option}_"
                f"dst_{self.dataset_type}_dsn_{self.dataset_name}.pth"
            )
            torch.save(self.model.state_dict(), modelname)

        if mp.plot_centers:
            _, _ = tsne_plot_with_centers(
                self.model,
                self.num_layers,
                self.dataset_name,
                data.to(self.device),
                self.model.layer_type,
                self.number_classes,
            )

    def _train(self, data) -> (float, float):
        """
        Private method to perform the training for a single epoch.

        Parameters:
            data: The dataset (torch_geometric.data.Data) containing the input features, edge_index, and masks.

        Returns:
            The accuracy and loss after the training for a single epoch.
        """

        self.model.train()
        self.optimizer.zero_grad()
        _, out = self.model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        # gradient clipping to avoid exploding gradients
        if isinstance(self.activation, RationalOnCluster):
            torch.nn.utils.clip_grad_norm_(self.activation.parameters, 1.0)
        self.optimizer.step()
        pred = out.argmax(dim=1)
        correct = (pred[data.train_mask] == data.y[data.train_mask]).sum()
        acc = float(correct) / data.train_mask.sum()
        return acc, loss.item()

    def _train_batches(self, train_loader, current_epoch):
        """
        Perform batch training for one epoch.

        Parameters:
            train_loader (torch_geometric.data.NeighborLoader): DataLoader for training.

        Returns:
            tuple: Tuple containing training accuracy and loss.
        """
        self.model.train()
        pbar = tqdm(total=int(len(train_loader.dataset)))
        pbar.set_description(f"Epoch {current_epoch:02d}")
        total_loss = total_correct = total_examples = 0

        for batch in train_loader:
            if batch.num_nodes >= self.clusters:
                self.optimizer.zero_grad()
                y = batch.y[: batch.batch_size]
                _, y_hat = self.model(batch.x, batch.edge_index.to(self.device))[
                    : batch.batch_size
                ]
                loss = F.cross_entropy(y_hat, y)
                loss.backward()
                self.optimizer.step()
                total_loss += float(loss) * batch.batch_size
                total_correct += int((y_hat.argmax(dim=-1) == y).sum())
                total_examples += batch.batch_size
                pbar.update(batch.batch_size)

        pbar.close()
        train_acc, train_loss = (
            total_loss / total_examples,
            total_correct / total_examples,
        )

        return train_acc, train_loss

    def _valid(self, data) -> (float, float):
        """
        Private method to perform the validation for a single epoch.

        Parameters:
            data: The dataset (torch_geometric.data.Data) containing the input features,
             edge_index, and masks.

        Returns:
            The accuracy and loss after the validation for a single epoch.
        """
        self.model.eval()
        _, out = self.model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
        pred = out.argmax(dim=1)
        correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
        acc = float(correct) / data.val_mask.sum()
        return acc, loss.item()

    def _test(self, data) -> (float, float):
        """
        Private method to perform the testing for a single epoch.

        Parameters:
            data: The dataset (torch_geometric.data.Data) containing the input features,
             edge_index, and masks.

        Returns:
            The accuracy and loss after the testing for a single epoch.
        """
        self.model.eval()
        _, out = self.model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.test_mask], data.y[data.test_mask])
        pred = out.argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = float(correct) / data.test_mask.sum()
        return acc, loss.item()

    def log_metrics(
        self,
        epoch: int,
        training_loss: float,
        test_accuracy: float,
        validation_accuracy: float,
    ) -> None:
        """
        Logs the training metrics (loss, test accuracy, and validation accuracy) for
        each epoch.

        Parameters:
            epoch : The current epoch number.
            training_loss : The training loss for the current epoch.
            test_accuracy : The test accuracy for the current epoch.
            validation_accuracy : The validation accuracy for the current epoch.
        """
        logging.info(
            f"Epoch {epoch} - Loss: {training_loss:.4f} - Test Accuracy: {test_accuracy:.4f} - Validation Accuracy: {validation_accuracy:.4f}"
        )

    @torch.no_grad()
    def test_ogb(self, edge_index, data, evaluator):
        self.model.eval()
        _, out = self.model(data.x, edge_index)
        y_pred = out.argmax(dim=-1, keepdim=True)

        y_true = data.y.unsqueeze(1)
        train_acc = evaluator.eval(
            {
                "y_true": y_true[data.train_mask],
                "y_pred": y_pred[data.train_mask],
            }
        )["acc"]
        valid_acc = evaluator.eval(
            {
                "y_true": y_true[data.val_mask],
                "y_pred": y_pred[data.val_mask],
            }
        )["acc"]
        test_acc = evaluator.eval(
            {
                "y_true": y_true[data.test_mask],
                "y_pred": y_pred[data.test_mask],
            }
        )["acc"]

        return train_acc, valid_acc, test_acc

    def train_ogb(self, edge_index, data):
        self.model.train()
        self.optimizer.zero_grad()
        _, out = self.model(data.x, edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        self.optimizer.step()

        return loss.item()
