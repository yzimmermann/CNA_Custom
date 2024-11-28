import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import logging
from datetime import datetime
import numpy as np
import torch

from clustering.rationals_on_clusters import RationalOnCluster
from networks.reg_network import RegNet
from utils.model_params import ActivationType
from utils.model_params import ModelParams as mp
from utils.plot_data import tsne_plot_with_centers
from utils.utils import get_data, load_dataset


class RegressionTrainer:
    """
    Initializes a RegressionTrainer object with the given configuration.

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.experiment_name = f"experiment_{self.experiment_number}"
        self.dataset_name = config["dataset_name"]
        self.log_path = config["log_path"]
        self.dataset_type = config["dataset_type"]
        self.layer_type = config["model_type"]
        self.batch_train = False
        self.number_classes = None
        self.lf = torch.nn.MSELoss()

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
                n=self.n,
                m=self.m,
                activation_type=self.activation_type,
                mode=self.mode,
                recluster_option=self.recluster_option,
            )

        if self.dataset_name == "Chameleon":
            node_features = 3132
        else:
            node_features = 3148

        self.model = RegNet(
            activation=self.activation,
            input_features=node_features,
            output_features=1,
            hidden_features=self.num_hidden_features,
            num_layer=self.num_layers,
            layer_type=self.layer_type,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            [{"params": self.model.parameters()}],
            lr=self.lr_model,
            weight_decay=self.weight_decay,
        )

    def train(self, split) -> None:
        """
        Trains the model using the provided dataset for the specified number of epochs.
        """
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"The number of required params : {num_params}")
        losses = {'train': list(), 'test': list(), 'val':list()}
        if self.dataset_name == "Chameleon":
            data = get_data("chameleon", split)
        else:
            data = get_data("squirrel", split)
        print("Here to do some regression estimation!")
        data = data.to(self.device)

        print(self.dataset_name, data)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        print(self.model)
        for epoch in range(self.epochs):
            train_loss, val_loss, test_loss = self._train(data)
            losses["train"].append(train_loss)
            losses["test"].append(test_loss)
            losses["val"].append(val_loss)
            print(
                f"Split {split} - "
                f"Epoch: {epoch:03d}, "
                f"Loss: {train_loss:.4f}, "
                f"Test Loss: {test_loss:.4f}, "
                f"Validation Loss: {val_loss:.4f}"
            )
            self.log_metrics(split, epoch, train_loss, test_loss, val_loss)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        results = min(zip(losses["train"], losses["test"], losses["val"]), key=lambda x: x[2])
        info = (
            f"Split {split} - "
            f"Epochs {self.epochs} - "
            f"Min Validation Loss: {results[2]:.4f} - "
            f"Train Loss: {results[0]:.4f} - "
            f"Test Loss: {results[1]:.4f} - "
            f"layers:{self.num_layers}"
        )
        logging.info(info)
        print(info)

        if mp.plot_centers:
            _, _ = tsne_plot_with_centers(
                self.model,
                self.num_layers,
                self.dataset_name,
                data.to(self.device),
                self.model.layer_type,
                self.number_classes,
            )

        return results[1]

    @torch.no_grad()
    def test(self, data):
        self.model.eval()
        out, losses = self.model(data.x, data.edge_index).squeeze(-1), []
        for _, mask in data("train_mask", "val_mask", "test_mask"):
            loss = self.lf(
                out[mask].float(), data.y.squeeze()[mask].float()
            ) / torch.mean(data.y.float())
            losses.append(loss.item())
        return losses

    def _train(self, data) -> (float, float):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(data.x, data.edge_index).squeeze(-1)
        loss = self.lf(
            out[data.train_mask].float(), data.y.squeeze()[data.train_mask].float()
        )
        loss.backward()
        self.optimizer.step()

        [train_loss, val_loss, test_loss] = self.test(data)

        return (train_loss, val_loss, test_loss)

    def log_metrics(
        self,
        split: int,
        epoch: int,
        training_loss: float,
        test_loss: float,
        validation_loss: float,
    ) -> None:
        """
        Logs the training metrics (loss, test accuracy, and validation accuracy) for
        each epoch.

        Parameters:
            epoch : The current epoch number.
            training_loss : The training loss for the current epoch.
            test_loss : The test loss for the current epoch.
            validation_loss : The validation loss for the current epoch.
        """
        logging.info(
            f"Split {split} - Epoch {epoch} - Train Loss: {training_loss:.4f} - Test Loss: {test_loss:.4f} "
            f"- Validation Loss: {validation_loss:.4f}"
        )
