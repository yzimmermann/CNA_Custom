import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from activations.torch import Rational
from torch_geometric.nn import GCNConv
from utils import *

# Load the Cora dataset
dataset = load_dataset("Cora")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(torch.nn.Module):
    def __init__(
        self, activation, input_features, output_features, hidden_features, num_layer
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

    def forward(self, x, edge_index):
        """
        The forward function of the GCN class.

        :param self: Represent the instance of the class
        :param x: Store the features of each node
        :param edge_index: Pass the graph structure to the gcn layer
        :return: The log_softmax of the output
        """

        for i in range(self.num_layer - 1):
            x = self.conv_list[i](x, edge_index)
            x = self.activation(x)
            x = F.dropout(x, training=self.training)
        x = self.conv_list[self.num_layer - 1](x, edge_index)

        return F.log_softmax(x, dim=1)


def train(model, optimizer, data):
    """
    The train function takes in a model, optimizer, and data.
    It sets the model to train mode and zeros out the gradients of all parameters.
    Then it runs a forward pass on the data to get predictions from our model.
    We then calculate loss using negative log likelihood loss (NLLLoss) which is
    commonly used for classification problems with multiple classes (like ours).
    We use NLLLoss because we are predicting discrete labels for each node in our
    graph rather than continuous values like regression models do.
    After calculating loss we call .backward() on it which calculates gradients
    of all parameters with

    :param model: Pass the model to be trained
    :param optimizer: Update the weights of the model
    :param data: Pass the data to the model
    :return: The accuracy and loss
    """
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    pred = out.argmax(dim=1)
    correct = (pred[data.train_mask] == data.y[data.train_mask]).sum()
    acc = float(correct) / data.train_mask.sum()
    return acc, loss.item()


def validation(model, data):
    """
    The validation function is used to evaluate the performance of a model on
     a validation dataset.
    It takes as input:
        - A PyTorch model, which has been trained on some training data.
        - A PyTorch Data object containing the validation data (features and labels).
        The features are stored in `data.x` and the labels are stored in `data.y`.
        There is also an index for each node's edges, which can be accessed with
        `data.edge_index`.

    The function returns two values: accuracy and loss.

    :param model: Pass the model to the validation function
    :param data: Pass the data to the model
    :return: The accuracy and the loss
    """
    model.eval()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
    pred = out.argmax(dim=1)
    correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
    acc = float(correct) / data.val_mask.sum()
    return acc, loss.item()


def test(model, data):
    """
    The test function is used to evaluate the performance of a model on a dataset.
    It takes as input:
        - model: The trained model that we want to test.
        - data: The dataset on which we want to test the performance of our trained
                model.

    :param model: Pass the model to be trained
    :param data: Pass the data to the model
    :return: The accuracy and the loss
    """
    model.eval()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.test_mask], data.y[data.test_mask])
    pred = out.argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = float(correct) / data.test_mask.sum()
    return acc, loss.item()


def run_experiment(
    activation, hidden_features=50, num_layers=3, lr=5e-3, weight_decay=5e-4, epochs=200
):
    """
    The run_experiment function trains a model with the given activation function,
    hidden features, number of layers, learning rate and weight decay.
    It returns a list of test accuracies for each epoch.

    :param activation: Determine the activation function used in each layer
    :param hidden_features: Specify the number of hidden features in each layer
    :param num_layers: Specify the number of hidden layers in the network
    :param lr: Set the learning rate of the optimizer
    :param weight_decay: Control the l2 regularization strength
    :param epochs: Determine how many times the model will be trained on the data
    :return: A list of accuracies
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(
        activation,
        dataset.num_features,
        dataset.num_classes,
        hidden_features,
        num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    test_acc_list = []
    test_loss_list = []
    val_acc_list = []
    val_loss_list = []
    train_acc_list = []
    train_loss_list = []
    for epoch in range(1, epochs + 1):
        train_acc, train_loss = train(model, optimizer, dataset[0].to(device))
        val_acc, val_loss = validation(model, dataset[0].to(device))
        test_acc, test_loss = test(model, dataset[0].to(device))
        train_acc_list.append(train_acc.cpu())
        val_acc_list.append(val_acc.cpu())
        train_loss_list.append(train_loss)
        test_acc_list.append(test_acc.cpu())
        test_loss_list.append(test_loss)
        val_loss_list.append(val_loss)
        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Test Acc: {test_acc:.4f}"
            )
    return (
        test_acc_list,
        test_loss_list,
        train_acc_list,
        train_loss_list,
        val_acc_list,
        val_loss_list,
    )


def run_cases(params):
    # set parameters
    """
    The run_cases function takes a dictionary of parameters as input.
    The keys in the dictionary are:
        - rows: number of rows in the subplot grid (int)
        - cols: number of columns in the subplot grid (int)
        - n_layers: list containing lists with numbers representing how many
        layers to use for each experiment, e.g.: [[2, 3], [4]] means that there
        will be two experiments with 2 layers and one experiment with 4 layers;
        this is used to create a rectangular grid where each row represents an
        experiment using different numbers of hidden features and each column
        represents an experiment

    :param params: Pass the parameters to the function
    :return: stored figures
    """
    rows = params["rows"]
    cols = params["cols"]
    n_layers = params["n_layers"]
    relu = params["relu"]
    hidden_features = params["hidden_features"]
    lr = params["learning_rate"]
    epochs = params["epochs"]

    test_acc_fig, test_acc_axs = plt.subplots(nrows=rows, ncols=cols, figsize=(14, 10))
    test_loss_fig, test_loss_axs = plt.subplots(
        nrows=rows, ncols=cols, figsize=(14, 10)
    )
    train_acc_fig, train_acc_axs = plt.subplots(
        nrows=rows, ncols=cols, figsize=(14, 10)
    )
    train_loss_fig, train_loss_axs = plt.subplots(
        nrows=rows, ncols=cols, figsize=(14, 10)
    )
    val_acc_fig, val_acc_axs = plt.subplots(nrows=rows, ncols=cols, figsize=(14, 10))
    val_loss_fig, val_loss_axs = plt.subplots(nrows=rows, ncols=cols, figsize=(14, 10))

    for row in range(rows):
        for col in range(cols):
            n_layer = n_layers[row][col]
            (
                relu_test_acc_list,
                relu_test_loss_list,
                relu_train_acc_list,
                relu_train_loss_list,
                relu_val_acc_list,
                relu_val_loss_list,
            ) = run_experiment(
                relu,
                num_layers=n_layer,
                hidden_features=hidden_features,
                lr=lr,
                epochs=epochs,
            )
            rational = Rational(approx_func="relu")
            (
                rational_test_acc_list,
                rational_test_loss_list,
                rational_train_acc_list,
                rational_train_loss_list,
                rational_val_acc_list,
                rational_val_loss_list,
            ) = run_experiment(
                rational,
                num_layers=n_layer,
                hidden_features=hidden_features,
                lr=lr,
                epochs=epochs,
            )

            test_acc_axs[row, col].plot(
                range(1, len(relu_test_acc_list) + 1), relu_test_acc_list, label="ReLU"
            )
            test_acc_axs[row, col].plot(
                range(1, len(rational_test_acc_list) + 1),
                rational_test_acc_list,
                label="Rational",
            )
            test_acc_axs[row, col].set_title(f"n. layers = {n_layer}")
            test_acc_axs[row, col].set_xlabel("Epoch")
            test_acc_axs[row, col].set_ylabel("Test Accuracy")
            test_acc_axs[row, col].legend()
            test_acc_fig.suptitle(
                f"Test Accuracy: n. hidden feat. = {hidden_features}, lr = {lr}"
            )

            test_loss_axs[row, col].plot(
                range(1, len(relu_test_loss_list) + 1),
                relu_test_loss_list,
                label="ReLU",
            )
            test_loss_axs[row, col].plot(
                range(1, len(rational_test_loss_list) + 1),
                rational_test_loss_list,
                label="Rational",
            )
            test_loss_axs[row, col].set_title(f"n. layers = {n_layer}")
            test_loss_axs[row, col].set_xlabel("Epoch")
            test_loss_axs[row, col].set_ylabel("Test Loss")
            test_loss_axs[row, col].legend()
            test_loss_fig.suptitle(
                f"Test Loss: n. hidden feat. = {hidden_features}, lr = {lr}"
            )

            train_acc_axs[row, col].plot(
                range(1, len(relu_train_acc_list) + 1),
                relu_train_acc_list,
                label="ReLU",
            )
            train_acc_axs[row, col].plot(
                range(1, len(rational_train_acc_list) + 1),
                rational_train_acc_list,
                label="Rational",
            )
            train_acc_axs[row, col].set_title(f"n. layers = {n_layer}")
            train_acc_axs[row, col].set_xlabel("Epoch")
            train_acc_axs[row, col].set_ylabel("Train Accuracy")
            train_acc_axs[row, col].legend()
            train_acc_fig.suptitle(
                f"Train Accuracy: n. hidden feat. = {hidden_features}, lr = {lr}"
            )

            train_loss_axs[row, col].plot(
                range(1, len(relu_train_loss_list) + 1),
                relu_train_loss_list,
                label="ReLU",
            )
            train_loss_axs[row, col].plot(
                range(1, len(rational_train_loss_list) + 1),
                rational_train_loss_list,
                label="Rational",
            )
            train_loss_axs[row, col].set_title(f"n. layers = {n_layer}")
            train_loss_axs[row, col].set_xlabel("Epoch")
            train_loss_axs[row, col].set_ylabel("Train Loss")
            train_loss_axs[row, col].legend()
            train_loss_fig.suptitle(
                f"Train Loss: n. hidden feat. = {hidden_features}, lr = {lr}"
            )

            val_acc_axs[row, col].plot(
                range(1, len(relu_val_acc_list) + 1),
                relu_val_acc_list,
                label="ReLU",
            )
            val_acc_axs[row, col].plot(
                range(1, len(rational_val_acc_list) + 1),
                rational_val_acc_list,
                label="Rational",
            )
            val_acc_axs[row, col].set_title(f"n. layers = {n_layer}")
            val_acc_axs[row, col].set_xlabel("Epoch")
            val_acc_axs[row, col].set_ylabel("Validation Accuracy")
            val_acc_axs[row, col].legend()
            val_acc_fig.suptitle(
                f"Validation Accuracy: n. hidden feat. = {hidden_features}, lr = {lr}"
            )

            val_loss_axs[row, col].plot(
                range(1, len(relu_val_loss_list) + 1),
                relu_val_loss_list,
                label="ReLU",
            )
            val_loss_axs[row, col].plot(
                range(1, len(rational_val_loss_list) + 1),
                rational_val_loss_list,
                label="Rational",
            )
            val_loss_axs[row, col].set_title(f"n. layers = {n_layer}")
            val_loss_axs[row, col].set_xlabel("Epoch")
            val_loss_axs[row, col].set_ylabel("Validation Loss")
            val_loss_axs[row, col].legend()
            val_loss_fig.suptitle(
                f"Validation Loss: n. hidden feat. = {hidden_features}, lr = {lr}"
            )

    test_acc_fig.savefig(
        f"figures/trained_with_fixed_seed/epochs_{epochs}/hidden_features_{hidden_features}/learning_rate_{lr}/rational_relu_test_acc_hf_{hidden_features}_lr_{lr}.png"
    )
    test_loss_fig.savefig(
        f"figures/trained_with_fixed_seed/epochs_{epochs}/hidden_features_{hidden_features}/learning_rate_{lr}/rational_relu_test_loss_hf_{hidden_features}_lr_{lr}.png"
    )
    train_acc_fig.savefig(
        f"figures/trained_with_fixed_seed/epochs_{epochs}/hidden_features_{hidden_features}/learning_rate_{lr}/rational_relu_train_acc_hf_{hidden_features}_lr_{lr}.png"
    )
    train_loss_fig.savefig(
        f"figures/trained_with_fixed_seed/epochs_{epochs}/hidden_features_{hidden_features}/learning_rate_{lr}/rational_relu_train_loss_hf_{hidden_features}_lr_{lr}.png"
    )
    val_acc_fig.savefig(
        f"figures/trained_with_fixed_seed/epochs_{epochs}/hidden_features_{hidden_features}/learning_rate_{lr}/rational_relu_val_acc_hf_{hidden_features}_lr_{lr}.png"
    )
    val_loss_fig.savefig(
        f"figures/trained_with_fixed_seed/epochs_{epochs}/hidden_features_{hidden_features}/learning_rate_{lr}/rational_relu_val_loss_hf_{hidden_features}_lr_{lr}.png"
    )

    plt.tight_layout()

    plt.close(test_acc_fig)
    plt.close(test_loss_fig)
    plt.close(train_acc_fig)
    plt.close(train_loss_fig)
    plt.close(val_acc_fig)
    plt.close(val_loss_fig)
    # plt.show()


if __name__ == "__main__":
    """
    The main function runs the experiment.
    """

    # fixed seed
    set_manual_seed()

    params = {
        "rows": 2,
        "cols": 2,
        "n_layers": [[3, 4], [5, 10]],
        "relu": torch.nn.ReLU(),
        "hidden_features": 150,
        "learning_rate": 5e-3,
        "epochs": 400,
    }

    epochs_list = [200, 400, 1000]
    hf_list = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    lr_list = [5e-3, 1e-3, 5e-4, 1e-4]

    for num_epoch in epochs_list:
        print(
            f"=================== Start with epochs = {num_epoch} ==========================="
        )
        for num_hf in hf_list:
            print(
                f"=================== Start with hidden features = {num_hf} ==========================="
            )
            for lr in lr_list:
                print(
                    f"=================== Start with learning rate = {lr} ==========================="
                )
                params["epochs"] = num_epoch
                params["hidden_features"] = num_hf
                params["learning_rate"] = lr

                run_cases(params)
