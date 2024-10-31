import fast_pytorch_kmeans as fpk
import matplotlib.animation as animation
import torch.nn.functional as F
import torch.optim
from activations.torch import Rational

from rational_power_mean import *
#from power_mean import *
from rationals import RationalsModel
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.decomposition import PCA
from torch_geometric.nn import GCNConv
from utils import *
import os

# Load the Cora dataset
dataset = load_dataset("Cora")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(torch.nn.Module):
    def __init__(
        self,
        activations,
        input_features,
        output_features,
        hidden_features,
        num_layer,
        num_clusters=7,
    ):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the attributes of an object, which are sometimes called
        fields or properties.

        :param self: Represent the instance of the class
        :param activations: Specify the activation function to be used in the gcn
        :param input_features: Specify the number of features in the input data
        :param output_features: Specify the number of output nodes in the final layer
        :param hidden_features: Define the number of nodes in each hidden layer
        :param num_layer: Define the number of layers in the network
        :param num_clusters: Specify the number of clusters to be used in clustering
        """
        super(Net, self).__init__()

        self.conv_list = torch.nn.ModuleList([])
        self.num_layer = num_layer
        self.conv_list.append(GCNConv(input_features, hidden_features))
        for i in range(self.num_layer - 2):
            self.conv_list.append(GCNConv(hidden_features, hidden_features))
        self.conv_list.append(GCNConv(hidden_features, output_features))
        self.activations = activations
        self.data_np = []
        self.centroids = []
        self.labels = []
        self.cluster_indices = []
        self.num_clusters = num_clusters
        self.current_centroids = []
        self.pca = PCA(n_components=2)
        self.current_data = []
        self.features = dataset[0].x
        self.mask = torch.ones(self.features.size(0), self.features.size(0)).numpy()
        self.act_params = None

    def forward(self, x, edge_index, flag=False, with_clustering=False):
        """
        The forward function of the model.

        :param self: Represent the instance of the class
        :param x: Pass the node features to the first layer of convolution
        :param edge_index: Pass the graph structure to the convolutional layer
        :param flag: Indicate whether the clustering is performed for the first time or not
        :return: The log_softmax of the last layer,
        """
        self.data_np.clear()
        self.current_data.clear()
        for i in range(self.num_layer - 1):
            x = self.conv_list[i](x, edge_index)
            embeddings = torch.clone(x)
            if with_clustering:
                if flag:
                    kmeans = fpk.KMeans(
                        n_clusters=self.num_clusters,
                        mode="cosine",
                    )
                    if i == 0:
                        self.centroids.clear()
                        self.labels.clear()
                    self.labels.append(kmeans.fit_predict(x))
                    self.cluster_indices.clear()
                    [
                        self.cluster_indices.append(
                            torch.nonzero(self.labels[i] == l).squeeze()
                        )
                        for l in range(self.num_clusters)
                    ]
                    self.centroids.append(kmeans.centroids)
                    # self.data_np.append(
                    #    self.pca.fit_transform(self.centroids[i].detach().cpu().numpy())
                    # )
                    # self.current_data.append(self.data_np[i].copy())

                else:
                    self.current_centroids.clear()
                    [
                        self.current_centroids.append(
                            torch.mean(x[self.cluster_indices[c]], dim=0)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        for c in range(self.num_clusters)
                    ]
                    # self.current_data.append(self.pca.fit_transform(self.current_centroids))
                    #self.data_np.append(
                    #    self.pca.fit_transform(self.centroids[i].detach().cpu().numpy())
                    #)
                    self.data_np.append([])

                for c in range(self.num_clusters):
                    x[self.cluster_indices[c]] = self.activations[c](x[self.cluster_indices[c]].clone())
            else:
                x = self.activations[0](x)
                #x = F.dropout(x, training=self.training)

        x = self.conv_list[self.num_layer - 1](x, edge_index)

        return F.log_softmax(x, dim=1), self.data_np, self.current_data, embeddings


def train(
    model,
    train_optimizer,
    activation_optimizer,
    data,
    flag=False,
    with_clustering=False,
    activations=None
):
    """
    The train function takes in a model, optimizer, and data.
    It then sets the model to train mode and zeros out the gradients
    of all parameters. The output is calculated by passing in the features (x),
    edge indices (edge_index), and flag into our model. The loss is calculated
    using negative log likelihood loss on only those nodes that are training
    nodes (train_mask). The backward pass calculates the gradients for each
    parameter based on this loss function. Finally, we step forward with our
    optimizer to update weights based on these gradients.

    :param model: Define the model architecture
    :param optimizer: Optimize the model
    :param data: Pass the data to the model
    :param flag: Indicate whether the model is in training or testing mode
    :return: The accuracy, loss, data_np and current_data
    """
    model.train()
    train_optimizer.zero_grad()
    if activation_optimizer:
        activation_optimizer.zero_grad()
    out, data_np, current_data, embeddings = model(
        data.x, data.edge_index, flag, with_clustering=with_clustering
    )
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()

    # clip the gradients
    if model.act_params:
        torch.nn.utils.clip_grad_norm_(model.act_params, max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    train_optimizer.step()
    if activation_optimizer:
        activation_optimizer.step()
    pred = out.argmax(dim=1)
    correct = (pred[data.train_mask] == data.y[data.train_mask]).sum()
    acc = float(correct) / data.train_mask.sum()
    return acc, loss.item(), data_np, current_data, embeddings


def validation(model, data, flag=False, with_clustering=False):
    """
    The validation function is used to evaluate the model on a validation
    set.
    It takes as input:
        - The model, which should be in eval mode (model.eval())
        - A data object containing the validation set and other information
        about it (e.g., mask)

    :param model: Pass the model to the validation function
    :param data: Pass the data to the model
    :return: The accuracy and the loss
    """
    model.eval()
    out, _, _, _ = model(data.x, data.edge_index, flag, with_clustering=with_clustering)
    loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
    pred = out.argmax(dim=1)
    correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
    acc = float(correct) / data.val_mask.sum()
    return acc, loss.item()


def test(model, data, flag=False, with_clustering=False):
    """
    The test function is used to evaluate the performance of a model on a dataset.
    It takes as input:
        - model: The trained model that we want to test.
        - data: The dataset on which we want to test the performance of our trained
                model. This is an object from the PyTorch Geometric library, and it
                contains all information about our graph (nodes, edges, features etc.)

    :param model: Pass the model to be trained
    :param data: Pass the data object to the test function
    :return: The test accuracy and the test loss
    """
    model.eval()
    out, _, _, _ = model(data.x, data.edge_index, flag, with_clustering=with_clustering)
    loss = F.nll_loss(out[data.test_mask], data.y[data.test_mask])
    pred = out.argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = float(correct) / data.test_mask.sum()
    return acc, loss.item()


def run_experiment(
    activation,
    hidden_features=50,
    num_layers=3,
    lr=5e-3,
    weight_decay=5e-4,
    epochs=200,
    num_clusters=4,
    visualize=False,
    with_clustering=False,
):
    """
    The run_experiment function takes in a list of activations, the number
    of hidden features, the number of layers (including input and output),
    the learning rate, weight decay value, number of epochs to train for
    and whether or not to visualize. It returns a tuple containing test accuracy
    list, test loss list, training accuracy list ,training loss list , validation
    accuracy and validation loss lists.

    :param activations: Specify the activation function used in each layer
    :param hidden_features: Set the number of features in each layer
    :param num_layers: Specify the number of hidden layers in the network
    :param lr: Set the learning rate of the optimizer
    :param weight_decay: Control the strength of regularization
    :param epochs: Specify the number of epochs to train for
    :param num_clusters: Determine the number of centroids to be used in the k-means clustering
    :param visualize: Generate a gif of the voronoi diagrams for each epoch
    :return: A tuple of 6 lists:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    activations = []
    if activation == "relu":
        with_clustering = False
        for act in range(num_clusters):
            activations.append(torch.nn.ReLU())
    elif activation == "relu_cluster":
        with_clustering = True
        for act in range(num_clusters):
            activations.append(torch.nn.ReLU())
    elif activation == "rational_relu":
        with_clustering = False
        for act in range(num_clusters):
            activations.append(Rational("relu"))
    elif activation == "rational_relu_cluster":
        with_clustering = True
        for act in range(num_clusters):
            activations.append(Rational("relu"))
    elif activation == "rpm_4_activations":
        with_clustering = False
        rpm = RationalPowerMean(["relu", "relu", "relu", "relu"]).cuda()
        activations.append(rpm)
    elif activation == "rpm_4_activations_cluster":
        with_clustering = True
        rpm = RationalPowerMean(["relu", "relu", "relu", "relu"]).cuda()
        for act in range(num_clusters):
            activations.append(rpm)
    elif activation == "our_rational_relu":
        with_clustering = False
        [activations.append(RationalsModel(n=5, m=5, function="relu", use_coefficients=True)) for _ in range(num_clusters)]
    elif activation == "our_rational_relu_cluster":
        with_clustering = True
        [activations.append(RationalsModel(n=5, m=5, function="relu", use_coefficients=True)) for _ in range(num_clusters)]

    model = Net(
        activations,
        dataset.num_features,
        dataset.num_classes,
        hidden_features,
        num_layers,
        num_clusters,
    ).to(device)
    train_optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    print(model)
    model.act_params = []
    if activation == "relu" or activation == "relu_cluster":
        model.act_params = None
        activation_optimizer = None
    elif activation == "rpm_4_activations" or activation == "rpm_4_activations_cluster":
        for act_list in activations:
            [model.act_params.extend(act.parameters()) for act in act_list.rationals]
        activation_optimizer = torch.optim.Adam(model.act_params, lr=0.00001)

    elif activation == "rational_relu" or activation == "rational_relu_cluster":
        [model.act_params.extend(act.parameters()) for act in activations]
        activation_optimizer = torch.optim.Adam(model.act_params, lr=0.01)

    else:
        [model.act_params.extend(act.parameters()) for act in activations]
        activation_optimizer = torch.optim.Adam(model.act_params, lr=0.01, weight_decay=0.001)

    test_acc_list = []
    test_loss_list = []
    val_acc_list = []
    val_loss_list = []
    train_acc_list = []
    train_loss_list = []
    if visualize:
        fig, ax = plt.subplots(
            nrows=1, ncols=num_layers - 1, figsize=((num_layers - 1) * 10, 10)
        )
    epoch_values = []
    centroids_dict = {}
    curr_centroids_dict = {}
    for layer in range(num_layers - 1):
        centroids_dict[str(layer)] = []
        curr_centroids_dict[str(layer)] = []

    def update_plot(epoch):
        """
        The update_plot function is called by the FuncAnimation function
        in order to update the plot. It takes an epoch value as input,
        and uses that to index into the centroids_dict and curr_centroids_dict
        dictionaries. The Voronoi diagram for each layer is then plotted using
        matplotlib's voronoi_plot_2d function, with a few modifications:
            - The show points argument has been set to True so that we can see
            where each point lies on the plot. This helps us visualize how
              well our model is learning over time (i.e., if

        :param epoch: Determine which epoch to plot
        :return: A dictionary with the current centroids
        """
        for layer in range(num_layers - 1):
            if num_layers > 2:
                ax[layer].cla()
            else:
                ax.cla()
            np_data_list = centroids_dict[str(layer)]
            current_data_list = curr_centroids_dict[str(layer)]
            vor = Voronoi(np_data_list[epoch])

            if num_layers > 2:
                voronoi_plot_2d(
                    vor, ax=ax[layer], show_points=True, show_vertices=False, s=1
                )
            else:
                voronoi_plot_2d(vor, ax=ax, show_points=True, show_vertices=False, s=1)

            # Add points from current_data_list to the plot with a different color
            current_data_points = current_data_list[epoch]
            if num_layers > 2:
                ax[layer].scatter(
                    current_data_points[:, 0],
                    current_data_points[:, 1],
                    c="red",
                    label="current features",
                )
            else:
                ax.scatter(
                    current_data_points[:, 0],
                    current_data_points[:, 1],
                    c="red",
                    label="current features",
                )

            for r in range(len(vor.point_region)):
                region = vor.regions[vor.point_region[r]]
                if not -1 in region:
                    polygon = [vor.vertices[i] for i in region]
                    if num_layers > 2:
                        ax[layer].fill(*zip(*polygon), "white", alpha=0.5)
                    else:
                        ax.fill(*zip(*polygon), "white", alpha=0.5)

            if num_layers > 2:
                ax[layer].set_title(
                    f"Output of layer {layer+1}, Epoch {epoch_values[epoch]}"
                )
                ax[layer].legend(loc="upper right")
            else:
                ax.set_title(
                    f"Output of layer {layer + 1}, Epoch {epoch_values[epoch]}"
                )
                ax.legend(loc="upper right")

    best_metric = float("inf")
    patience = 100

    for epoch in range(1, epochs + 1):
        if epoch == 1 or epoch % 50 == 0:
            train_acc, train_loss, data_np, current_data, embeddings = train(
                model,
                train_optimizer,
                activation_optimizer,
                dataset[0].to(device),
                flag=True,
                with_clustering=with_clustering,
                activations=activations
            )
            val_acc, val_loss = validation(
                model, dataset[0].to(device), flag=True, with_clustering=with_clustering
            )
            test_acc, test_loss = test(
                model, dataset[0].to(device), flag=True, with_clustering=with_clustering
            )
        else:
            train_acc, train_loss, data_np, current_data, embeddings = train(
                model,
                train_optimizer,
                activation_optimizer,
                dataset[0].to(device),
                flag=False,
                with_clustering=with_clustering,
            )
            val_acc, val_loss = validation(
                model,
                dataset[0].to(device),
                flag=False,
                with_clustering=with_clustering,
            )
            test_acc, test_loss = test(
                model,
                dataset[0].to(device),
                flag=False,
                with_clustering=with_clustering,
            )
            # Check if validation metric has improved
        """
        if val_acc < best_metric:
            best_metric = val_acc
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        # Early stopping check
        if epochs_since_improvement >= patience:
            print("Early stopping triggered. No improvement in validation metric.")
            break
        """
        # if with_clustering:
        # for layer in range(num_layers - 1):
        # centroids_dict[str(layer)].append(data_np[layer])
        # curr_centroids_dict[str(layer)].append(current_data[layer])

        train_acc_list.append(train_acc.cpu())
        val_acc_list.append(val_acc.cpu())
        train_loss_list.append(train_loss)
        test_acc_list.append(test_acc.cpu())
        test_loss_list.append(test_loss)
        val_loss_list.append(val_loss)

        epoch_values.append(epoch)
        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Test Acc: {test_acc:.4f}"
            )

    _, _, _, embeddings = model(
        dataset[0].x.to(device), dataset[0].edge_index.to(device)
    )
    # mean_avg_distance = compute_mad(embeddings.detach().cpu().numpy(), dataset[0].train_mask.detach().cpu().numpy())
    mean_avg_distance = None

    if visualize and with_clustering:
        ani = animation.FuncAnimation(
            fig, update_plot, frames=len(epoch_values), interval=1000
        )
        ani.save(
            f"voronoi_layers_{num_layers - 1}.gif",
            writer="ffmpeg",
            fps=300,
        )
        plt.close()

    return (
        test_acc_list,
        test_loss_list,
        train_acc_list,
        train_loss_list,
        val_acc_list,
        val_loss_list,
        mean_avg_distance,
    )


def run_cases(params):
    """
    The run_cases function takes in a dictionary of parameters and
    runs the experiment with those parameters. The function then
    plots the results for each activation function, as well as for
    each metric (accuracy/loss).

    :param params: Pass the parameters to the run_cases function
    :return: A dictionary with the following keys and values:
    """
    n_layers = params["n_layers"]
    # set parameters
    activations = params["activations"]
    hidden_features = params["hidden_features"]
    lr = params["learning_rate"]
    epochs = params["epochs"]
    num_clusters = params["num_clusters"]
    visualize = params["visualize"]
    with_clustering = params["with_clustering"]

    mean_avg_distances = np.zeros((len(activations), len(n_layers)))
    test_acc_results = {activation: [] for activation in activations}
    for i, activation in enumerate(activations):
        print(
            f"=================== Start with activation function = {activation} ==========================="
        )

        for j, num_layers in enumerate(n_layers):
            #torch.cuda.empty_cache()
            #random.seed(1)
            #np.random.seed(1)
            #torch.manual_seed(1)
            #torch.cuda.manual_seed_all(1)

            print(
                f"=================== Start with layer structure = {num_layers} ==========================="
            )
            (
                relu_test_acc_list,
                relu_test_loss_list,
                relu_train_acc_list,
                relu_train_loss_list,
                relu_val_acc_list,
                relu_val_loss_list,
                mean_avg_distance,
            ) = run_experiment(
                activation,
                num_layers=num_layers,
                hidden_features=hidden_features,
                lr=lr,
                epochs=epochs,
                num_clusters=num_clusters,
                visualize=visualize,
                with_clustering=with_clustering,
            )
            mean_avg_distances[i, j] = mean_avg_distance
            test_acc_results[activation].append(max(relu_test_acc_list))

        if visualize:
            test_acc_fig, test_acc_axs = plt.subplots(figsize=(14, 10))
            test_loss_fig, test_loss_axs = plt.subplots(figsize=(14, 10))
            train_loss_fig, train_loss_axs = plt.subplots(figsize=(14, 10))
            val_acc_fig, val_acc_axs = plt.subplots(figsize=(14, 10))
            val_loss_fig, val_loss_axs = plt.subplots(figsize=(14, 10))
            test_acc_axs.plot(
                range(1, len(relu_test_acc_list) + 1), relu_test_acc_list, label="ReLU"
            )
            test_acc_axs.set_title(f"n. layers = {n_layers}")
            test_acc_axs.set_xlabel("Epoch")
            test_acc_axs.set_ylabel("Test Accuracy")
            test_acc_axs.legend()
            test_acc_fig.suptitle(
                f"Test Accuracy: n. hidden feat. = {hidden_features}, lr = {lr}"
            )

            test_loss_axs.plot(
                range(1, len(relu_test_loss_list) + 1),
                relu_test_loss_list,
                label="ReLU",
            )

            test_loss_axs.set_title(f"n. layers = {n_layers}")
            test_loss_axs.set_xlabel("Epoch")
            test_loss_axs.set_ylabel("Test Loss")
            test_loss_axs.legend()
            test_loss_fig.suptitle(
                f"Test Loss: n. hidden feat. = {hidden_features}, lr = {lr}"
            )

            train_loss_axs.plot(
                range(1, len(relu_train_loss_list) + 1),
                relu_train_loss_list,
                label="ReLU",
            )

            train_loss_axs.set_title(f"n. layers = {n_layers}")
            train_loss_axs.set_xlabel("Epoch")
            train_loss_axs.set_ylabel("Train Loss")
            train_loss_axs.legend()
            train_loss_fig.suptitle(
                f"Train Loss: n. hidden feat. = {hidden_features}, lr = {lr}"
            )

            val_acc_axs.plot(
                range(1, len(relu_val_acc_list) + 1),
                relu_val_acc_list,
                label="ReLU",
            )

            val_acc_axs.set_title(f"n. layers = {n_layers}")
            val_acc_axs.set_xlabel("Epoch")
            val_acc_axs.set_ylabel("Validation Accuracy")
            val_acc_axs.legend()
            val_acc_fig.suptitle(
                f"Validation Accuracy: n. hidden feat. = {hidden_features}, lr = {lr}"
            )

            val_loss_axs.plot(
                range(1, len(relu_val_loss_list) + 1),
                relu_val_loss_list,
                label="ReLU",
            )

            val_loss_axs.set_title(f"n. layers = {n_layers}")
            val_loss_axs.set_xlabel("Epoch")
            val_loss_axs.set_ylabel("Validation Loss")
            val_loss_axs.legend()
            val_loss_fig.suptitle(
                f"Validation Loss: n. hidden feat. = {hidden_features}, lr = {lr}"
            )
            plt.show()

    # Plotting
    plt.figure(figsize=(10, 6))

    for activation, test_acc_list in test_acc_results.items():
        plt.plot(n_layers, test_acc_list, label=activation)

    plt.xlabel("Number of Layers")
    plt.ylabel("Test Accuracy")
    plt.title(f"Test Accuracy per Layer Structure")
    plt.xticks(n_layers)
    plt.legend()
    plt.grid(True)
    save_path = f"clustering_plots/server/epochs_{epochs}/num_cluster_{num_clusters}/hidden_features_{hidden_features}/"
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(
        save_path + f"plot_with_ep_{epochs}_cluster_{num_clusters}_hf_{hidden_features}_lr_{lr}.png"
    )
    #plt.show()
    """
    fig = plt.figure(figsize=(10, 16))
    ax_mattrix = fig.add_subplot(111)
    im = ax_mattrix.imshow(mean_avg_distances, cmap='YlGnBu', interpolation='nearest')

    # Add the text
    for i in range(len(activations)):
        for j in range(len(n_layers)):
            text = ax_mattrix.text(j, i, round(mean_avg_distances[i, j], 2),
                                   ha="center", va="center", color="gray")

    plt.xlabel('Layer Number')
    plt.ylabel('Activation Function')
    plt.xticks(np.arange(len(n_layers)), n_layers)
    plt.yticks(np.arange(len(activations)), activations)

    fig.colorbar(im)
    plt.title(f'MAD with {num_clusters} clusters')
    #plt.show()
    """


if __name__ == "__main__":
    """
    The main function runs the experiment.
    """

    # fixed seed
    set_manual_seed()
    #torch.autograd.set_detect_anomaly(True)

    params = {
        "n_layers": [3, 4, 6, 10],  # min. 2 , 6, 10, 15
        # "activations": ["relu", "relu_cluster", "rational_relu", "rational_relu_cluster",
        # "rpm_4_activations", "rpm_4_activations_cluster", "our_rational_relu",
        # "our_rational_relu_cluster"],
        #"activations": ["relu", "relu_cluster", "rational_relu", "",
        #                "rpm_4_activations", "rpm_4_activations_cluster", "our_rational_relu",
        #                "our_rational_relu_cluster"
        #                ],
        "activations": ["relu", "rational_relu", "rational_relu_cluster"],
        "hidden_features": [100],  # test
        "learning_rate": 0.0001,  # test
        "epochs": [400],
        "num_clusters": [7],
        "visualize": False,
        "with_clustering": False,
    }

    epochs_list = params["epochs"]
    hf_list = params["hidden_features"]
    lr_list = [params["learning_rate"]]
    clusters_list = params["num_clusters"]

    for num_epoch in epochs_list:
        print(
            f"=================== Start with epochs = {num_epoch} ==========================="
        )
        for cluster in clusters_list:
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
                    params["num_clusters"] = cluster

                    run_cases(params)
