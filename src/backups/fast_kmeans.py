import time

import faiss
import fast_pytorch_kmeans as fpk
from sklearn.cluster import KMeans as sk_kmean
import matplotlib.pyplot as plt
import numpy as np
import torch


class KMeansComparison:
    def __init__(
        self,
        n_samples=100000,
        n_features=256,
        max_iterations=100,
        tol=-1,
        minibatch=None,
    ):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the initial values for all of the attributes that are defined in this class.
        The self parameter refers to an instance of a class, and it's required by convention.

        :param self: Represent the instance of the class
        :param n_samples: Determine the number of samples in the dataset
        :param n_features: Specify the number of features in the data
        :param max_iterations: Determine the maximum number of iterations that the k-means algorithm will run for
        :param tol: Determine when to stop the algorithm
        :param minibatch: Determine whether the algorithm is run in minibatch mode or not
        :return: The object itself
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_samples = n_samples
        self.n_features = n_features
        self.max_iterations = max_iterations
        self.tol = tol
        self.minibatch = minibatch
        self.x = []
        self.times_fkm = []
        self.times_faiss = []
        self.sklearn_time = []

    def generate_random_data(self):
        """
        The generate_random_data function generates random data for the model to train on.

        :param self: Represent the instance of the class
        :return: A numpy array of random numbers
        """
        self.data_numpy = np.random.rand(self.n_samples, self.n_features).astype(
            np.float32
        )
        self.data = torch.from_numpy(self.data_numpy).type(torch.float).to(self.device)

    def run_fast_pytorch_kmeans(self):
        """
        The run_fast_pytorch_kmeans function is used to run the Fast PyTorch KMeans algorithm on the data.
            It takes in a self parameter, which is an instance of the class that contains all of our data and parameters.
            The function then runs through a for loop from 1 to 10, incrementing by 1 each time. This will be used as our exponent value for 2^i (where i = exponent).
            We then append this value to x, which will be used later when plotting our results graphically.

        :param self: Access the attributes and methods of a class
        :return: The time it takes to run the fast pytorch kmeans algorithm
        """
        for i in range(1, 11):
            self.x.append(2**i)
            start_time = time.time()
            fpk.KMeans(
                n_clusters=2**i,
                max_iter=self.max_iterations,
                tol=self.tol,
                minibatch=self.minibatch,
            ).fit(self.data)
            self.times_fkm.append(time.time() - start_time)

    def train_faiss_kmeans(self):
        """
        The train_faiss_kmeans function trains the k-means model using the FAISS library.

        :param self: Represent the instance of the class
        :return: The clustering time
        """
        for n_clusters in self.x:
            # Create the k-means index
            index = faiss.IndexFlatL2(self.n_features)

            # Create the k-means trainer
            kmeans = faiss.Clustering(self.n_features, n_clusters)

            # Set the k-means parameters
            kmeans.verbose = False
            kmeans.niter = 20
            kmeans.max_points_per_centroid = 100000

            # Train the k-means model and measure the time
            t0 = time.time()
            kmeans.train(self.data_numpy, index)
            t1 = time.time()

            # Compute the clustering time
            clustering_time = t1 - t0
            self.times_faiss.append(clustering_time)




    def train_sklearn_kmeans(self):
        """
        The train_sklearn_kmeans function trains the k-means model using the sklearn library.

        :param self: Represent the instance of the class
        :return: The clustering time
        """
        for n_clusters in self.x:
            # Create the k-means trainer
            kmeans_cpu = sk_kmean(n_clusters=n_clusters, n_init=1, max_iter=self.max_iterations, tol=-1, random_state=42)

            t0 = time.time()

            kmeans_cpu.fit(self.data_numpy)
            t1 = time.time()

            # Compute the clustering time
            clustering_time = t1 - t0

            self.sklearn_time.append(clustering_time)

    def plot_results(self):
        """
        The plot_results function creates a figure with two subplots. The first subplot plots the time it takes to run
        fast-pytorch KMeans and faiss KMeans for each number of clusters in the x array. The second subplot plots the logarithm
        of these times, which is useful for comparing algorithms that have very different running times.

        :param self: Represent the instance of the class
        :return: A plot of the time it takes to run fast-pytorch kmeans and faiss kmeans
        """
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot the first subplot
        ax1.plot(self.x, self.times_fkm, label="fast-pytorch KMeans")
        ax1.plot(self.x, self.times_faiss, label="faiss KMeans")
        ax1.plot(self.x, self.sklearn_time, label="sklearn KMeans")
        ax1.set_xlabel("Clusters")
        ax1.set_ylabel("Time (seconds)")
        ax1.set_title("fast-pytorch KMeans Speed Comparison")

        # Plot the second subplot
        ax2.plot(self.x, np.log(self.times_fkm), label="fast-pytorch KMeans")
        ax2.plot(self.x, np.log(self.times_faiss), label="faiss KMeans")
        ax2.plot(self.x, np.log(self.sklearn_time), label="sklearn KMeans")
        ax2.set_xlabel("Clusters")
        ax2.set_ylabel("log(Time)")
        ax2.set_title("fast-pytorch KMeans Speed Comparison")

        # Adjust spacing between subplots
        fig.subplots_adjust(wspace=0.3)

        # Show the plot
        plt.legend()
        plt.show()

        # store the figure
        fig.savefig('figures/k_means_comparison_sklearn_faiss_fast.png', dpi=fig.dpi)


if __name__ == "__main__":
    kmeans_comparison = KMeansComparison()
    kmeans_comparison.generate_random_data()
    kmeans_comparison.run_fast_pytorch_kmeans()
    kmeans_comparison.train_faiss_kmeans()
    kmeans_comparison.train_sklearn_kmeans()
    kmeans_comparison.plot_results()
