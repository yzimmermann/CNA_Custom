import numpy as np
import pandas as pd


def store_data_as_csv(data, filename="heatmap_data.csv"):
    """
    Store the given data in a CSV file.

    Parameters:
    - data: The 2D array or list containing the data.
    - filename: The name of the CSV file (default is "heatmap_data.csv").
    """
    # Convert data to NumPy array
    data_array = np.array(data)

    # Create a DataFrame
    df = pd.DataFrame(
        data_array,
        columns=[2, 3, 4, 5, 6],
        index=[
            "ARMA",
            "ChebGCN",
            "DNA",
            "FeaSt",
            "GAT",
            "GCN",
            "GGNN",
            "GraphSAGE",
            "HighOrder",
            "HyperGraph",
        ],
    )

    # Save DataFrame to CSV
    df.to_csv(filename, index_label="Model")


# Example usage:
data = [
    [0.629, 0.860, 0.608, 0.305, 0.004],
    [0.557, 0.756, 0.138, 0.024, 0.018],
    [0.665, 0.352, 0.347, 0.172, 0.096],
    [0.778, 0.770, 0.677, 0.182, 0.072],
    [0.794, 0.704, 0.232, 0.047, 0.005],
    [0.796, 0.765, 0.714, 0.602, 0.289],
    [0.661, 0.078, 0.021, 0.033, 0.039],
    [0.925, 0.816, 0.632, 0.303, 0.053],
    [0.629, 0.145, 0.023, 0.004, 0.012],
    [0.828, 0.742, 0.493, 0.046, 0.023],
]

store_data_as_csv(data)
