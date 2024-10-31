import json

import pandas as pd


def generate_latex_table(file_paths, metrics):
    """Generate a LaTeX table from the given data.

    Args:
        file_paths (list): A list of file paths to the JSON files containing the data.
        metrics (list): A list of metrics to be included in the table.

    Returns:
        str: A string containing the LaTeX table.
    """
    data_frames = []

    for file_path in file_paths:
        with open(file_path, "r") as json_file:
            data = json.load(json_file)

        value_to_filter = 128
        filtered_data = [
            entry for entry in data if entry.get("layers") != value_to_filter
        ]

        df = pd.DataFrame(filtered_data)
        data_frames.append(df)

    latex_table = "\\begin{tabular}{|c|"
    for _ in metrics:
        latex_table += "c|"
    latex_table += "}\n\\hline\n"

    # Add header
    latex_table += "Layers"
    for metric in metrics:
        latex_table += f" & {metric}"
    latex_table += "\\\\\n\\hline\n"

    # Add data
    for layer in df["layers"].unique():
        latex_table += f"{layer}"

        for metric in metrics:
            mean_value = df[df["layers"] == layer][metric].mean()
            std_value = df[df["layers"] == layer][metric].std()

            latex_table += f" & {mean_value:.3f} $\\pm$ {std_value:.3f}"

        latex_table += "\\\\\n"

    # Add footer
    latex_table += "\\hline\n\\end{tabular}"

    return latex_table


if __name__ == "__main__":
    file_paths_to_process = [
        "data_default_solution_without_cluster_gcn_cora.json",
        "result_data_our_solution_default_subsets_gcn_cora.json",
        "result_data_50_25_25.json",
        "result_data_70_15_15.json",
        "result_data_80_10_10.json",
    ]

    metrics_to_process = ["max_test_accuracy", "MADC"]

    latex_table_result = generate_latex_table(file_paths_to_process, metrics_to_process)

    print(latex_table_result)
