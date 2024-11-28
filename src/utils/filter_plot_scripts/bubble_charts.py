# Step 1: Install necessary libraries if not already installed
# !pip install matplotlib seaborn scientificplots
import matplotlib
import numpy as np
# Step 2: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as col_
import scienceplots
from matplotlib.colors import LinearSegmentedColormap

plt.style.use(['science','ieee','grid'])
matplotlib.rcParams.update(
        {
            "font.family": "serif",  # Font family to be used
            "font.serif": "Times New Roman", # Times New Roman
            "text.usetex": True,  # Render texts/mathematics using pdflatex compiler
            "legend.fontsize": 80
        }
)

models = [
    "GraphSAGE",
    "GCN",
    "SAGEConv + CNA",
    "DAGNN",
    "DeeperGCN",
    "GCNII",
    "GAT",
    "UniMP",
    "RevGCN-Deep",
    "RevGAT-Wide",
    "RevGAT-SelfKD",
    "GCNConv + CNA",
]

accuracy = [
    71.49,
    71.74,
    71.79,
    72.09,
    72.32,
    72.74,
    73.91,
    73.97,
    73.01,
    74.05,
    74.26,
    74.64
]

params = [
    219000,
    143000,
    34900,
    43900,
    491000,
    2150000,
    1440000,
    687000,
    262000,
    3880000,
    2100000,
    389200
]
log_params = np.log10(params)
params = log_params
print(sorted(params))

data = pd.DataFrame({"Model": models, "Accuracy": accuracy, "Params": params})
print(data.sort_values(by='Params'))
# Define the x range
x_min = 4.3
x_max = 8.0  #9.7

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = col_.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

palette_options = ['coolwarm']
palette_options_r = ['coolwarm_r']
cm = 0
for palette in palette_options:
    # Step 4: Plot the bubble chart
    plt.figure(figsize=(10, 10))

    scatter = sns.scatterplot(data=data, x="Params", y="Accuracy", size="Accuracy", sizes=(500, 500), hue="Params", legend=False, palette=palette)

    start_point = (5.8, 71.1)
    end_point = (9.5, 77.2)

    # Calculate midpoint between start and end points for text positioning
    text_pos = ((start_point[0] + end_point[0]) / 2, (start_point[1] + end_point[1]) / 2)

    # Calculate rotation angle for text
    angle = np.degrees(np.arctan2(end_point[1] - start_point[1], end_point[0] - start_point[0]))
    num_points = 40
    # Plot arrow with gradient color and text in the middle
    cmap = plt.cm.get_cmap(palette)
    cm += 1
    new_cmap = truncate_colormap(cmap, 0.2, 1.0)


    start_point = (4.3, 72.3)  # Example start point
    end_point = (5.6, 76.0)  # Example end point
    num_points = 15
    new_cmap = truncate_colormap(cmap, 0.0, 0.4)

    plt.ylim(70.5, 75.1)
    plt.xlim(4.0, 6.8)
    # Step 5: Customize the plot
    plt.xlabel(r'$\log_{10}$(parameters)')
    plt.ylabel("Accuracy (\%)")
    #plt.title("Accuracy vs Learnable Parameters")

    plt.savefig("log_params_models_comp_acc_vs_params.pdf", dpi=600)
    # Step 7: Show the plot
    #plt.show()
