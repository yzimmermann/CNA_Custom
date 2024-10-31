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
#plt.style.use(['science','grid'])
#matplotlib.rcParams.update(
#        {
#            "font.family": "serif",  # Font family to be used
#            "text.usetex": True,  # Render texts/mathematics using pdflatex compiler
#            "legend.fontsize": 22
 #       }
  #  )
# Step 3: Create DataFrame from the given data
# Define colormap
# Define custom colormap
#custom_palette = ["lightgreen", "green", "darkgreen", "darkgreen", "cyan", "cyan",
#                  "cyan", "orange", "orange", "red", "red", "blue", "red", "red", "blue"]


models = [
    "GraphSAGE",
    "GCN",
    "NCA (SAGEConv)",
    "DAGNN",
    "DeeperGCN",
    "GCNII",
    "GAT",
    "UniMP",
    "RevGCN-Deep",
    "RevGAT-Wide",
    "RevGAT-SelfKD",
    "NCA (GCNConv)",
    "SimTeG+TAPE + GraphSAGE",
    "TAPE + RevGAT",
    "SimTeG + TAPE + RevGAT"
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
    74.64,
    77.48,
    77.50,
    78.03
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
    389200,
    1380000000,
    280280000,
    1386000000,
]
log_params = np.log10(params)
params = log_params

data = pd.DataFrame({"Model": models, "Accuracy": accuracy, "Params": params})

# Define the x range
x_min = 4.3
x_max = 9.7

# Create a custom colormap
#cmap = plt.cm.get_cmap('inferno')
#norm = plt.Normalize(x_min, x_max)

# Define colors for specific x values
#colors = [cmap(norm(x)) for x in [4.3, 9.7]]

# Set the custom colors for the specific x values
#custom_palette = {4.3: colors[0], 9.7: colors[1]}

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = col_.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


#palette_options = ['YlOrBr', 'RdYlGn_r', 'rocket_r', 'mako_r', 'flare', 'crest', 'magma_r',  'viridis_r', 'Spectral_r', 'coolwarm']
#palette_options_r = ['YlOrBr_r', 'RdYlGn', 'rocket', 'mako', 'flare_r', 'crest_r', 'magma',  'viridis', 'Spectral', 'coolwarm_r']
palette_options = ['coolwarm']
palette_options_r = ['coolwarm_r']
cm = 0
for palette in palette_options:
    # Step 4: Plot the bubble chart
    plt.figure(figsize=(10, 10))
    # Add grid
    #plt.grid(True)

    # Reverse the colormap


    scatter = sns.scatterplot(data=data, x="Params", y="Accuracy", size="Accuracy", sizes=(100, 100), hue="Params", legend=False, palette=palette)

    # Add colorbar and adjust ticks afterwards
    #cbar = plt.colorbar(scatter.collections[0], cax=scatter, orientation='horizontal')
    #cbar = plt.colorbar(scatter.get_children()[0],  orientation='horizontal')
    #cbar.set_ticks(ticks=[-1, 0, 1], labels=['Low', 'Medium', 'High'])


    start_point = (5.8, 71.1)
    end_point = (9.5, 77.2)
    arrow_text = "current"

    # Calculate midpoint between start and end points for text positioning
    text_pos = ((start_point[0] + end_point[0]) / 2, (start_point[1] + end_point[1]) / 2)

    # Calculate rotation angle for text
    angle = np.degrees(np.arctan2(end_point[1] - start_point[1], end_point[0] - start_point[0]))
    num_points = 40
    # Plot arrow with gradient color and text in the middle
    cmap = plt.cm.get_cmap(palette)
    cm += 1
    new_cmap = truncate_colormap(cmap, 0.2, 1.0)
    arrow_colors = [new_cmap(i / num_points) for i in range(num_points)]

    for i in range(num_points - 1):
        plt.plot([start_point[0] + (end_point[0] - start_point[0]) * i / num_points,
                  start_point[0] + (end_point[0] - start_point[0]) * (i + 1) / num_points],
                 [start_point[1] + (end_point[1] - start_point[1]) * i / num_points,
                  start_point[1] + (end_point[1] - start_point[1]) * (i + 1) / num_points],
                 color=arrow_colors[i], linewidth=1, linestyle='--')


    start_point = (4.3, 72.3)  # Example start point
    end_point = (5.6, 76.0)  # Example end point
    num_points = 15
    #cmap = plt.cm.get_cmap('YlOrBr')
    new_cmap = truncate_colormap(cmap, 0.0, 0.4)
    arrow_colors = [new_cmap(i / num_points) for i in range(num_points)]  # Interpolate colors along the arrow

    for i in range(num_points - 1):
        plt.plot([start_point[0] + (end_point[0] - start_point[0]) * i / num_points,
                  start_point[0] + (end_point[0] - start_point[0]) * (i + 1) / num_points],
                 [start_point[1] + (end_point[1] - start_point[1]) * i / num_points,
                  start_point[1] + (end_point[1] - start_point[1]) * (i + 1) / num_points],
                 color=arrow_colors[i], linewidth=1, linestyle='--')


    # Annotate each point
    #for i in range(len(data)):
    #    print(data["Model"][i])
        #plt.annotate(data["Model"][i], (data["Params"][i], data["Accuracy"][i]), weight="bold" if data["Model"][i].startswith("NCA") else "normal", textcoords="offset points", fontsize=6, xytext=(0,5), ha='center')

    plt.ylim(70.5, 80.1)
    plt.xlim(4.0, 9.9)
    # Step 5: Customize the plot
    plt.xlabel(r'$\log_{10}$ (parameters)')
    plt.ylabel("Accuracy (\%)")
    #plt.title("Accuracy vs Learnable Parameters")
    plt.title(palette)

    #plt.savefig("log_params_models_comp_acc_vs_params_variant_new_v2.png", dpi=600)
    # Step 7: Show the plot
    plt.show()
