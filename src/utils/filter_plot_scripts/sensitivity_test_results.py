import os 
from os import walk
import matplotlib.pyplot as plt
import matplotlib
import re
import operator
import numpy as np
from heatmap import heatmap, annotate_heatmap

matplotlib.rcParams.update(
    {
        "font.family": "serif",  # Font family to be used
        "font.serif": "Times New Roman",  # Times New Roman
        "text.usetex": True,  # Render texts/mathematics using pdflatex compiler
        "legend.fontsize": 80,
        "text.latex.preamble": r"\usepackage{times}"
    }
)


def setup_neurips_plotting():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",  # STIX fonts are compatible with Times New Roman
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })


# Call this function at the beginning of your script
setup_neurips_plotting()

print(plt.rcParams['font.serif'])
print(plt.rcParams['font.family'])


results = []
folder = os.getcwd()+"/log_files/experiment800_node_classification_ds_Cora_type_Planetoid_layers_4_Model_SAGEConv/seed0/"
hypparRegex = r'_hf_[0-9]+_layers_[0-9]+_cl_True_ncl_[0-9]+_'
hfRegex = r'hf_[0-9]+'
nclRegex = r'ncl_[0-9]+'
accRegex = r'Max Test Accuracy: [0-9].+'
filenames = sorted(next(walk(folder), (None, None, []))[2])
for file in filenames:
    tmp = re.findall(hypparRegex, file)[0]
    hf = re.findall(hfRegex, tmp)[0]
    ncl = re.findall(nclRegex, tmp)[0]
    lines = open(folder+file).readlines()
    result = [re.findall(accRegex, line) for line in lines][-1][0].split(': ')[1].split(' - ')[0]
    results.append({'hf': int(hf.split('_')[1]), 'ncl': int(ncl.split('_')[1]), 'test_acc': float(result)})
results = sorted(results, key=lambda k: (k['hf'], k['ncl']))
# results = [elm['test_acc'] for elm in results]

# Data
data = np.array([elm['test_acc'] for elm in results], dtype=np.float32)
data = np.reshape(data, (14, 14))
data[data < 0.93] = 0.93
# print(data.shape)
# Labels
# xlabs = [elm['hf'] for elm in results]
# ylabs = [elm['ncl'] for elm in results]
xlabs = [str(20*i) for i in np.arange(1,15)]
ylabs = [str(i) for i in np.arange(1,15)]


# print(len(xlabs))
# print(len(ylabs))
# Heat map
fig, ax = plt.subplots()
# Set label for the axes
ax.set_xlabel("Hidden Features")
ax.set_ylabel("#Clusters")

im, cbar = heatmap(data, row_labels=xlabs, col_labels=ylabs,
                   ax=ax, cmap="YlGn", cbarlabel="Test Accuracy")
# texts = annotate_heatmap(im, valfmt = "{x:.3f}")

plt.savefig("sensitivy_analysis.pdf", dpi=300)
