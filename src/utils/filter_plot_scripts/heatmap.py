import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
   data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # # Plot the heatmap
    # im = ax.imshow(100*data, vmin=68, vmax=75, **kwargs)

    # # Create colorbar
    # cbar = ax.figure.colorbar(im, shrink=0.6, pad=0.1, ax=ax, location="bottom",
    #                           **cbar_kw)
    # cbar.ax.set_ylabel(cbarlabel, rotation=0, va="center", labelpad=25)
    # # cbar.ax.tick_params(length=0.5)

    # # Show all ticks and label them with the respective list entries.
    # ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    # ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # # Let the horizontal axes labeling appear on top.
    # ax.tick_params(top=True, bottom=False,
    #                labeltop=True, labelbottom=False)

    # # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), ha="center",
    #          rotation_mode="anchor")

    # # Turn spines off and create white grid.
    # ax.spines[:].set_visible(False)

    # ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    # ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    # ax.tick_params(which="minor", bottom=False, left=False)

    # return im, cbar

    # rescale the data
    data = 100*data

    # Calculate mean for centering the colormap
    vmin, vmax = np.min(data), np.max(data)
    vcenter = np.mean(data)

    # Create a custom normalization centered around the mean
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    # Plot the heatmap
    im = ax.imshow(data, norm=norm, cmap='YlGn', **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im,
                              shrink=0.6, pad=0.1, ax=ax,
                              location="bottom", **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=0, va="center", labelpad=25)

    # Generate seven ticks from [vmin, vmax]
    ticks = np.linspace(vmin, vmax, 8)
    # print(ticks)
    
    # Set colorbar ticks
    cbar.set_ticks(ticks)
    
    # Format tick labels
    tick_labels = [f'{tick:.2f}' for tick in ticks]
    
    # Highlight vmin and vmax
    # min_index = np.argmin(ticks)  # np.abs(ticks - vmin))
    # max_index = np.argmin(ticks)  # np.abs(ticks - vmax))
    # tick_labels[min_index] = f'{vmin:.3f}'
    # tick_labels[max_index] = f'{vmax:.3f}'
    
    cbar.set_ticklabels(tick_labels)
    
    # # Format colorbar ticks to show more decimal places
    # cbar.formatter.set_powerlimits((0, 0))
    # cbar.update_ticks()

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="center", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Find the cell with the highest value
    max_idx = np.unravel_index(np.argmax(data), data.shape)
    
    # Add a red dashed rectangle around the cell with the highest value
    # rect = Rectangle((max_idx[1] - 0.5, max_idx[0] - 0.5), 1, 1, fill=False, 
    #                  edgecolor='red', linestyle='--', linewidth=2)
    # ax.add_patch(rect)

    # Add a bright blue dashed rectangle around the cell with the highest value
    rect = Rectangle((max_idx[1] - 0.5, max_idx[0] - 0.5), 1, 1, fill=False, 
                     edgecolor='#00FFFF', linestyle='--', linewidth=3)
    ax.add_patch(rect)

    # Add a star marker to the cell with the highest value
    ax.plot(max_idx[1], max_idx[0], marker='*', markersize=20, 
            markeredgecolor='black', markerfacecolor='#00FFFF')

    # Add text annotations
    # texts = annotate_heatmap(im, data, valfmt="{x:.4f}")

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center,
    # but allow it to be overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)  
    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
