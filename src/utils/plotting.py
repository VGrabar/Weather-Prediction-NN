from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import random


def make_confusion_matrix(
    matrix,
    labels=None,
    title="Confusion Matrix",
    annotate=True,
    colormap="Blues",
    colorbar=True,
    colorbar_orientation="vertical",
    normalize=True,
    size=(8, 6),
    interpolation="nearest",
    xlabel=None,
    xlabel_rotate=45,
    ylabel=None,
    ylabel_rotate=0,
    data_format="0.2f",
    filename="confusion_matrix.png",
):
    """
    Create a confusion matrix image.
    Arguments:
        matrix: an N x N confusion matrix given as either counts, or accuracies
        labels: the labels for each category. Must match the order of matrix
        title: title of figure, or None
        annotate: if True, then display the matrix value in cell
        colormap: None, or a valid matplotlib colormap name
        colorbar: if True, display a colorbar
        colorbar_orientation: 'vertical' or 'horizontal'
        normalize: if True, then matrix values are the taken as counts;
            otherwise, matrix values are taken as raw values
        size: size of the figure, in inches
        interpolation: method for mapping matrix values to color. Valid values
            are: None, 'nearest', 'none', 'bilinear', 'bicubic', 'spline16',
           'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
           'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', and 'lanczos'
        xlabel: title for the y-axis
        xlabel_rotate: degrees to rotate the x-labels
        ylabel: title for the y-axis
        ylabel_rotate: degrees to rotate the y-labels
        data_format: the Python format, e.g. '0.4f' or None
        filename: name of filename to save image to, or None. Should end in
            'png', 'jpg', etc. if given. If given, can begin with "~" meaning
            HOME.
    """

    matrix = np.array(matrix)
    fig = Figure(figsize=size, frameon=False)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    im = ax.imshow(matrix, interpolation=interpolation, cmap=colormap)
    if colorbar:
        layout = make_axes_locatable(ax)
        if colorbar_orientation == "vertical":
            cax = layout.append_axes("right", size="5%", pad=0.05)
        else:
            cax = layout.append_axes("top", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation=colorbar_orientation)
    if labels is not None:
        tick_marks = np.arange(len(labels))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(labels, rotation=xlabel_rotate)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(labels, rotation=ylabel_rotate)
    if normalize:
        matrix = matrix.astype("float") / matrix.sum(axis=1)[:, np.newaxis]
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if annotate:
        thresh = matrix.mean()
        for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
            if data_format:
                str_format = "{:" + data_format + "}"
            else:
                str_format = "{:,}"
            ax.text(
                j,
                i,
                str_format.format(matrix[i, j]),
                horizontalalignment="center",
                color="white" if matrix[i, j] > thresh else "black",
            )
    fig.tight_layout()
    filename = os.path.expanduser(filename)
    fig.savefig(filename)
    return fig


def make_pred_vs_target_plot(
    preds,
    targets,
    title="Comparison",
    size=(8, 6),
    xlabel=None,
    xlabel_rotate=45,
    ylabel=None,
    ylabel_rotate=0,
    filename="forecasts.png",
):

    fig = Figure(figsize=size, frameon=False)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    x_length = targets.shape[2]
    y_length = targets.shape[3]
    x_random = random.choice(list(range(x_length)))
    y_random = random.choice(list(range(y_length)))
    targets = targets[:, 0, x_random, y_random].cpu()
    preds = preds[:, 0, x_random, y_random].cpu()
    time_periods = np.arange(0, targets.shape[0])
    ax.plot(time_periods, targets, "g-", label="actual")
    ax.plot(time_periods, preds, "g-", label="predictions")
    ax.legend()

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    fig.tight_layout()
    filename = os.path.expanduser(filename)
    fig.savefig(filename)
    return fig
