"""This script has the necessary functions related to plotting"""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2022
ALL RIGHTS RESERVED
"""

import numpy as np
import matplotlib.pyplot as plt
from .Data_Calculation import create_sphere_surface
import plotly.graph_objects as go
import matplotlib
from plotly.subplots import make_subplots
from .Data_Calculation import kabsch_distance_numpy


def plot_structure_in_sphere(synthetic_biological_structure: np.ndarray) -> None:
    """Plots a structure in a spherical surface

    Args:
        synthetic_biological_structure: array of the structure to plot in a spherical surface
    """
    unit_sphere_surface = create_sphere_surface()
    synthetic_biological_structure_scatter = go.Scatter3d(
        x=synthetic_biological_structure[:, 0],
        y=synthetic_biological_structure[:, 1],
        z=synthetic_biological_structure[:, 2],
        marker=dict(
            size=4,
            color=np.asarray(range(len(synthetic_biological_structure[:, 0]))),
            colorscale="Viridis",
        ),
        line=dict(color="darkblue", width=2),
    )
    layout = go.Layout(
        width=900,
        height=900,
    )
    fig = go.Figure(
        data=[unit_sphere_surface, synthetic_biological_structure_scatter],
        layout=layout,
    )

    fig.show()


def plot_hic(hic: np.ndarray) -> None:
    """Function to plot HIC matrices

    Args:
        hic: array like hic matrix to plot
    """
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))

    axs.imshow(hic, cmap="hot", interpolation="nearest")
    axs.tick_params(axis="both", which="major", labelsize=30, width=4)

    # plt.savefig('synthetic_biological_hic_example.png')
    plt.show()


def plot_optimal_transport(xs: np.ndarray, xt: np.ndarray, i1te: np.ndarray) -> None:
    """Function to plot the source, target and transport histogram distribution from the optimal transport being done.

    Args:
        Xs: array of source histogram distribution
        Xt: array of target histogram distribution
        I1te: array of the transported source histogram distribution
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    counts, bins = np.histogram(xs, bins=30)  # source
    axs[0].hist(bins[:-1], bins, weights=counts)
    axs[0].set_title("Source Histogram", fontstyle="italic")
    counts, bins = np.histogram(xt, bins=30)  # target
    axs[1].hist(bins[:-1], bins, weights=counts)
    axs[1].set_title("Target Histogram", fontstyle="italic")
    counts, bins = np.histogram(i1te, bins=30)  # transported source
    axs[2].hist(bins[:-1], bins, weights=counts)
    axs[2].set_title("Transported source Histogram", fontstyle="italic")


def plot_grad_flow(named_parameters):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            print(p.grad.abs().mean())
            print(n)
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c", log=True)
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b", log=True)
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    # bottome -0.0001
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend(
        [
            matplotlib.lines.Line2D([0], [0], color="c", lw=4),
            matplotlib.lines.Line2D([0], [0], color="b", lw=4),
            matplotlib.lines.Line2D([0], [0], color="k", lw=4),
        ],
        ["max-gradient", "mean-gradient", "zero-gradient"],
    )


def plot_losses(
    losses,
    train_biological_losses_all_epochs,
    test_biological_losses_all_epochs,
    train_kabsch_losses_all_epochs,
    test_kabsch_losses_all_epochs,
    trussart_test_kabsch_losses_all_epochs,
    train_distance_losses_all_epochs,
    test_distance_losses_all_epochs,
):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].plot(losses, label="Losses")
    axs[0, 0].legend()

    axs[0, 1].plot(train_biological_losses_all_epochs, label="Train Bio")
    axs[0, 1].plot(test_biological_losses_all_epochs, label="Test Bio")
    axs[0, 1].legend()

    axs[1, 0].plot(train_kabsch_losses_all_epochs, label="Train Kabsch")
    axs[1, 0].plot(test_kabsch_losses_all_epochs, label="Test Kabsch")
    axs[1, 0].plot(
        trussart_test_kabsch_losses_all_epochs, label="Kabsch Distance Trussart"
    )
    axs[1, 0].legend()

    axs[1, 1].plot(train_distance_losses_all_epochs, label="Train Dist")
    axs[1, 1].plot(test_distance_losses_all_epochs, label="Test Dist")
    axs[1, 1].plot(
        trussart_test_kabsch_losses_all_epochs, label="Kabsch Distance Trussart"
    )
    axs[1, 1].legend()


def plot_test_distance_matrix(ground_truth_matrix, reconstruction_matrix):
    fig, axes = plt.subplots(1, 2, figsize=(15, 15))

    axes[0].imshow(ground_truth_matrix, cmap="hot", interpolation="nearest")

    axes[1].imshow(reconstruction_matrix, cmap="hot", interpolation="nearest")

    plt.show()


def plot_true_pred_structures(
    x_pred,
    y_pred,
    z_pred,
    x_true,
    y_true,
    z_true,
    colorscale1,
    colorscale2,
    color1,
    color2,
):

    fig = plt.figure(figsize=(500, 500))

    # Initialize figure with 4 3D subplots
    fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]]
    )

    # adding surfaces to subplots.
    fig.add_trace(
        go.Scatter3d(
            x=x_true,
            y=y_true,
            z=z_true,
            marker=dict(
                size=4,
                color=colorscale1,
                colorscale=color1,
            ),
            line=dict(color="darkblue", width=2),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter3d(
            x=x_pred,
            y=y_pred,
            z=z_pred,
            marker=dict(
                size=4,
                color=colorscale2,
                colorscale=color2,
            ),
            line=dict(color="darkblue", width=2),
        ),
        row=1,
        col=2,
    )

    fig.update_layout(height=1000, width=1300)
    # fig.write_image(file='bla.png', format='.png')

    fig.show()


def plot_hist_kabsch_distances(
    test_size, test_true_structures, test_pred_structures, embedding_size
):
    kabsch_distances = []

    for graph_index in range(test_size):

        test_true_structure = test_true_structures[graph_index, :, :]
        test_pred_structure = test_pred_structures[graph_index, :, :]

        d = kabsch_distance_numpy(
            test_pred_structure, test_true_structure, embedding_size
        )
        kabsch_distances.append(d)

    n, bins, patches = plt.hist(kabsch_distances, 100, facecolor="blue", alpha=0.5)
    plt.show()

    print("mean: " + str(np.mean(kabsch_distances)))
    print("median: " + str(np.median(kabsch_distances)))
    print("variance: " + str(np.var(kabsch_distances)))


def plot_pred_conf(trussart_pred_structure_superposed, pldts, color):
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scatter3d"}]])

    # adding surfaces to subplots.
    fig.add_trace(
        go.Scatter3d(
            x=trussart_pred_structure_superposed[:, 0],
            y=trussart_pred_structure_superposed[:, 1],
            z=trussart_pred_structure_superposed[:, 2],
            opacity=0.7,
            marker=dict(size=6, color=pldts, colorscale=color, line=dict(width=3)),
            line=dict(color="darkblue", width=2),
        ),
        row=1,
        col=1,
    )

    fig.update_layout(height=1000, width=1000)
    fig.write_image(file="caib_plot.png", format="png")

    fig.show()
