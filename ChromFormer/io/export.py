import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import imageio


def make_gif(path_in: str, path_out: str):
    """Creates a gif

    Args:
        path_in: path to find images to use for gif making
        path_out: path to write the gif to

    """
    png_dir = f"{path_in}images/"
    images = []
    for file_name in sorted(os.listdir(png_dir)):
        if file_name.endswith(".png"):
            file_path = os.path.join(png_dir, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave(path_out, images, duration=0.2, loop=1)

def save_structure(
    model,
    epoch,
    trussart_structures,
    trussart_hic,
    nb_bins: int,
    batch_size: int,
    embedding_size: int,
    other_params: bool = False,
):
    """Function that saves structures over epochs to be used later for gifs making

    Args:
        model: model
        epoch: current model epoch
        trussart_structures: structure to be aligned with when present
        trussart_hic: hic matrix used by the model to predict the structure
        nb_bins: number of loci
        batch_size: size of the batch
        embedding_size: 3D dimension
        other_params: set whether alignment of structures needs to be applied

    """
    trussart_true_structure = np.mean(trussart_structures, axis=0)

    # Trussart predicted structure
    torch_trussart_hic = torch.FloatTensor(trussart_hic)
    torch_trussart_hic = torch.reshape(torch_trussart_hic, (1, nb_bins, nb_bins))
    torch_trussart_hic = torch.repeat_interleave(torch_trussart_hic, batch_size, 0)
    if other_params:
        trussart_pred_structure, _, _ = model(torch_trussart_hic)
    else:
        trussart_pred_structure, _ = model(torch_trussart_hic)
    trussart_pred_structure = trussart_pred_structure.detach().numpy()[0]

    # Superpose structure using Kabsch algorithm
    (
        trussart_pred_structure_superposed,
        trussart_true_structure_superposed,
    ) = kabsch_superimposition_numpy(
        trussart_pred_structure, trussart_true_structure, embedding_size
    )

    # Plot and compare the two structures
    x_pred = trussart_pred_structure_superposed[:, 0]
    y_pred = trussart_pred_structure_superposed[:, 1]
    z_pred = trussart_pred_structure_superposed[:, 2]

    x_true = trussart_true_structure_superposed[:, 0]
    y_true = trussart_true_structure_superposed[:, 1]
    z_true = trussart_true_structure_superposed[:, 2]

    # Initialize figure with 4 3D subplots
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scatter3d"}]])

    fig.add_trace(
        go.Scatter3d(
            x=x_pred,
            y=y_pred,
            z=z_pred,
            marker=dict(
                size=4,
                color=np.asarray(range(len(x_pred))),
                colorscale="Viridis",
            ),
            line=dict(color="darkblue", width=2),
        ),
        row=1,
        col=1,
    )

    fig.update_layout(height=1000, width=1000)
    if not os.path.exists("images"):
        os.makedirs("images")
    fig.write_image(file="images/structure{:03d}.png".format(epoch), format="png")
    # plt.close(fig)

def save_structure_fission_yeast(
    model, epoch, trussart_hic, nb_bins, batch_size, embedding_size, other_params=False
):
    """Function that saves structures over epochs to be used later for gifs making

    Args:
        model: model
        epoch: current model epoch
        trussart_structures: structure to be aligned with when present
        trussart_hic: hic matrix used by the model to predict the structure
        nb_bins: number of loci
        batch_size: size of the batch
        embedding_size: 3D dimension
        other_params: set whether alignment of structures needs to be applied

    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Trussart predicted structure
    torch_trussart_hic = torch.FloatTensor(trussart_hic)
    torch_trussart_hic = torch.reshape(torch_trussart_hic, (1, nb_bins, nb_bins))
    torch_trussart_hic = torch.repeat_interleave(torch_trussart_hic, batch_size, 0).to(
        device
    )
    if other_params:
        trussart_pred_structure, _, _ = model(torch_trussart_hic)
    else:
        trussart_pred_structure, _ = model(torch_trussart_hic)
    trussart_pred_structure = trussart_pred_structure.cpu().detach().numpy()[0]

    # Plot and compare the two structures
    x_pred = trussart_pred_structure[:, 0]
    y_pred = trussart_pred_structure[:, 1]
    z_pred = trussart_pred_structure[:, 2]

    # Initialize figure with 4 3D subplots
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scatter3d"}]])

    fig.add_trace(
        go.Scatter3d(
            x=x_pred,
            y=y_pred,
            z=z_pred,
            marker=dict(
                size=4,
                color=np.asarray(range(len(x_pred))),
                colorscale="Viridis",
            ),
            line=dict(color="darkblue", width=2),
        ),
        row=1,
        col=1,
    )

    fig.update_layout(height=1000, width=1000)
    if not os.path.exists("images"):
        os.makedirs("images")
    fig.write_image(file="images/structure{:03d}.png".format(epoch), format="png")
    # plt.close(fig)