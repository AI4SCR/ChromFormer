"""This script has the necessary functions related to computations of data such as import, transformation and calling the correct normalisation techniques"""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2022
ALL RIGHTS RESERVED
"""

import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from scipy.spatial import distance_matrix
from .Normalisation import ice, centralize_and_normalize_numpy
from .Optimal_Transport import ot_data, transport
from typing import Tuple
from plotly.subplots import make_subplots
import torch
import imageio
from ..Model.lddt_tools import get_confidence_metrics
from sklearn.metrics import mean_squared_error
import pandas as pd
from scipy.spatial import distance


def import_trussart_data(path) -> Tuple[np.ndarray, np.ndarray]:
    """Accesses the stored data to return the Trussart structures and hic

    Args:
        path: constant path to access Data folder

    Returns:
        trussart_hic: numpy array of the trussart interaction matrices
        trussart_structures: numpy array of the the trussart structures
    """
    trussart_hic_path = f"{path}/trussart/hic_matrices/150_TADlike_alpha_150_set0.mat"
    trussart_structure_path = f"{path}/trussart/structure_matrices/"
    trussart_hic = np.loadtxt(trussart_hic_path, dtype="f", delimiter="\t")
    scaler = MinMaxScaler()
    trussart_hic = scaler.fit_transform(trussart_hic)
    trussart_structures = []

    file_list = os.listdir(trussart_structure_path)
    file_list = filter(lambda f: f.endswith(".xyz"), file_list)

    for file_name in file_list:
        current_trussart_structure = np.loadtxt(
            trussart_structure_path + file_name, dtype="f", delimiter="\t"
        )
        current_trussart_structure = current_trussart_structure[:, 1:]
        trussart_structures.append(current_trussart_structure)

    return trussart_hic, trussart_structures


def compute_hic_matrix(distance_matrix: np.ndarray, alpha: int) -> np.ndarray:
    """Computes the distance to interaction calculation

    In order to the compute the interaction from the distance, this function computes 1/root alpha of the distance matrix

    Args:
        distance_matrix: array of the distance matric
        alpha: integer to know which root alpha power to use

    Returns:
        hic_matrix: array of the interaction matric
    """

    distance_matrix = np.where(distance_matrix == 0, np.inf, distance_matrix)

    hic_matrix = np.zeros((len(distance_matrix), len(distance_matrix)))
    hic_matrix = np.where(
        distance_matrix == np.inf, hic_matrix, np.power(distance_matrix, -1 / alpha)
    )

    return hic_matrix


def create_sphere_coordinates(x_0=0, y_0=0, z_0=0, radius=1):
    """Generate the sphere coordinates for the plotting

    Args:
        x: integer
        y: integer
        z: integer

    Return:
        x: x integer coordinate
        y: y integer coordinate
        z: z integer coordinate
    """
    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi, 100)

    x = radius * np.outer(np.cos(theta), np.sin(phi)) + x_0
    y = radius * np.outer(np.sin(theta), np.sin(phi)) + y_0
    z = radius * np.outer(np.ones(100), np.cos(phi)) + z_0

    return x, y, z


def create_sphere_surface(x_0=0, y_0=0, z_0=0, radius=1):
    """Creates a sphere surface for the plotting"""
    x, y, z = create_sphere_coordinates(x_0, y_0, z_0, radius)
    return go.Surface(x=x, y=y, z=z, opacity=0.1)


def generate_hic(
    rng,
    synthetic_biological_structure: np.ndarray,
    trussart_hic: np.ndarray,
    use_ice: bool = True,
    use_minmax: bool = False,
    use_ot: bool = True,
    use_softmax: bool = False,
    seed: int = 42,
    plot_optimal_transport: bool = False,
    exponent: int = 1,
):
    """This function is used to generate hic from a given structure matrix that depends on parameters

    Args:
        rng: seed random state to use and that is passed accross functions
        synthetic_biological_structure: array of the structure to use to generate an interaction matrix
        trussart_hic: array of the trussart hic to use for optimal transport
        use_ice: boolean that declares if ice normalisation should be used
        use_minmax: boolean that declares if minmax scaling should be used
        use_ot: boolean that declares if optimal transport should be used
        use_softmax: boolean that declares if softmax should be used instead of optimal transport usually
        seed: seed integer
        plot_optimal_transport: boolean that says if we should plot the histograms of optimal transport
        exponent: the exponent that will be used to compute from distance to interaction

    Returns:
        hic_matrix: array of the generated hic matrix
    """

    scaler = MinMaxScaler()
    precomputed_distances = distance_matrix(
        synthetic_biological_structure, synthetic_biological_structure
    )
    hic_matrix = compute_hic_matrix(precomputed_distances, exponent)

    if use_ice:
        hic_matrix = ice(hic_matrix)

    if use_minmax:
        hic_matrix = scaler.fit_transform(hic_matrix)

    if use_ot:
        orig_hic = hic_matrix
        # rng = np.random.RandomState(seed)
        xs, xt, x1, x2 = ot_data(hic_matrix, trussart_hic, rng)
        hic_matrix, i2te = transport(
            xs, xt, x1, x2, hic_matrix.shape, trussart_hic.shape
        )

    if use_softmax:
        hic_matrix = 1 / (1 + np.exp(-8 * hic_matrix + 4))

    if plot_optimal_transport:
        return hic_matrix, orig_hic, xs, xt
    return hic_matrix

def kabsch_superimposition_numpy(pred_structure: np.ndarray, true_structure: np.ndarray, embedding_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Performs an alignment of the structure
    
    Args:
        pred_structure: array of the predicted structure
        true structure: array representing the true structure
        embedding_size: integer representing the dimension
    
    Returns:
        pred_structure_unit_ball: predicted structure once aligned with the true structure
        true_structure_unit_ball: true structure once aligned with the pred structure
    """
    # Centralize and normalize to unit ball
    pred_structure_unit_ball = centralize_and_normalize_numpy(pred_structure)
    true_structure_unit_ball = centralize_and_normalize_numpy(true_structure)
    
    # Rotation (solution for the constrained orthogonal Procrustes problem, subject to det(R) = 1)
    m = np.matmul(np.transpose(true_structure_unit_ball), pred_structure_unit_ball)
    u, s, vh = np.linalg.svd(m)
    
    d = np.sign(np.linalg.det(np.matmul(u, vh)))
    a = np.eye(embedding_size)
    a[-1,-1] = d
    
    r = np.matmul(np.matmul(u, a), vh)
    
    pred_structure_unit_ball = np.transpose(np.matmul(r, np.transpose(pred_structure_unit_ball)))
    
    return pred_structure_unit_ball, true_structure_unit_ball

def kabsch_distance_numpy(pred_structure: np.ndarray, true_structure: np.ndarray, embedding_size: int) -> int:
    """Performs an alignment of the structure and a score of the mean scared error of point position
    
    Args:
        pred_structure: array of the predicted structure
        true structure: array representing the true structure
        embedding_size: integer representing the dimension
    
    Returns:
        d: integer representing the score
    """
    pred_structure_unit_ball, true_structure_unit_ball = \
            kabsch_superimposition_numpy(pred_structure, true_structure, embedding_size)
    
    # Structure comparison
    d = np.mean(np.sum(np.square(pred_structure_unit_ball - true_structure_unit_ball), axis=1))
    
    return d

def save_structure(model, epoch, trussart_structures, trussart_hic, nb_bins, batch_size, embedding_size, other_params=False):
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
    trussart_pred_structure_superposed, trussart_true_structure_superposed = \
            kabsch_superimposition_numpy(trussart_pred_structure, trussart_true_structure, embedding_size)

    # Plot and compare the two structures
    x_pred = trussart_pred_structure_superposed[:, 0]  
    y_pred = trussart_pred_structure_superposed[:, 1]
    z_pred = trussart_pred_structure_superposed[:, 2]

    x_true = trussart_true_structure_superposed[:, 0]  
    y_true = trussart_true_structure_superposed[:, 1]
    z_true = trussart_true_structure_superposed[:, 2]

    # Initialize figure with 4 3D subplots
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{'type': 'scatter3d'}]])


    fig.add_trace(
        go.Scatter3d(
        x=x_pred, y=y_pred, z=z_pred,
        marker=dict(
            size=4,
            color=np.asarray(range(len(x_pred))),
            colorscale='Viridis',
        ),
        line=dict(
            color='darkblue',
            width=2
        )
    ),row=1, col=1)

    fig.update_layout(
        height=1000,
        width=1000
    )
    if not os.path.exists('images'):
        os.makedirs('images')
    fig.write_image(file='images/structure{:03d}.png'.format(epoch), format='png')
    #plt.close(fig) 


def make_gif(path_in, path_out):

    png_dir = f"{path_in}images/"
    images = []
    for file_name in sorted(os.listdir(png_dir)):
        if file_name.endswith('.png'):
            file_path = os.path.join(png_dir, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave(path_out, images, duration=0.2, loop=1)

def scale_logits(trussart_pred_logits, scaled_model, batch_size, nb_bins):

    pred_logits = trussart_pred_logits.detach()
    pred_logits = torch.reshape(pred_logits, (batch_size*nb_bins, 100))
    logits_trussart = scaled_model.temperature_scale(pred_logits)
    scaled_trussart_logits = torch.reshape(logits_trussart, (batch_size,nb_bins, 100))
    confidence_metric_scaled, plddt_scaled = get_confidence_metrics(scaled_trussart_logits.detach().numpy()[0])
    return confidence_metric_scaled, plddt_scaled

def mse_unscaled_scaled(value, plldts, plldt_scaled):
    true = value.numpy()[0]*100
    pred = plldts
    pred_scaled = plldt_scaled
    mse_unscaled = mean_squared_error(true, pred)
    mse_scaled = mean_squared_error(true, pred_scaled)
    return mse_unscaled, mse_scaled

def import_fission_yeast(path):
    
    FISSION_YEAST_HIC_PATH = f"{path}/fission_yeast/hic_matrices/GSM2446256_HiC_20min_10kb_dense.matrix"
    fission_yeast_hic = np.loadtxt(FISSION_YEAST_HIC_PATH, dtype='f', delimiter=' ')
    scaler = MinMaxScaler()
    fission_yeast_hic = scaler.fit_transform(fission_yeast_hic)
    return fission_yeast_hic

def FISH_values_Tanizawa(filename):
    fish_table = pd.read_csv(filename, sep=";", header=0, names=None, dtype=float)
    temp = fish_table[['Chr1', 'StartChr1', 'Chr2', 'EndChr2', 'FISH_dist']]
    fish_distances = pd.DataFrame()
    fish_distances['Chr1'] = temp['Chr1'].astype(int)
    fish_distances['StartChr1'] = round(temp['StartChr1'] / 10000).astype(int)
    fish_distances['Chr2'] = temp['Chr2'].astype(int)
    fish_distances['EndChr2'] = round(temp['EndChr2'] / 10000).astype(int)
    fish_distances['FISH_dist'] = temp['FISH_dist']
    fish_distances.columns = ['Chr1', 'Loci1', 'Chr2', 'Loci2', 'FISH_dist']

    return fish_distances

def dist_Tanizawa_FISH(coordinates, fish_table):
    startChr=[0, 558, 1012]  
    
    ratio = 1
    reconstr_dist = []
    for i in fish_table.index.values.tolist():
        chr1 = fish_table['Chr1'][i]
        chr2 = fish_table['Chr2'][i]
        loci1 = fish_table['Loci1'][i]
        loci2 = fish_table['Loci2'][i]
        dist_fish = fish_table['FISH_dist'][i]
        dist = distance.euclidean(coordinates[startChr[chr1-1] + loci1], coordinates[startChr[chr2-1] + loci2])
        reconstr_dist.append(dist)
        
    return reconstr_dist

def save_structure_fission_yeast(model, epoch, trussart_hic, nb_bins, batch_size, embedding_size, other_params=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Trussart predicted structure
    torch_trussart_hic = torch.FloatTensor(trussart_hic)
    torch_trussart_hic = torch.reshape(torch_trussart_hic, (1, nb_bins, nb_bins))
    torch_trussart_hic = torch.repeat_interleave(torch_trussart_hic, batch_size, 0).to(device)
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
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{'type': 'scatter3d'}]])


    fig.add_trace(
        go.Scatter3d(
        x=x_pred, y=y_pred, z=z_pred,
        marker=dict(
            size=4,
            color=np.asarray(range(len(x_pred))),
            colorscale='Viridis',
        ),
        line=dict(
            color='darkblue',
            width=2
        )
    ),row=1, col=1)

    fig.update_layout(
        height=1000,
        width=1000
    )
    if not os.path.exists('images'):
        os.makedirs('images')
    fig.write_image(file='images/structure{:03d}.png'.format(epoch), format='png')
    #plt.close(fig) 
