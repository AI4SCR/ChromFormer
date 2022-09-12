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
from .Normalisation import ice
from .Optimal_Transport import ot_data, transport
from typing import Tuple


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
