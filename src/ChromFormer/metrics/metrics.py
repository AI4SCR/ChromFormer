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
from scipy.spatial import distance
from typing import Tuple
import torch
from sklearn.metrics import mean_squared_error
import pandas as pd

from ..processing.normalisation import centralize_and_normalize_numpy
from ..models.lddt_tools import get_confidence_metrics




def kabsch_superimposition_numpy(pred_structure: np.ndarray,
                                 true_structure: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Performs an alignment of the structure

    Args:
        pred_structure: array of the predicted structure
        true structure: array representing the true structure

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
    a = np.eye(pred_structure.shape[1])
    a[-1, -1] = d

    r = np.matmul(np.matmul(u, a), vh)

    pred_structure_unit_ball = np.transpose(
        np.matmul(r, np.transpose(pred_structure_unit_ball))
    )

    return pred_structure_unit_ball, true_structure_unit_ball


def kabsch_distance_numpy(pred_structure: np.ndarray, true_structure: np.ndarray) -> int:
    """Performs an alignment of the structure and a score of the mean scared error of point position

    Args:
        pred_structure: array of the predicted structure
        true structure: array representing the true structure

    Returns:
        d: integer representing the score
    """
    pred_structure_unit_ball, true_structure_unit_ball = kabsch_superimposition_numpy(pred_structure, true_structure)

    # Structure comparison
    d = np.mean(np.sum(np.square(pred_structure_unit_ball - true_structure_unit_ball), axis=1))

    return d


def scale_logits(trussart_pred_logits, scaled_model, nb_bins: int) -> Tuple[int, np.ndarray]:
    batch_size = trussart_pred_logits.shape[0]
    pred_logits = trussart_pred_logits.detach()
    pred_logits = torch.reshape(pred_logits, (batch_size * nb_bins, 100))
    logits_trussart = scaled_model.temperature_scale(pred_logits)
    scaled_trussart_logits = torch.reshape(logits_trussart, (batch_size, nb_bins, 100))
    confidence_metric_scaled, plddt_scaled = get_confidence_metrics(
        scaled_trussart_logits.detach().numpy()[0]
    )
    return confidence_metric_scaled, plddt_scaled


def mse_unscaled_scaled(value, plldts, plldt_scaled):
    true = value.numpy()[0] * 100
    pred = plldts
    pred_scaled = plldt_scaled
    mse_unscaled = mean_squared_error(true, pred)
    mse_scaled = mean_squared_error(true, pred_scaled)
    return mse_unscaled, mse_scaled


def FISH_values_Tanizawa(filename):
    fish_table = pd.read_csv(filename, sep=";", header=0, names=None, dtype=float)
    temp = fish_table[["Chr1", "StartChr1", "Chr2", "EndChr2", "FISH_dist"]]
    fish_distances = pd.DataFrame()
    fish_distances["Chr1"] = temp["Chr1"].astype(int)
    fish_distances["StartChr1"] = round(temp["StartChr1"] / 10000).astype(int)
    fish_distances["Chr2"] = temp["Chr2"].astype(int)
    fish_distances["EndChr2"] = round(temp["EndChr2"] / 10000).astype(int)
    fish_distances["FISH_dist"] = temp["FISH_dist"]
    fish_distances.columns = ["Chr1", "Loci1", "Chr2", "Loci2", "FISH_dist"]

    return fish_distances


def dist_Tanizawa_FISH(coordinates, fish_table):
    startChr = [0, 558, 1012]

    ratio = 1
    reconstr_dist = []
    for i in fish_table.index.values.tolist():
        chr1 = fish_table["Chr1"][i]
        chr2 = fish_table["Chr2"][i]
        loci1 = fish_table["Loci1"][i]
        loci2 = fish_table["Loci2"][i]
        dist_fish = fish_table["FISH_dist"][i]
        dist = distance.euclidean(
            coordinates[startChr[chr1 - 1] + loci1],
            coordinates[startChr[chr2 - 1] + loci2],
        )
        reconstr_dist.append(dist)

    return reconstr_dist
