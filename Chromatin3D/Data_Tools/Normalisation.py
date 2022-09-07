import numpy as np
from numpy import linalg as LAnumpy


def ICE(hic_matrix: np.ndarray) -> np.ndarray:

    weighted_ice = np.sum(hic_matrix) / (
        np.sum(hic_matrix, axis=0) * np.sum(hic_matrix, axis=1)
    )

    mu_ice = np.log10(
        np.sum(np.multiply(hic_matrix, weighted_ice)) / (hic_matrix.shape[0] ** 2)
    )

    sigma_ice = np.sqrt(
        np.sum(np.power(np.multiply(hic_matrix, weighted_ice) - mu_ice, 2))
        / hic_matrix.shape[0]
    )

    mult = np.multiply(hic_matrix, weighted_ice)
    result = np.where(mult > 0.0000000001, mult, -np.inf)
    z_score = (np.log10(result, out=result, where=result > 0) - mu_ice) / sigma_ice

    z_score = np.where(z_score == -np.inf, 0, z_score)

    return z_score


def normalize_numpy(z: np.ndarray) -> np.ndarray:

    norm = LAnumpy.norm(z, 2, axis=1)
    max_norm = np.max(norm, axis=0)
    if max_norm == 0:
        max_norm = 1

    return z / max_norm


def centralize_numpy(z: np.ndarray) -> np.ndarray:
    return z - np.mean(z, axis=0)


def centralize_and_normalize_numpy(z: np.ndarray) -> np.ndarray:

    # Translate
    z = centralize_numpy(z)

    # Scale
    z = normalize_numpy(z)

    return z
