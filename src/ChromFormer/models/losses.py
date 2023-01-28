"""This script calculates the losses used by the model"""

import torch
import numpy as np
from ..metrics.metrics import kabsch_distance_numpy


def compute_trussart_test_kabsch_loss(
        trussart_hic,
        trussart_structures,
        model,
        nb_bins,
        batch_size,
        embedding_size,
        other_params=False,
):
    """Computes the kabsch distance between the trussart data and the synthetic data

    Args:
        trussart_hic: true trussart contact matrix
        trussart_structures: true trussart structures
        model: learnt model
        nb_bins: number of loci
        batch_size: size of the batch
        embedding_size: 3D dimension
        other_params: boolean to know if the model outputs more than 2 parameters

    Returns:
        The kabsch distance score
    """
    torch_trussart_hic = torch.FloatTensor(trussart_hic)
    torch_trussart_hic = torch.reshape(torch_trussart_hic, (1, nb_bins, nb_bins))
    torch_trussart_hic = torch.repeat_interleave(torch_trussart_hic, batch_size, 0)
    if other_params:
        trussart_pred_structure, _, _ = model(torch_trussart_hic)
    else:
        trussart_pred_structure, _ = model(torch_trussart_hic)
    trussart_pred_structure = trussart_pred_structure.detach().numpy()[0]

    kabsch_distances = []
    for true_structure in trussart_structures:
        kabsch_distances.append(
            kabsch_distance_numpy(
                trussart_pred_structure, true_structure, embedding_size
            )
        )

    return np.mean(kabsch_distances)


def biological_loss_fct(
        pred_structure, true_structure, pred_distance, true_distance, nb_bins, batch_size
):
    """Computes the biofeasibility loss. This is done by penalising non smooth angles between consecutive trajectories and the variance of distances between consecutive loci

    Args:
        pred_structure: predicted 3D structure
        true_structure: ground truth structure
        pred_distance: predicted distance matrix
        true_distance: ground truth distance matrix
        nb_bins: number of loci
        batch_size: size of the batch

    Returns:
        A biological score
    """
    ####### Pairwise distances loss ########

    # NOTE: removed pred_distance.reshape((batch_size, nb_bins, nb_bins))
    between_bin_distance = torch.diagonal(pred_distance, offset=1, dim1=1, dim2=2)
    between_bin_distance_loss = torch.var(between_bin_distance)

    ######### Consecutive angles loss ##########

    pred_structure_vectors = (pred_structure - torch.roll(pred_structure, 1, dims=1))[:, 1:, :]
    pred_structure_dot_products = torch.diagonal(
        torch.matmul(pred_structure_vectors,
                     torch.transpose(pred_structure_vectors, dim0=1, dim1=2)), offset=-1, dim1=2)

    # NOTE: changed
    #   torch.ones((10,... -> torch.ones((batch_size,...
    pairwise_angles_loss = torch.where(
        pred_structure_dot_products < 0,
        torch.ones((batch_size, nb_bins - 2)) * 0.1,
        torch.zeros((batch_size, nb_bins - 2)),
    )

    pairwise_angles_loss = torch.mean(pairwise_angles_loss)

    return between_bin_distance_loss + pairwise_angles_loss


def kabsch_loss_fct(pred_structure, true_structure, embedding_size, batch_size):
    """Aligns structures and finds their mean square error

    Args:
        pred_structure: predicted 3D structure
        true_structure: ground truth structure
        embedding_size: 3D dimension
        batch_size: size of the batch

    Returns:
        trussart_hic: numpy array of the trussart interaction matrices
        trussart_structures: numpy array of the the trussart structures
    """
    m = torch.matmul(torch.transpose(true_structure, 1, 2), pred_structure)
    u, s, vh = torch.linalg.svd(m)

    d = torch.sign(torch.linalg.det(torch.matmul(u, vh)))
    a = (
        torch.eye(embedding_size)
        .reshape((1, embedding_size, embedding_size))
        .repeat_interleave(batch_size, dim=0)
    )
    a[:, -1, -1] = d

    r = torch.matmul(torch.matmul(u, a), vh)

    pred_structure = torch.transpose(
        torch.matmul(r, torch.transpose(pred_structure, 1, 2)), 1, 2
    )

    return torch.mean(torch.sum(torch.square(pred_structure - true_structure), axis=2))
