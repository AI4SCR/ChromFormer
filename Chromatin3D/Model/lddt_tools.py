import torch
import numpy as np
import scipy
import torch.nn.functional as f


def lddt(pred_structure, true_structure,cutoff=15.,per_residue=False):
  # Compute true and predicted distance matrices.
    dmat_true = torch.sqrt(1e-10 + torch.sum(
        (true_structure[:, :, None] - true_structure[:, None, :])**2, dim=-1))
    dmat_predicted = torch.sqrt(1e-10 + torch.sum(
        (pred_structure[:, :, None] -
        pred_structure[:, None, :])**2, dim=-1))

    dists_to_score = (
        (dmat_true).type(torch.FloatTensor) *(1. - torch.eye(dmat_true.shape[1]))  # Exclude self-interaction.
    )

  # Shift unscored distances to be far away.
    dist_l1 = torch.abs(dmat_true - dmat_predicted)
    
  # True lDDT uses a number of fixed bins.
  # Normalize over the appropriate axes.
    relative_error = torch.div(dist_l1, torch.abs(dmat_true))

    score = 1 - relative_error#######
    score = torch.nn.functional.relu(score)######

    reduce_axes = (-1,) if per_residue else (-2,-1)
    norm = 1. / (1e-10 + torch.sum(dists_to_score, dim=reduce_axes))
    score = norm * (1e-10 + torch.sum(dists_to_score * score, dim=reduce_axes))

    return score

def softmax_cross_entropy(logits, labels):
    #"""Computes softmax cross entropy given logits and one-hot class labels."""
    m = torch.nn.LogSoftmax(dim=2) ##dim 1 before
    loss = -torch.sum(labels * m(logits), dim=0)
    return torch.as_tensor(loss)

def loss_lddt(pred_structure, true_structure, logits, num_bins_logits):
    

    # Shape (num_res,)
    lddt_ca = lddt(
        # Shape (batch_size, num_res, 3)
        pred_structure,
        # Shape (batch_size, num_res, 3)
        true_structure,
        # Shape (batch_size, num_res, 1)
        cutoff=15.,
        per_residue=True)
 
    num_bins = num_bins_logits
    bin_index = torch.floor(lddt_ca * num_bins).type(torch.torch.IntTensor)

    # protect against out of range for lddt_ca == 1
    bin_index = torch.minimum(bin_index, torch.tensor(num_bins, dtype=torch.int) - 1)
    lddt_ca_one_hot = f.one_hot(bin_index.to(torch.int64), num_classes=num_bins)

    errors = softmax_cross_entropy(logits=logits, labels=lddt_ca_one_hot.detach())

    # Shape (num_res,)

    loss = torch.mean(errors)

    return loss

def compute_plddt(logits: np.ndarray) -> np.ndarray:
    """Computes per-residue pLDDT from logits.
    Args:
    logits: [num_res, num_bins] output from the PredictedLDDTHead.
    Returns:
    plddt: [num_res] per-residue pLDDT.
    """
    num_bins = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bin_centers = np.arange(start=0 * bin_width, stop=1.0, step=bin_width)
    probs = scipy.special.softmax(logits, axis=-1)
    predicted_lddt_ca = np.sum(probs * bin_centers[None, :], axis=-1)
    return predicted_lddt_ca * 100

def get_confidence_metrics(pred_logits):
    #"""Post processes prediction_result to get confidence metrics."""
    pLLDTs = compute_plddt(
      pred_logits)

    confidence_metrics = np.mean(
        pLLDTs)

    return confidence_metrics, pLLDTs

def compute_plddt_post_soft(logits: np.ndarray) -> np.ndarray:
    """Computes per-residue pLDDT from logits.
    Args:
    logits: [num_res, num_bins] output from the PredictedLDDTHead.
    Returns:
    plddt: [num_res] per-residue pLDDT.
    """
    num_bins = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bin_centers = np.arange(start=0 * bin_width, stop=1.0, step=bin_width)
    probs = np.divide(logits,np.sum(logits, axis=1)[:, np.newaxis])
    predicted_lddt_ca = np.sum(probs * bin_centers[None, :], axis=-1)
    return predicted_lddt_ca * 100

def get_confidence_metrics_post_soft(pred_logits):
    #"""Post processes prediction_result to get confidence metrics."""
    pLLDTs = compute_plddt_post_soft(
      pred_logits)

    confidence_metrics = np.mean(
        pLLDTs)

    return confidence_metrics, pLLDTs