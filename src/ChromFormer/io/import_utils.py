"""This file contains all the relevant functions needed to retrieve the data and load the data loader"""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2022
ALL RIGHTS RESERVED
"""

from sklearn.preprocessing import MinMaxScaler
from asyncio import constants
from typing import Tuple, List
import os
import numpy as np
import torch
import torch.nn.functional as f
from ..models.lddt_tools import lddt
from pathlib import Path


def import_fission_yeast(path):
    FISSION_YEAST_HIC_PATH = (
        f"{path}/fission_yeast/hic_matrices/GSM2446256_HiC_20min_10kb_dense.matrix"
    )
    fission_yeast_hic = np.loadtxt(FISSION_YEAST_HIC_PATH, dtype="f", delimiter=" ")
    scaler = MinMaxScaler()
    fission_yeast_hic = scaler.fit_transform(fission_yeast_hic)
    return fission_yeast_hic


def last_4digits(x: str) -> str:
    """Function that returns the last 4 digit of our file name string

    Args:
        x: string containing the filename
    """
    return x[-8:-4]


def load_from_path_save(
        path: Path,
) -> Tuple[
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray]
]:
    """This function retrieves the distance, hic and structure data from the provided folder

    Args:
        path: constant path to the desired folder

    Returns:
        transfer_learning_hics: array of the training hic
        transfer_learning_structures: array of the training structures
        transfer_learning_distances: array of the training distance matrix
    """

    # Train HIC retrieval
    transfer_learning_hics = []

    file_list = [str(i) for i in (path / "hic_matrices").glob('*.txt')]

    for file_name in sorted(file_list, key=last_4digits):
        current_transfer_learning_hic = np.loadtxt(path / "hic_matrices" / file_name, dtype="f", delimiter=" ")
        transfer_learning_hics.append(current_transfer_learning_hic)

    # Train structure retrieval
    transfer_learning_structures = []
    file_list = [str(i) for i in (path / "structure_matrices").glob('*.txt')]
    for file_name in sorted(file_list, key=last_4digits):
        current_transfer_learning_structure = np.loadtxt(path / "structure_matrices" / file_name, dtype="f",
                                                         delimiter=" ")
        transfer_learning_structures.append(current_transfer_learning_structure)

    # Train distance matrix retrieval
    transfer_learning_distances = []
    file_list = [str(i) for i in (path / "distance_matrices").glob('*.txt')]

    for file_name in sorted(file_list, key=last_4digits):
        current_transfer_learning_distance = np.loadtxt(path / "distance_matrices" / file_name, dtype="f",
                                                        delimiter=" ")
        transfer_learning_distances.append(current_transfer_learning_distance)

    return (
        transfer_learning_hics,
        transfer_learning_structures,
        transfer_learning_distances,
    )


def get_data_from_path(
        path: constants,
) -> Tuple[
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
]:
    """This function retrieves the distance, hic and structure data from the provided folder

    Args:
        path: constant path to the desired folder

    Returns:
        train_transfer_learning_hics: array of the training hic
        test_transfer_learning_hics: array of the testing hic
        train_transfer_learning_structures: array of the training structures
        test_transfer_learning_structures: array of the testing structures
        train_transfer_learning_distances: array of the training distance matrix
        test_transfer_learning_distances: array of the testing distance matric
    """

    # Train HIC retrieval
    train_transfer_learning_hics = []

    file_list = os.listdir(f"{path}/train/hic_matrices/")

    for file_name in sorted(
            filter(lambda x: x.endswith(".txt"), file_list), key=last_4digits
    ):
        current_train_transfer_learning_hic = np.loadtxt(
            f"{path}/train/hic_matrices/" + file_name, dtype="f", delimiter=" "
        )
        train_transfer_learning_hics.append(current_train_transfer_learning_hic)

    # Test HIC retrieval
    test_transfer_learning_hics = []

    file_list = os.listdir(f"{path}/test/hic_matrices/")

    for file_name in sorted(
            filter(lambda x: x.endswith(".txt"), file_list), key=last_4digits
    ):
        current_test_transfer_learning_hic = np.loadtxt(
            f"{path}/test/hic_matrices/" + file_name, dtype="f", delimiter=" "
        )
        test_transfer_learning_hics.append(current_test_transfer_learning_hic)

    # Train structure retrieval
    train_transfer_learning_structures = []

    file_list = os.listdir(f"{path}/train/structure_matrices/")

    for file_name in sorted(
            filter(lambda x: x.endswith(".txt"), file_list), key=last_4digits
    ):
        current_train_transfer_learning_structure = np.loadtxt(
            f"{path}/train/structure_matrices/" + file_name, dtype="f", delimiter=" "
        )
        train_transfer_learning_structures.append(
            current_train_transfer_learning_structure
        )

    # Test structure retrieval
    test_transfer_learning_structures = []

    file_list = os.listdir(f"{path}/test/structure_matrices/")

    for file_name in sorted(
            filter(lambda x: x.endswith(".txt"), file_list), key=last_4digits
    ):
        current_test_transfer_learning_structure = np.loadtxt(
            f"{path}/test/structure_matrices/" + file_name, dtype="f", delimiter=" "
        )
        test_transfer_learning_structures.append(
            current_test_transfer_learning_structure
        )

    # Train distance matrix retrieval
    train_transfer_learning_distances = []

    file_list = os.listdir(f"{path}/train/distance_matrices/")

    for file_name in sorted(
            filter(lambda x: x.endswith(".txt"), file_list), key=last_4digits
    ):
        current_train_transfer_learning_distance = np.loadtxt(
            f"{path}/train/distance_matrices/" + file_name, dtype="f", delimiter=" "
        )
        train_transfer_learning_distances.append(
            current_train_transfer_learning_distance
        )

    # Test distance matrix retrieval
    test_transfer_learning_distances = []

    file_list = os.listdir(f"{path}/test/distance_matrices/")

    for file_name in sorted(
            filter(lambda x: x.endswith(".txt"), file_list), key=last_4digits
    ):
        current_test_transfer_learning_distance = np.loadtxt(
            f"{path}/test/distance_matrices/" + file_name, dtype="f", delimiter=" "
        )
        test_transfer_learning_distances.append(current_test_transfer_learning_distance)

    return (
        train_transfer_learning_hics,
        test_transfer_learning_hics,
        train_transfer_learning_structures,
        test_transfer_learning_structures,
        train_transfer_learning_distances,
        test_transfer_learning_distances,
    )


def set_logits_data(
        loader, device, model, batch_size, nb_bins, embedding_size, num_bins_logits
):
    """Function that generates logits data for calibration

    Args:
        loader: data loader
        device: device to set the data to
        model: model
        batch_size: integer for the batch size
        nb_bins: integer of how many loci
        embedding_size: integer for the 3D dimension
        num_bin_logits: integer for how many confidence bin exist
    """
    logits_list = []
    labels_list = []
    for batch in loader:
        # batch[0] := hic matrix
        # batch[1] := structure
        # batch[2] := distance
        true_hic, true_structure, true_distance = batch

        pred_structure, pred_distance, logits = model(true_hic)

        lddt_ca = lddt(
            # Shape (batch_size, num_res, 3)
            pred_structure,
            # Shape (batch_size, num_res, 3)
            true_structure,
            # Shape (batch_size, num_res, 1)
            cutoff=15.0,
            per_residue=True,
        )

        num_bins = num_bins_logits
        bin_index = torch.floor(lddt_ca * num_bins).type(torch.torch.IntTensor)

        # protect against out of range for lddt_ca == 1
        bin_index = torch.minimum(
            bin_index, torch.tensor(num_bins, dtype=torch.int) - 1
        )
        label = f.one_hot(bin_index.to(torch.int64), num_classes=num_bins)
        label = torch.reshape(label, (batch_size * nb_bins, num_bins))

        logits = torch.reshape(logits, (batch_size * nb_bins, num_bins))
        logits_list.append(logits)
        labels_list.append(label)
    logits_test_temp = torch.cat(logits_list)
    labels_test_temp = torch.cat(labels_list).type(torch.float)
    return logits_test_temp, labels_test_temp
