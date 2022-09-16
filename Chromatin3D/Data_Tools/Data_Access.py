"""This file contains all the relevant functions needed to retrieve the data and load the data loader"""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2022
ALL RIGHTS RESERVED
"""

from asyncio import constants
from typing import Tuple
import os
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
import torch
from tqdm import tqdm
from ..Model.lddt_tools import lddt
import torch.nn.functional as f


def last_4digits(x: str) -> str:
    """Function that returns the last 4 digit of our file name string

    Args:
        x: string containing the filename
    """
    return x[-8:-4]


def get_data_from_path(
    path: constants,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


class VanillaDataset(InMemoryDataset):
    """Class that contains functions to generate the Data Loader

    Generates a Data Loader containing the interaction and distances matrices as well as the structures

    Args:
        root: path to generate the data
        transform:
        pre_transform:
        is_training: boolean to know whether to generate training data or testing data
        dataset_size: integer of how many data to generate
        hics: array of the hics to store
        structures: array of the structures to store
        distances: array of the distances matrices to store
    """

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        is_training: bool = True,
        dataset_size: int = 800,
        hics: np.ndarray = None,
        structures: np.ndarray = None,
        distances: np.ndarray = None,
    ):
        """Initialises the Vanilla Dataset"""
        self.is_training = is_training
        self.dataset_size = dataset_size
        self.hics = hics
        self.structures = structures
        self.distances = distances
        super(VanillaDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self) -> list:
        """Returns the name of the file to write data on

        Returns:
            returns the name of the file
        """
        if self.is_training:
            return ["synthetic_biological_trussart_linear_train_data.txt"]
        else:
            return ["synthetic_biological_trussart_linear_test_data.txt"]

    def download(self):
        pass

    def process(self) -> None:
        """Processes the according data to be saved"""
        data_list = []
        if self.is_training:
            dataset_size = self.dataset_size
        else:
            dataset_size = self.dataset_size

        for i in tqdm(range(dataset_size)):

            transfer_learning_hic = self.hics[i]
            transfer_learning_structure = self.structures[i]
            transfer_learning_distance_matrix = self.distances[i]

            hic_matrix = torch.FloatTensor(transfer_learning_hic)
            structure_matrix = torch.FloatTensor(transfer_learning_structure)
            distance_matrix = torch.FloatTensor(transfer_learning_distance_matrix)

            data = Data(
                hic_matrix=hic_matrix,
                structure_matrix=structure_matrix,
                distance_matrix=distance_matrix,
            )
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
def set_logits_data(loader, device, model, batch_size, nb_bins, embedding_size, num_bins_logits):
    logits_list = []
    labels_list = []
    for data in loader:
        data = data.to(device)
            
        pred_structure, pred_distance, logits = model(data.hic_matrix)

        pred_structure = pred_structure.detach().cpu()
            
        true_hic = data.hic_matrix.detach().cpu()
        true_structure = data.structure_matrix.detach().cpu()
        true_distance = data.distance_matrix.detach().cpu()
            
        true_structure = torch.reshape(true_structure, (batch_size, nb_bins, embedding_size))
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
        label = f.one_hot(bin_index.to(torch.int64), num_classes=num_bins)
        label = torch.reshape(label, (batch_size*nb_bins, num_bins))

        logits = torch.reshape(logits, (batch_size*nb_bins, num_bins))
        logits_list.append(logits)
        labels_list.append(label)
    logits_test_temp = torch.cat(logits_list)
    labels_test_temp = torch.cat(labels_list).type(torch.float)
    return logits_test_temp, labels_test_temp