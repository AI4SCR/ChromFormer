import numpy as np
from torch_geometric.data import Data, InMemoryDataset
import torch
from tqdm import tqdm
from typing import List
from pathlib import Path

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
            root: Path,
            transform=None,
            pre_transform=None,
            is_training: bool = True,
            dataset_size: int = 800,
            hics: List[np.ndarray] = None,
            structures: List[np.ndarray] = None,
            distances: List[np.ndarray] = None,
    ):
        """Initialises the Vanilla Dataset"""
        self.is_training = is_training
        self.dataset_size = dataset_size
        self.hics = hics
        self.structures = structures
        self.distances = distances
        super(VanillaDataset, self).__init__(str(root), transform, pre_transform)
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
