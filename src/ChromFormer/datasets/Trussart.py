from pathlib import Path
from torch.utils.data import Dataset
from scipy.spatial import distance_matrix
from dataset import BaseDataset, DownloadMixIn
from ..processing.normalisation import centralize_and_normalize_numpy

class Trussart(Dataset, BaseDataset, DownloadMixIn):
    path = Path(
        "~/.ai4src/ChromFormer/datasets/20150115_Trussart_Dataset.zip"
    ).expanduser()  # location where the file is downloaded to
    url = "https://figshare.com/ndownloader/files/38945396"

    root_model = "Toy_Models"
    genomic_architecture = "150_TAD"
    set_number = 0
    path_models = (
        Path(path.stem) / f"{root_model}/res_{genomic_architecture}/set_{set_number}/"
    )

    root_HiC = "Simulated_HiC"
    HiC_alpha = 150
    path_HiC = (
        Path(path.stem)
        / f"{root_HiC}/res_{genomic_architecture}/{genomic_architecture}like_alpha_{HiC_alpha}_set{set_number}.mat"
    )

    def __init__(self, force_download=False):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        super().__init__(force_download=force_download)

    def __getitem__(self, index):
        return self.hic, self.structures[index], self.distances[index]

    def __len__(self):
        return len(self.structures)

    def setup(self):
        """Accesses the stored data to return the Trussart structures and hic

        Saves
            hic: numpy array of the trussart interaction matrix
            structures: numpy array of the trussart structures for the HiC

        """
        from sklearn.preprocessing import MinMaxScaler
        import numpy as np
        import os
        from zipfile import ZipFile

        with ZipFile(self.path, "r") as zip:
            path_HiC_extracted = self.path.parent / self.path_HiC
            path_models_extracted = self.path.parent / self.path_models

            files_to_extract = zip.namelist()
            files_to_extract = list(
                filter(lambda x: str(self.path_models) + "/" in x, files_to_extract)
            )
            files_to_extract += [str(self.path_HiC)]

            zip.extractall(members=files_to_extract, path=self.path.parent)

            trussart_hic = np.loadtxt(path_HiC_extracted, dtype="f", delimiter="\t")
            scaler = MinMaxScaler()
            trussart_hic = scaler.fit_transform(trussart_hic)
            trussart_structures = []
            distances = []

            file_list = os.listdir(path_models_extracted)
            file_list = filter(lambda f: f.endswith(".xyz"), file_list)

            for file_name in file_list:
                current_trussart_structure = np.loadtxt(
                    path_models_extracted / file_name, dtype="f", delimiter="\t"
                )
                current_trussart_structure = current_trussart_structure[:, 1:]
                current_trussart_structure = centralize_and_normalize_numpy(current_trussart_structure)
                trussart_structures.append(current_trussart_structure)

                # compute distance matrix
                distances.append(distance_matrix(current_trussart_structure, current_trussart_structure))

        self.hic = trussart_hic
        self.structures = trussart_structures
        self.distances = distances
