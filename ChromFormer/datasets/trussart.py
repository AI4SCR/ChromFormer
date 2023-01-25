from pathlib import Path
from datasets import BaseDataset, DownloadMixIn

class Trussart(BaseDataset, DownloadMixIn):
    path = Path('~/.ai4src/ChromFormer/datasets/20150115_Trussart_Dataset.zip').expanduser()  # location where the file is downloaded to
    url = 'https://figshare.com/ndownloader/files/38721213'
    data = None

    root_model = 'Toy_Models'
    genomic_architecture = 'res_150_TAD'
    set_number = 0
    path_models = Path(f'{root_model}/{genomic_architecture}/set_{set_number}')

    root_HiC = 'Simulated_HiC'
    HiC_alpha = 150
    path_HiC = Path(f'{root_HiC}/{genomic_architecture}/{genomic_architecture}like_alpha{HiC_alpha}set_{set_number}.mat')

    def __init__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        super().__init__()

    def __getitem__(self, index):
        return self.data

    def __len__(self):
        return None

    def setup(self):
        """Accesses the stored data to return the Trussart structures and hic

        Saves
            trussart_hic: numpy array of the trussart interaction matrices
            trussart_structures: numpy array of the the trussart structures
        in self.data

        """
        import tarfile
        from sklearn.preprocessing import MinMaxScaler
        import numpy as np
        import os

        with tarfile.open(self.path, 'r:gz') as tar:
            tar.extractall(self.path.parent)

        trussart_hic = np.loadtxt(self.path_HiC, dtype="f", delimiter="\t")
        scaler = MinMaxScaler()
        trussart_hic = scaler.fit_transform(trussart_hic)
        trussart_structures = []

        file_list = os.listdir(self.path_models)
        file_list = filter(lambda f: f.endswith(".xyz"), file_list)

        for file_name in file_list:
            current_trussart_structure = np.loadtxt(
                self.path_models / file_name, dtype="f", delimiter="\t"
            )
            current_trussart_structure = current_trussart_structure[:, 1:]
            trussart_structures.append(current_trussart_structure)

        self.data = trussart_hic, trussart_structures
        