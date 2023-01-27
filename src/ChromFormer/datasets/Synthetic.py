from torch.utils.data import Dataset
import numpy as np
from pathlib import Path

from ..generator import synthetic_biological_uniform_data_generator
from ..io.import_utils import load_from_path_save

# Data Generation relevant parameters
DELTA = 0.45  # Smoothness parameter
ST_SIG = 5
END_SIG = 7
SIG = 4  # structure compactness
CLUST_SIG = 1.5  # TADs compactness
CLUST_PROB = 0.1  # Probability of entering a TAD
SECONDSTEP = False
SEED = 42
NB_BINS = 202
ALPHA = EXPONENT = 1  # root power value for the inverse function (Distance -> Hi-C)
ICING = True  # Whether to use ICE normalisation with Z_score or not
MINMAXUSE = False  # Whether MinMax needs to be used before optimal transport on the synthetic data or not
TRANSPORTATION = True  # Whether to use optimal transport or not
SOFTMAXING = False  # Whether to use a synthetic to true HiC softmax function or not. Not needed if already using optimal transport
AGING_STEP = 30
NB_PER_CLUSTER = 30


class SyntheticDataset(Dataset):

    def __init__(self,
                 n_structures: int,
                 path_save: Path,
                 **kwargs):
        super().__init__()

        default_kwargs = dict(
            nb_bins=NB_BINS,
            delta=DELTA,
            st_sig=ST_SIG,
            end_sig=END_SIG,
            sig=SIG,
            clust_sig=CLUST_SIG,
            clust_prob=CLUST_PROB,
            secondstep=SECONDSTEP,
            seed=SEED,
            alpha=ALPHA,
            icing=ICING,
            minmaxuse=MINMAXUSE,
            transportation=TRANSPORTATION,
            softmaxing=SOFTMAXING,
            aging_step=AGING_STEP,
            nb_per_cluster=NB_PER_CLUSTER)

        self.kwargs = {'path_save': path_save,
                       'n_structures': n_structures,
                       **default_kwargs, **kwargs}
        self.setup()

    def __len__(self):
        return self.kwargs['n_structures']

    def __getitem__(self, item):
        return (self.transfer_learning_hics[item],
                self.transfer_learning_structures[item],
                self.transfer_learning_distances[item])

    def setup(self):
        synthetic_biological_uniform_data_generator(**self.kwargs)
        self.transfer_learning_hics, self.transfer_learning_structures, self.transfer_learning_distances = \
            load_from_path_save(self.kwargs['path_save'])
