#!/usr/bin/env python
# coding: utf-8

# %%
import numpy as np
import torch
from dotenv import load_dotenv
import os
import random
from pathlib import Path

import pytorch_lightning as pylight
from pytorch_lightning.callbacks import ModelCheckpoint

load_dotenv()
from ChromFormer.datasets import Trussart, SyntheticDataset


# %% PARAMETERS
OUTPUT_DIR = Path(os.environ.get('OUTPUT_DIR')) / 'paper'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

TRAIN_DATASET_SIZE = 800  # training dataset size
VALIDATION_DATASET_SIZE = 100  # testing dataset size
TEST_DATASET_SIZE = 200  # testing dataset size
NB_BINS = 202  # number of points per structure

# Data Generation relevant parameters
DELTA = 0.45  # Smoothness parameter
ST_SIG = 5
END_SIG = 7
SIG = 4  # structure compactness
CLUST_SIG = 1.5  # TADs compactness
CLUST_PROB = 0.1  # Probability of entering a TAD
SECONDSTEP = False
SEED = 42
EXPONENT = 1  # root power value for the inverse function (Distance -> Hi-C)
ICING = True  # Whether to use ICE normalisation with Z_score or not
MINMAXUSE = False  # Whether MinMax needs to be used before optimal transport on the synthetic data or not
TRANSPORTATION = True  # Whether to use optimal transport or not
SOFTMAXING = False  # Whether to use a synthetic to true HiC softmax function or not. Not needed if already using optimal transport

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# %%
trussart = Trussart()
trussart_hic, trussart_structures, trussart_distances = trussart.hic, trussart.structures, trussart.distances

# %% # Generate Synthetic Data
synthetic_train = SyntheticDataset(path_save=OUTPUT_DIR / 'train', n_structures=TRAIN_DATASET_SIZE,
                                   target_HiC=trussart_hic, seed=SEED)
# synthetic_train = SyntheticDataset(path_save=OUTPUT_DIR / 'validate', n_structures=TRAIN_DATASET_SIZE,
#                                    target_HiC=trussart_hic, seed=SEED)
synthetic_test = SyntheticDataset(path_save=OUTPUT_DIR / 'test', n_structures=TEST_DATASET_SIZE,
                                  target_HiC=trussart_hic, seed=SEED)

