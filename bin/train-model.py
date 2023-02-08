#!/usr/bin/env python
# coding: utf-8


# %%
import numpy as np
import torch
from torch.utils.data import DataLoader
from dotenv import load_dotenv
import os
import random
from pathlib import Path

import pytorch_lightning as pylight
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
load_dotenv()

from ChromFormer.models import TransConf
from ChromFormer.datasets import Trussart, SyntheticDataset
from ChromFormer.models.lightning_module import LitChromFormer

# %% CONFIGURATION
OUTPUT_DIR = Path(os.environ.get('OUTPUT_DIR')) / 'paper'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Model relevant parameters
NB_HEAD = 2
NB_HIDDEN = 100
NB_LAYERS = 1
DROPOUT = 0.1
SECD_HID = 48
ZERO_INIT = False
EXPONENT = 1
NUM_BINS_LOGITS = 100
NB_EPOCHS = 96
SEED = 2
BATCH_SIZE = 10
NB_BINS = 202
EMBEDDING_SIZE = 3
ANGLE_PRED = 3
LAMBDA_BIO = 0.1
LAMBDA_KABSCH = 0.1
LAMBDA_LDDT = 0.1

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# %%
trussart = Trussart()
trussart_hic, trussart_structures, trussart_distances = trussart.hic, trussart.structures, trussart.distances

# %% # Load synthetic Data
synthetic_train = SyntheticDataset(path_load=OUTPUT_DIR / 'train')
synthetic_test = SyntheticDataset(path_load=OUTPUT_DIR / 'test')

# %% Data Loader

# train
train_loader = DataLoader(synthetic_train, batch_size=BATCH_SIZE, shuffle=True)

# test
test_loader = DataLoader(synthetic_test, batch_size=BATCH_SIZE)

# test_frac = 0.2
# test_train_calib, test_test_calib = random_split(dataset=synthetic_test, lengths=[1 - test_frac, test_frac],
#                                                  generator=torch.Generator().manual_seed(42))
# test_train_calib_loader = DataLoader(test_train_calib, batch_size=BATCH_SIZE)
# test_test_calib_loader = DataLoader(test_test_calib, batch_size=BATCH_SIZE)

# Trussart data
trussart_loader = DataLoader(trussart, batch_size=BATCH_SIZE)

# %% Model
"""
The device on which to run the model is first selected, then the model is declared with the following parameters:
    NB_BINS is the number of loci in the structure,
    EMBEDDING_SIZE representing the 3d dimension embedding,
    NUM_BINS_LOGITS represents the number of confidence classes,
    ZERO_INIT decides whether to use a zero initialisation of the confidence learning weights or a Xavier initialisation
    NB_HEAD is the number of heads in the encoder transformer layer,
    NB_HIDDEN is the first d projection dimension parameter as well as the number of feedforward passes in the first
        encoder layer,
    DROPOUT is the dropout rate of the transformer model,
    SECD_HID is the number of feedforward passes in the second encoder transformer layer. Then the optimizer is set to
        run on the model's parameters.
"""

distance_loss_fct = torch.nn.MSELoss()

model = TransConf(
    nb_bins=NB_BINS,
    embedding_size=EMBEDDING_SIZE,
    num_bins_logits=NUM_BINS_LOGITS,
    zero_init=ZERO_INIT,
    nb_head=NB_HEAD,
    nb_hidden=NB_HIDDEN,
    nb_layers=NB_LAYERS,
    dropout=DROPOUT,
    secd_hid=SECD_HID)

# %% Training Configuration
"""
The following first trains the model with
    LAMBDA_KABSCH and LAMBDA_BIO being the weights of the kabsch and biological losses.
    LAMBDA_LDDT is the weight of the confidence loss.
Then the model is evaluated with validation, testing and training results per epoch being stored in an array and being
printed for the user to see.
"""

optimizer = torch.optim.AdamW
optimizer_kwargs = dict(lr=0.0005, weight_decay=1e-5)

model_light = LitChromFormer(model=model,
                             optimizer=optimizer,
                             optimizer_kwargs=optimizer_kwargs,
                             nb_bins=NB_BINS,
                             embedding_size=EMBEDDING_SIZE,
                             lambda_bio=LAMBDA_BIO,
                             lambda_kabsch=LAMBDA_KABSCH,
                             distance_loss_fct=distance_loss_fct,
                             lambda_lddt=LAMBDA_LDDT,
                             num_bins_logits=NUM_BINS_LOGITS)

CKPT_PATH = OUTPUT_DIR / 'checkpoints'
for ckpt in CKPT_PATH.glob(".ckpt"):
    ckpt.unlink()

checkpoint_callback = ModelCheckpoint(dirpath=CKPT_PATH, save_top_k=-1, save_last=True)  # save model after every epoch
logger = CSVLogger(save_dir=OUTPUT_DIR / 'logs-train')
trainer = pylight.Trainer(min_epochs=NB_EPOCHS, max_epochs=NB_EPOCHS,
                          default_root_dir=OUTPUT_DIR,
                          logger=logger,
                          callbacks=[checkpoint_callback])

# %% TRAINING
trainer.fit(model=model_light, train_dataloaders=train_loader)

# %% Evaluation
ckpt_path = CKPT_PATH / 'last.ckpt'
epoch_model = LitChromFormer.load_from_checkpoint(ckpt_path, model=model, distance_loss_fct=distance_loss_fct)
test_dataloaders = {'train': train_loader, 'test': test_loader, 'trussart': trussart_loader}
test_results = trainer.test(model=model_light, dataloaders=[train_loader, test_loader, trussart_loader])