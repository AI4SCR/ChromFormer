#!/usr/bin/env python
# coding: utf-8


# %%
import re
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from dotenv import load_dotenv
import os
import random
from pathlib import Path

import pytorch_lightning as pylight
from pytorch_lightning.callbacks import ModelCheckpoint

load_dotenv()

# %% IMPORTS
"""
# from ChromFormer.data_generation.Uniform_Cluster_Walk import synthetic_biological_uniform_data_generator,
"""

# generate_biological_structure
from ChromFormer.generator import generate_biological_structure
import ChromFormer.plotting as pl
from ChromFormer.io.import_utils import set_logits_data
from ChromFormer.models import TransConf
from ChromFormer.datasets import Trussart, SyntheticDataset
from ChromFormer.generator import generate_hic
from ChromFormer.io.export_utils import save_structure, make_gif
from ChromFormer.models.lightning_module import LitChromFormer
from ChromFormer.metrics.metrics import kabsch_distance_numpy, kabsch_superimposition_numpy, scale_logits, mse_unscaled_scaled
from ChromFormer.models.lddt_tools import lddt, get_confidence_metrics
from ChromFormer.models.calibration_nn import (
    ModelWithTemperature,
    isotonic_calibration,
    beta_calibration,
)

# %% PARAMETERS
"""
The following uses the package python-dotenv that can be installed by pip to load the variable that contains your path to the data folder in a .env file
"""

DATA_DIR = Path(os.environ.get("DATA_DIR"))
DATA_PATH = DATA_DIR / "demo"  # Folder to which the training and testing data will be saved.
DATA_PATH.mkdir(exist_ok=True, parents=True)

OUTPUT_DIR = Path(os.environ.get('OUTPUT_DIR')) / 'demo'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

TRAIN_DATASET_SIZE = NB_TRAINING = 200  # training dataset size
TEST_DATASET_SIZE = NB_testing = 100  # testing dataset size
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

# Model relevant parameters
NB_HEAD = 2  # Number of head per Transformer Encoder
NB_HIDDEN = 100  # The first feedforward dimension and the Encoding dimension used to project the data before making it go through the transformer
NB_LAYERS = 1  # Number of transformer layers
DROPOUT = 0.1  # Dropout rate
SECD_HID = 48  # Second feedforward dimension of the transformer encoder
ZERO_INIT = False  # Whether to use Zero Initialisation or Xavier initialisation for confidence logits linear layer weights
NUM_BINS_LOGITS = 100  # Number of confidence logit classes
NB_EPOCHS = 40
BATCH_SIZE = 10
EMBEDDING_SIZE = 3  # Simple 3D structure embedding dimension
LAMBDA_BIO = 0.1  # Biological loss weight
LAMBDA_KABSCH = 0.1  # Kabsch loss weight
LAMBDA_LDDT = 0.1  # LDDT confidence loss weight

# %% set seeds
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# %% ## Data Generation
"""
The next part shows how to create a single structure. This structure is then plotted. This is done to find the desired 
parameters and to see how a structure with these parameters would look like. 
Important parameters are the following:
    nb_nodes representing the number of points in the structure,
    delta representing how smooth the structure should be, 
    sigma representing how compact the overall structure should be, 
    cluster_sigma representing how compact TADs should be, 
    cluster_proba representing the probability of entering a TAD along the structure, 
    aging_step decides how many steps must be taken before being elligible to enter a new cluster, and finally 
    nb_point_cluster decides how many points must be in each TADs. 

Other tweeking parameters are usually unused with step2 being false but are documented in the package documentation.
"""
synthetic_biological_structure = generate_biological_structure(
    nb_nodes=NB_BINS,
    delta=DELTA,
    start_sigma=ST_SIG,
    end_sigma=END_SIG,
    sigma=SIG,
    cluster_sigma=CLUST_SIG,
    cluster_proba=CLUST_PROB,
    step2=SECONDSTEP,
)

fig = pl.structure_in_sphere(synthetic_biological_structure)
# fig.show(renderer="browser")

# %% # Imports the trussart ground truth HiC used as a target distribution to match our generated HiCs.
trussart = Trussart()
trussart_hic, trussart_structures, trussart_distances = trussart.hic, trussart.structures, trussart.distances

# %% # HiC MATRIX GENERATION
"""
Then HiC matrices can be generated for a given structure the following way. The function takes a target HiC,
the synthetically generated structure.
Then the following important parameters are used and set to True:
    use_ice deciding whether to use ICE normalisation with z_score, 
    use_ot which decides whether to transport the generated HiC to match the target HiC.

The remaining parameters are usually set to false. The 
    exponent decides as to what should be the alpha in the inverse alpha root power function (distance to HiC).

The following is an HiC generated using optimal transport with minmax and ICE normalisation. 
Parameters can be played with untils the desired ones are found.
"""

rng = np.random.RandomState(SEED)
new_hic, orig_hic, Xs, Xt = generate_hic(
    synthetic_biological_structure,
    target_HiC=trussart_hic,
    use_ice=True,
    use_minmax=True,
    use_ot=True,
    use_softmax=False,
    seed=42,
    plot_optimal_transport=True,
    exponent=1
)

fig = pl.optimal_transport(Xs, Xt, new_hic)
fig.show()

fig = pl.hic(new_hic)
fig.show()

fig = pl.hic(trussart_hic)
fig.show()

# %% # Generate Synthetic Data
"""
The following section uses the desired parameters to output NB_TRAINING training data and NB_testing testing data.
"""

# generate synthetic data
synthetic_train = SyntheticDataset(path_save=OUTPUT_DIR / 'train', n_structures=NB_TRAINING,
                                   target_HiC=trussart_hic, force_generation=True, seed=SEED)
synthetic_test = SyntheticDataset(path_save=OUTPUT_DIR / 'test', n_structures=NB_testing,
                                  target_HiC=trussart_hic, force_generation=True, seed=SEED)

# %% Data Loader

# train
train_loader = DataLoader(synthetic_train, batch_size=BATCH_SIZE, shuffle=True)

# test
test_loader = DataLoader(synthetic_test, batch_size=BATCH_SIZE)

test_frac = 0.2
test_train_calib, test_test_calib = random_split(dataset=synthetic_test, lengths=[1 - test_frac, test_frac],
                                                 generator=torch.Generator().manual_seed(42))
test_train_calib_loader = DataLoader(test_train_calib, batch_size=BATCH_SIZE)
test_test_calib_loader = DataLoader(test_test_calib, batch_size=BATCH_SIZE)

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

# %% Training
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
for ckpt in CKPT_PATH.glob(".ckpt"): ckpt.unlink()
checkpoint_callback = ModelCheckpoint(dirpath=CKPT_PATH, save_top_k=-1, save_last=True)  # save model after every epoch
trainer = pylight.Trainer(min_epochs=NB_EPOCHS, max_epochs=NB_EPOCHS, callbacks=[checkpoint_callback])

trainer.fit(model=model_light, train_dataloaders=train_loader)

test_dataloaders = {'train': train_loader, 'test': test_loader, 'trussart': trussart_loader}
test_results = trainer.test(model=model_light, dataloaders=[train_loader, test_loader, trussart_loader])

# %% Create GIF
STRUCTURE_OUTPUT_PATH = OUTPUT_DIR / "images"

for ckpt_path in CKPT_PATH.glob("*.ckpt"):
    re_epoch_number = re.compile("epoch=([0-9]+)")
    search = re_epoch_number.search(str(ckpt_path))
    if search is None:
        continue
    epoch = int(search.groups()[0])

    epoch_model = LitChromFormer.load_from_checkpoint(ckpt_path, model=model, distance_loss_fct=distance_loss_fct)

    epoch_model.eval()
    save_structure(
        STRUCTURE_OUTPUT_PATH,
        epoch_model.model,
        epoch,
        trussart_structures,
        trussart_hic,
        NB_BINS,
        BATCH_SIZE,
        EMBEDDING_SIZE,
        True)

# # Make a gif of the structure in time
GIT_MODEL_TRAINING_PATH = OUTPUT_DIR / "gifs" / 'trussart_linear.gif'
GIT_MODEL_TRAINING_PATH.parent.mkdir(parents=True, exist_ok=True)

make_gif(STRUCTURE_OUTPUT_PATH, GIT_MODEL_TRAINING_PATH)

# %% best synthetically predicted structure
"""
The Following takes one of the best synthetically predicted structure and plots it along with the true synthetic
structure.
"""
ckpt_path = CKPT_PATH / 'last.ckpt'
last_model = LitChromFormer.load_from_checkpoint(ckpt_path, model=model, distance_loss_fct=distance_loss_fct)
last_model.eval()

kabsch_distances = []
true_structures, pred_structures = [], []
test_loader_bs1 = DataLoader(synthetic_test, batch_size=1)
for batch in test_loader_bs1:
    true_hic, true_structure, true_distance = batch
    pred_structure, pred_distance, logits = last_model.model(true_hic)

    pred_structure, pred_distance, logits = pred_structure.detach().numpy(), pred_distance.detach().numpy(), logits.detach().numpy()
    pred_structure, pred_distance, logits = pred_structure.squeeze(), pred_distance.squeeze(), logits.squeeze()
    true_structure = true_structure.numpy().squeeze()

    d = kabsch_distance_numpy(pred_structure, true_structure)
    kabsch_distances.append(d)
    true_structures.append(true_structure)
    pred_structures.append(pred_structure)

sorted_kabsch = np.argsort(kabsch_distances)
idx = sorted_kabsch[2]

true_structure = true_structures[idx]
pred_structure = pred_structures[idx]

pred_structure_superposed, true_structure_superposed = kabsch_superimposition_numpy(pred_structure, true_structure)

x_pred = pred_structure_superposed[:, 0]
y_pred = pred_structure_superposed[:, 1]
z_pred = pred_structure_superposed[:, 2]

x_true = true_structure_superposed[:, 0]
y_true = true_structure_superposed[:, 1]
z_true = true_structure_superposed[:, 2]

colorscale1 = np.asarray(range(len(x_true)))
colorscale2 = np.asarray(range(len(x_pred)))
color1 = "Viridis"
color2 = "Viridis"

fig = pl.true_pred_structures(
    x_pred,
    y_pred,
    z_pred,
    x_true,
    y_true,
    z_true,
    colorscale1,
    colorscale2,
    color1,
    color2,
)
fig.show(renderer='browser')

# Shape comparison
print("Kabsch distance is " + str(kabsch_distance_numpy(pred_structure, true_structure) / 3))

# %% Trussart Prediction Visualisation
"""
Plot consensus Trussart against prediction 
"""

trussart_consensus_structure = np.mean(trussart.structures, axis=0)

# Trussart predicted structure
torch_trussart_hic = torch.FloatTensor(trussart_hic)
torch_trussart_hic = torch_trussart_hic[None,]

pred_structure, pred_distance, logits = last_model.model(torch_trussart_hic)
pred_structure = pred_structure.detach().numpy().squeeze()

# Superpose structure using Kabsch algorithm
pred_structure_superposed, trussart_consensus_structure_superposed = kabsch_superimposition_numpy(pred_structure,
                                                                                                  trussart_consensus_structure)

# Plot and compare the two structures
x_pred = pred_structure_superposed[:, 0]
y_pred = pred_structure_superposed[:, 1]
z_pred = pred_structure_superposed[:, 2]

x_true = trussart_consensus_structure_superposed[:, 0]
y_true = trussart_consensus_structure_superposed[:, 1]
z_true = trussart_consensus_structure_superposed[:, 2]

colorscale1 = np.asarray(range(len(x_true)))
colorscale2 = np.asarray(range(len(x_pred)))
color1 = "Viridis"
color2 = "Viridis"

fig = pl.true_pred_structures(
    x_pred,
    y_pred,
    z_pred,
    x_true,
    y_true,
    z_true,
    colorscale1,
    colorscale2,
    color1,
    color2,
)
fig.show(renderer="browser")

# Shape comparison
print(
    "Kabsch distance is "
    + str(
        kabsch_distance_numpy(pred_structure, trussart_consensus_structure)
    )
)

# %% # predicted and true lddt prediction confidence of the trussart structure
confidence_metrics, pLLDTs = get_confidence_metrics(
    logits.detach().numpy()[0]
)
print(confidence_metrics)

value = lddt(
    torch.from_numpy(pred_structure_superposed).unsqueeze(0),
    torch.from_numpy(true_structure_superposed).unsqueeze(0),
    per_residue=True,
)
print(torch.mean(value))

# %% # Temperature Scaling
"""
Temperature Scaling with predicted scaled confidences and the Mean Squared error of calibrated and uncalibrated confidences
"""
orig_model = LitChromFormer.load_from_checkpoint(ckpt_path, model=model, distance_loss_fct=distance_loss_fct)
orig_model.eval()

device = 'cpu'
valid_loader = test_train_calib_loader
logits_test_temp, labels_test_temp = set_logits_data(
    test_test_calib_loader,
    device,
    last_model.model,
    BATCH_SIZE,
    NB_BINS,
    EMBEDDING_SIZE,
    NUM_BINS_LOGITS,
)
scaled_model = ModelWithTemperature(
    orig_model.model, device, BATCH_SIZE, NB_BINS, EMBEDDING_SIZE, NUM_BINS_LOGITS
)
scaled_model.set_temperature(valid_loader)
m = torch.nn.LogSoftmax(dim=1)
nll_criterion = torch.nn.BCEWithLogitsLoss()
logits_test_temps_scaled = scaled_model.temperature_scale(logits_test_temp)
confidence_metric_scaled, plddt_scaled = scale_logits(
    logits, scaled_model, NB_BINS
)
print(confidence_metric_scaled)
mse_unscalled, mse_scalled = mse_unscaled_scaled(value, pLLDTs, plddt_scaled)
print(mse_unscalled)
print(mse_scalled)

# %% # Isotonic Regression Calibration
"""
Isotonic Regression Calibration with predicted scaled confidences and the Mean Squared error of calibrated and uncalibrated confidences
"""

valid_loader = test_train_calib_loader
logits_test_temp, labels_test_temp = set_logits_data(
    test_train_calib_loader,
    device,
    last_model.model,
    BATCH_SIZE,
    NB_BINS,
    EMBEDDING_SIZE,
    NUM_BINS_LOGITS,
)
confidence_metric_iso, pLDDT_iso = isotonic_calibration(
    logits_test_temp, labels_test_temp, logits
)
print(confidence_metric_iso)
mse_unscalled, mse_scalled = mse_unscaled_scaled(value, pLLDTs, pLDDT_iso)
print(mse_unscalled)
print(mse_scalled)

# %% # Beta Calibration
"""
Beta Calibration with predicted scaled confidences and the Mean Squared error of calibrated and uncalibrated confidences
"""

valid_loader = test_train_calib_loader
logits_test_temp, labels_test_temp = set_logits_data(
    test_train_calib_loader,
    device,
    last_model.model,
    BATCH_SIZE,
    NB_BINS,
    EMBEDDING_SIZE,
    NUM_BINS_LOGITS,
)
confidence_metric_beta, pLDDT_beta = beta_calibration(
    logits_test_temp, labels_test_temp, logits
)
print(confidence_metric_beta)
mse_unscalled, mse_scalled = mse_unscaled_scaled(value, pLLDTs, pLDDT_beta)
print(mse_unscalled)
print(mse_scalled)
