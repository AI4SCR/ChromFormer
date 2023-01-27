# %%
import os
from pathlib import Path
from dotenv import load_dotenv

from ChromFormer.plotting import structure_in_sphere
from ChromFormer.data_generation.Uniform_Cluster_Walk import (
    generate_biological_structure,
)

load_dotenv()
# The following uses the package python-dotenv that can be installed by pip to load the variable that contains your path to the data folder in a .env file
DATA_DIR = os.environ.get("DATA_DIR")
## Folder to which the training and testing data will be saved.
DATA_PATH = f"{DATA_DIR}/demo/"
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
ANGLE_PRED = EMBEDDING_SIZE = 3  # Simple 3D structure embedding dimension
LAMBDA_BIO = 0.1  # Biological loss weight
LAMBDA_KABSCH = 0.1  # Kabsch loss weight
LAMBDA_LDDT = 0.1  # LDDT confidence loss weight

# %%
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
# %%
fig = plot_structure_in_sphere(synthetic_biological_structure)
fig.show(renderer="browser")
path_fig = Path("~/Downloads").expanduser() / "structure-1.pdf"
fig.write_image(str(path_fig))
