import numpy as np
from sklearn.preprocessing import MinMaxScaler


def import_fission_yeast(path):

    FISSION_YEAST_HIC_PATH = (
        f"{path}/fission_yeast/hic_matrices/GSM2446256_HiC_20min_10kb_dense.matrix"
    )
    fission_yeast_hic = np.loadtxt(FISSION_YEAST_HIC_PATH, dtype="f", delimiter=" ")
    scaler = MinMaxScaler()
    fission_yeast_hic = scaler.fit_transform(fission_yeast_hic)
    return fission_yeast_hic