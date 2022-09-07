import ot
import numpy as np
from typing import Tuple



def im2mat(img: np.ndarray) -> np.ndarray:
    """Converts an image to matrix (one pixel per line)"""
    return img.reshape((img.shape[0] * img.shape[1], 1))


def mat2im(X: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """Converts back a matrix to an image"""
    return X.reshape(shape)


def minmax(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 1)


def ot_data(
    hic_matrix: np.ndarray, trussart_hic: np.ndarray, rng
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X1 = im2mat(hic_matrix)
    X2 = im2mat(trussart_hic)
    nb = 500
    idx1 = rng.randint(X1.shape[0], size=(nb,))
    idx2 = rng.randint(X2.shape[0], size=(nb,))

    Xs = X1[idx1]
    Xt = X2[idx2]
    return Xs, Xt, X1, X2


def transport(
    Xs: np.ndarray,
    Xt: np.ndarray,
    X1: np.ndarray,
    X2: np.ndarray,
    hic_shape: np.ndarray,
    trussart_shape: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    M = np.ones((Xs.shape[0], Xs.shape[0]))

    ot_emd = ot.da.EMDTransport()
    ot_emd.fit(Xs=Xs, Xt=Xt)
    transp_Xs_emd = ot_emd.transform(Xs=X1)
    transp_Xt_emd = ot_emd.inverse_transform(Xt=X2)

    I1te = minmax(mat2im(transp_Xs_emd, hic_shape))
    I2te = minmax(mat2im(transp_Xt_emd, trussart_shape))
    return I1te, I2te

