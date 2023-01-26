"""The following takes use of the POT package with some changes that is under the MIT LICENSE

MIT License

Copyright (c) 2016 Rémi Flamary

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import ot
import numpy as np
from typing import Tuple


def im2mat(img: np.ndarray) -> np.ndarray:
    """Converts an image to matrix (one pixel per line)"""
    return img.reshape((img.shape[0] * img.shape[1], 1))


def mat2im(x: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """Converts back a matrix to an image"""
    return x.reshape(shape)


def minmax(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 1)


def ot_data(
    hic_matrix: np.ndarray, trussart_hic: np.ndarray, rng
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepares the data for optimal transport

    Args:
        hic_matrix: array representing the interaction matrix for a structure
        trussart_hic: array representing the trussart target interaction matrix

    Returns:
        xs: array training input sample
        xt: array training input sample
        x1: array source input sample
        x2 array target input sample
    """
    x1 = im2mat(hic_matrix)
    x2 = im2mat(trussart_hic)
    nb = 500
    idx1 = rng.randint(x1.shape[0], size=(nb,))
    idx2 = rng.randint(x2.shape[0], size=(nb,))

    xs = x1[idx1]
    xt = x2[idx2]
    return xs, xt, x1, x2


def transport(
    xs: np.ndarray,
    xt: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    hic_shape: np.ndarray,
    trussart_shape: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Performs optimal transport by using between images out of sample prediction

    Args:
        xs: array training input sample
        xt: array training input sample
        x1: array source input sample
        x2 array target input sample

    Returns:
        i1te: Transported source samples
        i2te: Transported target samples
    """
    ot_emd = ot.da.EMDTransport()
    ot_emd.fit(Xs=xs, Xt=xt)
    transp_xs_emd = ot_emd.transform(Xs=x1)
    transp_xt_emd = ot_emd.inverse_transform(Xt=x2)

    i1te = minmax(mat2im(transp_xs_emd, hic_shape))
    i2te = minmax(mat2im(transp_xt_emd, trussart_shape))
    return i1te, i2te
