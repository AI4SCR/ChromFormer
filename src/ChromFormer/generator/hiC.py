import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance_matrix
from ..utils.normalisation import ice
from ..utils.optimal_transport import ot_data, transport


def compute_hic_matrix(distance_matrix: np.ndarray, alpha: int) -> np.ndarray:
    """Computes the distance to interaction calculation

    In order to compute the interaction from the distance, this function computes 1/root alpha of the distance matrix

    Args:
        distance_matrix: array of the distance matric
        alpha: integer to know which root alpha power to use

    Returns:
        hic_matrix: array of the interaction matric
    """

    distance_matrix = np.where(distance_matrix == 0, np.inf, distance_matrix)

    hic_matrix = np.zeros((len(distance_matrix), len(distance_matrix)))
    hic_matrix = np.where(
        distance_matrix == np.inf, hic_matrix, np.power(distance_matrix, -1 / alpha)
    )

    return hic_matrix


def generate_hic(
    rng,
    synthetic_biological_structure: np.ndarray,
    trussart_hic: np.ndarray,
    use_ice: bool = True,
    use_minmax: bool = False,
    use_ot: bool = True,
    use_softmax: bool = False,
    seed: int = 42,
    plot_optimal_transport: bool = False,
    exponent: int = 1,
):
    """This function is used to generate hic from a given structure matrix that depends on parameters

    Args:
        rng: seed random state to use and that is passed accross functions
        synthetic_biological_structure: array of the structure to use to generate an interaction matrix
        trussart_hic: array of the trussart hic to use for optimal transport
        use_ice: boolean that declares if ice normalisation should be used
        use_minmax: boolean that declares if minmax scaling should be used
        use_ot: boolean that declares if optimal transport should be used
        use_softmax: boolean that declares if softmax should be used instead of optimal transport usually
        seed: seed integer
        plot_optimal_transport: boolean that says if we should plot the histograms of optimal transport
        exponent: the exponent that will be used to compute from distance to interaction

    Returns:
        hic_matrix: array of the generated hic matrix
    """

    scaler = MinMaxScaler()
    precomputed_distances = distance_matrix(
        synthetic_biological_structure, synthetic_biological_structure
    )
    hic_matrix = compute_hic_matrix(precomputed_distances, exponent)

    if use_ice:
        hic_matrix = ice(hic_matrix)

    if use_minmax:
        hic_matrix = scaler.fit_transform(hic_matrix)

    if use_ot:
        orig_hic = hic_matrix
        # rng = np.random.RandomState(seed)
        xs, xt, x1, x2 = ot_data(hic_matrix, trussart_hic, rng)
        hic_matrix, i2te = transport(
            xs, xt, x1, x2, hic_matrix.shape, trussart_hic.shape
        )

    if use_softmax:
        hic_matrix = 1 / (1 + np.exp(-8 * hic_matrix + 4))

    if plot_optimal_transport:
        return hic_matrix, orig_hic, xs, xt
    return hic_matrix
