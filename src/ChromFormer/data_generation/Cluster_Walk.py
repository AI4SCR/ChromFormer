"""This script has the necessary functions that creates 3D clustered random walk structures and generates data"""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2022
ALL RIGHTS RESERVED
"""

from asyncio import constants
import numpy as np
from tqdm import tqdm
from ..Data_Tools.Normalisation import centralize_and_normalize_numpy
from scipy.spatial import distance_matrix
from ..Data_Tools.Data_Calculation import generate_hic
import pandas as pd
from typing import Tuple
import random


class Stepper(object):
    """Class that contains functions to generate the random angle

    Generates the trajectory and the next step to take for the Cluster Walk. It contains the uniform walk that depends angles phi and theta for 3D structures.

    Args:
        delta: an integer that indicates how smooth the structure should be
    """

    def __init__(self, delta: int) -> None:
        """initialises the Stepper class"""

        self.delta = delta
        self.previous_theta = np.random.uniform(0, np.pi)
        self.previous_phi = np.random.uniform(0, 2 * np.pi)
        self.trajectory = [np.zeros(3)]
        self.sphere_center = np.zeros(3)

    @property
    def pos(self) -> np.ndarray:
        """Function that gets the latest position in the trajectory of the random walk

        Returns:
            A numpy array 3D coordinates of the latest position
        """
        return self.trajectory[-1]

    def __step(self, delta: int) -> Tuple[np.ndarray, int, int]:
        """Step that uses the phi and delta angles to find the next 3D coordinates. Angles each depend to a certain degree of the previous angle.

        Args:
            delta: integer of value between 0 and 1 that represents how much the next angle depends on the previous angle

        Returns:
            The new position according to the selected angles
            The generated phi and theta angles
        """
        random_theta = np.random.uniform(0, np.pi)
        random_phi = np.random.uniform(0, 2 * np.pi)
        new_theta = (1 - self.delta) * self.previous_theta + self.delta * random_theta
        new_phi = (1 - self.delta) * self.previous_phi + self.delta * random_phi

        pos = self.pos.copy()
        pos[0] += np.cos(new_phi) * np.sin(new_theta)
        pos[1] += np.sin(new_phi) * np.sin(new_theta)
        pos[2] += np.cos(new_theta)

        return pos, new_theta, new_phi

    def step(self) -> np.ndarray:
        """Function that gets the new angles and position. It adds the new position to the trajectory.

        This function is used if the angle based structure is being built.

        Returns:
            Numpy array of the coordinates of the latest position
        """
        new_pos, new_theta, new_phi = self.__step(delta=1)
        self.trajectory.append(new_pos)
        self.previous_theta = new_theta
        self.previous_phi = new_phi
        return self.pos

    def smooth_step(self, sigma: int) -> np.ndarray:
        """Function that creates a smooth step.

        Creates a smooth step by making sure that the step is contained in the main Gaussian structure sphere (main cluster) that we generated at the beginning with a certain variance
        which specifies how condensed each points must be. Performs rejection sampling of points that do not follow the constrains. It as well takes in consideration if the angles are going in the good
        direction in order not to be rejected as a sampled point.

        Used if an angle based structure is being built.

        Args:
            sigma: integer representing the variance of the sphere

        Returns:
            Numpy array of the coordinates of the latest position
        """
        while True:
            new_pos, new_theta, new_phi = self.__step(delta=self.delta)
            random_u = np.random.random()
            if (self.sphere_center == self.pos).all() or random_u <= np.exp(
                -((np.linalg.norm(new_pos - self.sphere_center) / sigma) ** 2)
            ) / np.exp(
                -(
                    (
                        np.linalg.norm(
                            (self.pos - self.sphere_center)
                            * max(
                                0, 1 - 1 / np.linalg.norm(self.pos - self.sphere_center)
                            )
                        )
                        / sigma
                    )
                    ** 2
                )
            ):
                self.trajectory.append(new_pos)
                self.previous_theta = new_theta
                self.previous_phi = new_phi
                return self.pos

    def cluster_step(
        self, cluster_center: np.ndarray, cluster_sigma: int
    ) -> np.ndarray:
        """Creates a step in the trajectory, that is contained in a smaller cluster.

        With a certain probability we will enter a small cluster. These cluster generate a gaussian sphere (smaller clusters) with a new center and variance of how big or condensed points must be.
        Performs rejection sampling of points that do not follow the constrains. It as well takes in consideration if the angles are going in the good direction in order not to be rejected as a sampled point.

        Args:
            cluster_center: Array of the coordinates of the center of the cluster to which the points must be contained in
            cluster_sigma: integer representing the variance for the smaller cluster that define how condensed the points must be

        Returns:
            Array representing the new position under the gaussian small cluster constraints.
        """
        while True:
            new_pos, new_theta, new_phi = self.__step(delta=0.9)
            random_u = np.random.random()
            if (cluster_center == self.pos).all() or random_u <= np.exp(
                -((np.linalg.norm(new_pos - cluster_center) / cluster_sigma) ** 2)
            ) / np.exp(
                -(
                    (
                        np.linalg.norm(
                            (self.pos - cluster_center)
                            * max(0, 1 - 1 / np.linalg.norm(self.pos - cluster_center))
                        )
                        / cluster_sigma
                    )
                    ** 2
                )
            ):
                self.trajectory.append(new_pos)
                self.previous_theta = new_theta
                self.previous_phi = new_phi
                return self.pos


def generate_biological_structure(
    nb_nodes: int,
    delta: int,
    start_sigma: int,
    end_sigma: int,
    sigma: int,
    cluster_sigma: int,
    cluster_proba: int,
    step2: bool,
) -> np.ndarray:
    """Calls the stepping function in order to create the structure.

    Creates a structure with the according parameters. It as well creates smaller clusters with a certain probability.
    This function makes sure that there is an aging period before entering a new small cluster once we got out of one.

    Args:
        nb_nodes: integer for the amount of nodes the structures should have
        delta: integer representing how smooth the structure should be.
        start_sigma: integer for the step2 first sigma period
        end_sigma: integer for the step2 second sigma period
        sigma: integer representing how big the main cluster for the whole structure. Shows how condensed the structure should be
        cluster_sigma: integer representing the variance for the smaller cluster that defines how condensed the points must be
        cluster_proba: integer for the probability of obtaining a cluster
        step2: a boolean that help to decide if their should be a change in the structure at some point

    Returns:
        A normalised and centralise array for the coordinates of each point of the structure.
    """
    stepper = Stepper(delta)
    aging = 0
    while len(stepper.trajectory) <= nb_nodes - 1:
        if np.random.random() >= cluster_proba or aging >= 0:
            if len(stepper.trajectory) < 1000 and step2:
                stepper.smooth_step(start_sigma)
            elif len(stepper.trajectory) >= 1000 and step2:
                stepper.smooth_step(end_sigma)
            else:
                stepper.smooth_step(sigma)
            aging -= 1
        else:
            center = stepper.pos.copy()
            aging = 30
            for _ in range(20):
                stepper.cluster_step(center, cluster_sigma)

    return centralize_and_normalize_numpy(stepper.trajectory[:nb_nodes])


def synthetic_biological_data_generator(
    rng,
    trussart_hic: np.ndarray,
    n_structure: int,
    data_path: constants,
    nb_bins: int,
    delta: int,
    st_sig: int,
    end_sig: int,
    sig: int,
    clust_sig: int,
    clust_prob: int,
    secondstep: bool,
    seed: int,
    alpha: int,
    is_training: bool = True,
    icing: bool = True,
    minmaxuse: bool = False,
    transportation: bool = True,
    softmaxing: bool = False,
) -> None:
    """Function that creates the dataset of structures, distance matrix and HIC matrices.

    Args:
        rng: random state
        trussart_hic: None as not requires without optimal transport
        n_structure: integer for how many structures to creates in the dataset
        data_path: constant path to locate where the data should be saved outside the package
        nb_bins: integer for the number of points in the structure (HIC bins)
        delta: integer representing how smooth the function should be
        st_sig: integer for the step2 first sigma period
        end_sig: integer for the step2 second sigma period
        sig: integer representing how big the main cluster for the whole structure. Shows how condensed the structure should be
        clust_sig: integer representing the variance for the smaller cluster that defines how condensed the points must be
        clust_prob: integer for the probability of obtaining a cluster
        secondstep: a boolean that help to decide if their should be a change in the structure at some point
        seed: integer for the seed to be set
        alpha: integer representing the exponent to be used in the distance to hic matrices transformation
        is_training: boolean that tells the function whether to save date in training or testing folder
        icing: boolean that says if we are using ICE normalisation or not
        minmaxuse: boolean that says if we are using MinMax scaling or not
        transportation: boolean that says if we are unsing optimal transport or not
        softmaxing: boolean that says if we should use the softmax function over thwe HIC matrix.
    """
    digits_format = "{0:0=4d}"

    if is_training:
        nb_structures = n_structure
        file_name = "train"
    else:
        nb_structures = n_structure
        file_name = "test"

    # Create Dataset
    for i in tqdm(range(0, nb_structures)):

        path = generate_biological_structure(
            nb_bins, delta, st_sig, end_sig, sig, clust_sig, clust_prob, secondstep
        )

        path = centralize_and_normalize_numpy(path)

        # Structure matrix to file
        df = pd.DataFrame(data=path.astype(float))
        df.to_csv(
            data_path
            + file_name
            + "/structure_matrices/biological_structure_"
            + digits_format.format(i)
            + ".txt",
            sep=" ",
            header=False,
            index=False,
        )

        # Compute distance matrix
        precomputed_distances = distance_matrix(path, path)

        # Distance matrix to file
        df = pd.DataFrame(data=precomputed_distances.astype(float))
        df.to_csv(
            data_path
            + file_name
            + "/distance_matrices/biological_distance_"
            + digits_format.format(i)
            + ".txt",
            sep=" ",
            header=False,
            index=False,
        )

        # Compute HiC matrix
        hic_matrix = generate_hic(
            rng,
            path,
            trussart_hic,
            use_ice=icing,
            use_minmax=minmaxuse,
            use_ot=transportation,
            use_softmax=softmaxing,
            seed=seed,
            plot_optimal_transport=False,
            exponent=alpha,
        )

        # HiC matrix to file
        df = pd.DataFrame(data=hic_matrix.astype(float))
        df.to_csv(
            data_path
            + file_name
            + "/hic_matrices/biological_hic_"
            + digits_format.format(i)
            + ".txt",
            sep=" ",
            header=False,
            index=False,
        )


def synthetic_biological_data_generator_special_values(
    rng,
    trussart_hic: np.ndarray,
    n_structure: int,
    data_path: constants,
    nb_bins: int,
    delta_ch: np.ndarray,
    st_sig: int,
    end_sig: int,
    sig_ch: np.ndarray,
    clust_sig: int,
    clust_prob_ch: np.ndarray,
    secondstep: bool,
    seed: int,
    alpha: int,
    is_training: bool = True,
    icing: bool = True,
    minmaxuse: bool = False,
    transportation: bool = True,
    softmaxing: bool = False,
) -> None:
    """Function that creates the dataset of structures, distance matrix and HIC matrices.

    Args:
        rng: random state
        trussart_hic: None as not requires without optimal transport
        n_structure: integer for how many structures to creates in the dataset
        data_path: constant path to locate where the data should be saved outside the package
        nb_bins: integer for the number of points in the structure (HIC bins)
        delta_ch: array representing how smooth the function should be
        st_sig: integer for the step2 first sigma period
        end_sig: integer for the step2 second sigma period
        sig_ch: array representing how big the main cluster for the whole structure. Shows how condensed the structure should be
        clust_sig: integer representing the variance for the smaller cluster that defines how condensed the points must be
        clust_prob_ch: array for the probability of obtaining a cluster
        secondstep: a boolean that help to decide if their should be a change in the structure at some point
        seed: integer for the seed to be set
        alpha: integer representing the exponent to be used in the distance to hic matrices transformation
        is_training: boolean that tells the function whether to save date in training or testing folder
        icing: boolean that says if we are using ICE normalisation or not
        minmaxuse: boolean that says if we are using MinMax scaling or not
        transportation: boolean that says if we are unsing optimal transport or not
        softmaxing: boolean that says if we should use the softmax function over thwe HIC matrix.
    """
    digits_format = "{0:0=4d}"

    if is_training:
        nb_structures = n_structure
        file_name = "train"
    else:
        nb_structures = n_structure
        file_name = "test"

    for i in tqdm(range(0, nb_structures)):

        choice = random.choice([0, 1, 2, 3])

        path = generate_biological_structure(
            nb_bins,
            delta_ch[choice],
            st_sig,
            end_sig,
            sig_ch[choice],
            clust_sig,
            random.choice(clust_prob_ch),
            secondstep,
        )
        path = centralize_and_normalize_numpy(path)

        # Structure matrix to file
        df = pd.DataFrame(data=path.astype(float))
        df.to_csv(
            data_path
            + file_name
            + "/structure_matrices/biological_structure_special_values_"
            + digits_format.format(i)
            + ".txt",
            sep=" ",
            header=False,
            index=False,
        )

        # Compute distance matrix
        precomputed_distances = distance_matrix(path, path)

        # Distance matrix to file
        df = pd.DataFrame(data=precomputed_distances.astype(float))
        df.to_csv(
            data_path
            + file_name
            + "/distance_matrices/biological_distance_special_values_"
            + digits_format.format(i)
            + ".txt",
            sep=" ",
            header=False,
            index=False,
        )

        # Compute HiC matrix
        hic_matrix = generate_hic(
            rng,
            path,
            trussart_hic,
            use_ice=icing,
            use_minmax=minmaxuse,
            use_ot=transportation,
            use_softmax=softmaxing,
            seed=seed,
            plot_optimal_transport=False,
            exponent=alpha,
        )

        # HiC matrix to file
        df = pd.DataFrame(data=hic_matrix.astype(float))
        df.to_csv(
            data_path
            + file_name
            + "/hic_matrices/biological_hic_special_values_"
            + digits_format.format(i)
            + ".txt",
            sep=" ",
            header=False,
            index=False,
        )
