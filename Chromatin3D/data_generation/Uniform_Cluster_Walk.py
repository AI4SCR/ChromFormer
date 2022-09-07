"""
This script has the necessary functions that creates 3D clustered random walk structures and generates data 
"""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2022
ALL RIGHTS RESERVED
"""

from asyncio import constants
import numpy as np
from numpy import linalg as LAnumpy
import random
from tqdm import tqdm
from ..Data_Tools.Normalisation import centralize_and_normalize_numpy
from scipy.spatial import distance_matrix
from ..Data_Tools.Data_Calculation import (
    compute_hic_matrix,
    generate_hic,
)
import pandas as pd
import os
from typing import Tuple


class Stepper(object):
    """Function that generates the trajectory and the next step to take for the Cluster Walk """
    def __init__(s, delta: int) -> None:
        s.delta = delta
        s.previous_theta = np.random.uniform(0, np.pi)
        s.previous_phi = np.random.uniform(0, 2 * np.pi)
        s.trajectory = [np.zeros(3)]
        s.sphere_center = np.zeros(3)
        while True:
            random_x_y_z = np.random.uniform(-1, 1, size=3)
            if LAnumpy.norm(random_x_y_z, ord=2) <= 1:
                break
        s.prev_vector = random_x_y_z / LAnumpy.norm(random_x_y_z, ord=2)

    @property
    def pos(s) -> np.ndarray:
        """function that gets the last position in the trajectory of the random walk

        Returns:
            A numpy array 3D coordinates of the latest position 
        """
        return s.trajectory[-1]

    def __stepU(s, delta: int) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform step that uses the sphere in the cube trick, to generate a uniform vector that depends on the previous vector direction.
         It as well keeps distance of point i and i+1 similar for all i 
        Args:
            delta: delta of value between 0 and 1 that represents how much the next direction depends on the previous direction
        Returns:
            The new position according to the selected vector
            The generated directional vector
        """
        while True:
            random_x_y_z = np.random.uniform(-1, 1, size=3)
            if LAnumpy.norm(random_x_y_z, ord=2) <= 1:
                break
        random_x_y_z = random_x_y_z / LAnumpy.norm(random_x_y_z, ord=2)
        X_vector = (1 - delta) * s.prev_vector + delta * random_x_y_z
        X_vector = X_vector / LAnumpy.norm(X_vector, ord=2)
        pos = s.pos.copy()
        pos += X_vector
        return pos, X_vector

    def __step(s, delta: int) -> Tuple[np.ndarray, int, int]:
        """Step that uses the phi and delta angles to find the next 3D coordinates. Angles each depend to a certain degree of the previous angle.
        Args:
            delta: delta of value between 0 and 1 that represents how much the next angle depends on the previous angle
        Returns:
            The new position according to the selected angles
            The generated phi and theta angles
        """
        random_theta = np.random.uniform(0, np.pi)
        random_phi = np.random.uniform(0, 2 * np.pi)
        # partial_theta = np.random.choice([np.sign(partial_theta),-np.sign(previous_theta)],
        # p=[delta, 1-delta])
        new_theta = (1 - delta) * s.previous_theta + delta * random_theta
        new_phi = (1 - delta) * s.previous_phi + delta * random_phi

        pos = s.pos.copy()
        pos[0] += np.cos(new_phi) * np.sin(new_theta)
        pos[1] += np.sin(new_phi) * np.sin(new_theta)
        pos[2] += np.cos(new_theta)

        return pos, new_theta, new_phi

    def step(s) -> np.ndarray:
        """Function that gets the new position
        
        Returns:
            The latest position
        """
        new_pos, new_theta, new_phi = s.__step(delta=1)
        s.trajectory.append(new_pos)
        s.previous_theta = new_theta
        s.previous_phi = new_phi
        return s.pos

    def smooth_step(s, sigma: int) -> np.ndarray:
        """Function that creates a smooth step by making sure that the step is contained in the main Gaussian structure sphere (main cluster) that we generated at the beginning with a certain variance
        which specifies how condensed each points must be. Performs rejection sampling of points that do not follow the constrains. It as well takes in consideration if the vector is going in the good 
        direction in order not to be rejected as a sampled point. 

        Args:
            sigma: variance of the sphere
        Returns:
            The new position under the gaussian constraints.
        """
        while True:
            # new_pos, new_theta, new_phi = s.__step(delta=s.delta)
            new_pos, new_X_vector = s.__stepU(delta=s.delta)
            random_u = np.random.random()
            # if random_u <= np.exp(-(np.linalg.norm(new_pos - s.sphere_center) / sigma)**2):
            if (s.sphere_center == s.pos).all() or random_u <= np.exp(
                -((np.linalg.norm(new_pos - s.sphere_center) / sigma) ** 2)
            ) / np.exp(
                -(
                    (
                        np.linalg.norm(
                            (s.pos - s.sphere_center)
                            * max(0, 1 - 1 / np.linalg.norm(s.pos - s.sphere_center))
                        )
                        / sigma
                    )
                    ** 2
                )
            ):
                # print("Accepted")
                s.trajectory.append(new_pos)
                # s.previous_theta = new_theta
                # s.previous_phi = new_phi
                s.prev_vector = new_X_vector
                return s.pos
        # s.trajectory.append(s.__step(delta=s.delta))
        # return s.pos

    def cluster_step(s, cluster_center: np.ndarray, cluster_sigma: int) -> np.ndarray:
        """With a certain probability we will enter a small cluster. These cluster generate a gaussian sphere (smaller clusters) with a new center and variance of how big or condensed points must be. 
        Performs rejection sampling of points that do not follow the constrains. It as well takes in consideration if the vector is going in the good direction in order not to be rejected as a sampled point. 
        Args:
            cluster_center: center of the cluster to which the point must be contained in
            cluster_sigma: variance for the smaller cluster that define how condensed the points must be
        Returns:
            The new position under the gaussian small cluster constraints. 
        """
        while True:
            # new_pos, new_theta, new_phi = s.__step(delta=0.9)
            new_pos, new_X_vector = s.__stepU(delta=0.5)
            # print(f"Trying {new_pos} from {cluster_center}, proba {np.exp(-(np.linalg.norm(new_pos - cluster_center) / cluster_sigma)**2)}")
            random_u = np.random.random()
            if (cluster_center == s.pos).all() or random_u <= np.exp(
                -((np.linalg.norm(new_pos - cluster_center) / cluster_sigma) ** 2)
            ) / np.exp(
                -(
                    (
                        np.linalg.norm(
                            (s.pos - cluster_center)
                            * max(0, 1 - 1 / np.linalg.norm(s.pos - cluster_center))
                        )
                        / cluster_sigma
                    )
                    ** 2
                )
            ):
                # print("Accepted")
                s.trajectory.append(new_pos)
                # s.previous_theta = new_theta
                # s.previous_phi = new_phi
                s.prev_vector = new_X_vector

                return s.pos


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
    """Calls the stepping function in order to create the structure with the according parametres. It as well creates smaller clusters with a certain probability. 
    This function makes sure that there is an aging period before entering a new small cluster once we got out of one. 
    
    Args:
        nb_nodes: integer for the amount of nodes the structures should have 
        cluster_sigma: variance for the smaller cluster that define how condensed the points must be
    
    Returns:
        The new position under the gaussian small cluster constraints. 
    """
    stepper = Stepper(delta)
    aging = 0
    # cluster_sigma = np.random.uniform(1.7, 4.5)
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
            # aging = np.random.randint(10,25)
            aging = 30
            # cluster_size = np.random.randint(15,25)
            # cluster_sigma = np.random.uniform(2, 3)
            for _ in range(30):
                stepper.cluster_step(center, cluster_sigma)

    return centralize_and_normalize_numpy(stepper.trajectory[:nb_nodes])


def synthetic_biological_uniform_data_generator(
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

    # Set the seeds

    DIGITS_FORMAT = "{0:0=4d}"
    # rng = np.random.RandomState(seed)
    # random.seed(seed)
    # np.random.seed(seed)
    if is_training:
        nb_structures = n_structure
        file_name = "train"
    else:
        nb_structures = n_structure
        file_name = "test"

    # Create Dataset (iterate with step 2 because of data augmentation)
    for i in tqdm(range(0, nb_structures)):

        # Set the seeds

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
            + DIGITS_FORMAT.format(i)
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
            + DIGITS_FORMAT.format(i)
            + ".txt",
            sep=" ",
            header=False,
            index=False,
        )

        # Compute HiC matrix + ICE Normalisation
        #####hic_matrix = compute_hic_matrix(precomputed_distances, EXPONENT)
        #####z_score = ICE(hic_matrix)

        # Min Max scaling
        #####scaler = MinMaxScaler()
        #####hic_matrix = scaler.fit_transform(z_score)
        ## OT preparation
        #####Xs, Xt, X1, X2 = ot_data(z_score, trussart_hic, rng)

        ######Xs, Xt, X1, X2 = ot_data(hic_matrix, trussart_hic, rng)
        ######hic_matrix, I2te = transport(Xs, Xt, X1, X2)
        # HiC matrix to file

        ########need to fetch trussart
        hic_matrix = generate_hic(
            rng,
            path,
            trussart_hic,
            use_ice=True,
            use_minmax=False,
            use_ot=True,
            use_softmax=False,
            seed=seed,
            plot_optimal_transport=False,
            exponent=alpha,
        )
        df = pd.DataFrame(data=hic_matrix.astype(float))
        df.to_csv(
            data_path
            + file_name
            + "/hic_matrices/biological_hic_"
            + DIGITS_FORMAT.format(i)
            + ".txt",
            sep=" ",
            header=False,
            index=False,
        )
