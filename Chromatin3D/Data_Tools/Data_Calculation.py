import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from scipy.spatial import distance_matrix
from Chromatin3D.Data_Tools.Normalisation import ICE
from Chromatin3D.Data_Tools.Optimal_Transport import ot_data, transport
from typing import Tuple

def import_trussart_data(path) -> Tuple[np.ndarray, np.ndarray]:

    TRUSSART_HIC_PATH = f'{path}/trussart/hic_matrices/150_TADlike_alpha_150_set0.mat'
    TRUSSART_STRUCTURES_PATH = f'{path}/trussart/structure_matrices/'
    trussart_hic = np.loadtxt(TRUSSART_HIC_PATH, dtype='f', delimiter='\t')
    scaler = MinMaxScaler()
    trussart_hic = scaler.fit_transform(trussart_hic)
    trussart_structures = []

    file_list = os.listdir(TRUSSART_STRUCTURES_PATH)
    file_list = filter(lambda f: f.endswith('.xyz'), file_list)

    for file_name in file_list:
        current_trussart_structure = np.loadtxt(TRUSSART_STRUCTURES_PATH + file_name, dtype='f', delimiter='\t')
        current_trussart_structure = current_trussart_structure[:,1:]
        trussart_structures.append(current_trussart_structure)
    
    return trussart_hic, trussart_structures

def compute_hic_matrix(distance_matrix: np.ndarray, alpha: int) -> np.ndarray:
        
    distance_matrix = np.where(distance_matrix == 0, np.inf, distance_matrix)
    
    hic_matrix = np.zeros((len(distance_matrix), len(distance_matrix)))
    ##-alpha
    ##-1/alpha
    hic_matrix = np.where(distance_matrix == np.inf, hic_matrix, np.power(distance_matrix, -1/alpha))
    
    return hic_matrix

def create_sphere_coordinates(x_0=0, y_0=0, z_0=0, radius=1):
    
    theta = np.linspace(0,2*np.pi,100)
    phi = np.linspace(0,np.pi,100)
    
    x = radius*np.outer(np.cos(theta), np.sin(phi)) + x_0
    y = radius*np.outer(np.sin(theta), np.sin(phi)) + y_0
    z = radius*np.outer(np.ones(100), np.cos(phi)) + z_0

    return x, y, z

def create_sphere_surface(x_0=0, y_0=0, z_0=0, radius=1):
    
    x, y, z = create_sphere_coordinates(x_0, y_0, z_0, radius)
    return go.Surface(x=x, y=y, z=z, opacity=0.1)

def generate_hic(rng, synthetic_biological_structure: np.ndarray, trussart_hic: np.ndarray, use_ice: bool =True, use_minmax: bool =False, use_ot: bool =True, use_softmax: bool =False, seed: int =42, plot_optimal_transport: bool =False, exponent: int =1):
    
    scaler = MinMaxScaler()
    precomputed_distances = distance_matrix(synthetic_biological_structure, synthetic_biological_structure)
    hic_matrix = compute_hic_matrix(precomputed_distances,exponent) 

    if use_ice:
        hic_matrix = ICE(hic_matrix)

    if use_minmax:
        hic_matrix = scaler.fit_transform(hic_matrix)

    if use_ot:
        orig_hic = hic_matrix
        #rng = np.random.RandomState(seed)
        Xs, Xt, X1, X2 = ot_data(hic_matrix, trussart_hic, rng)
        hic_matrix, I2te = transport(Xs, Xt, X1, X2, hic_matrix.shape, trussart_hic.shape)

    if use_softmax:
        hic_matrix = 1/(1 + np.exp(-8*hic_matrix+4))

    if plot_optimal_transport:
        return hic_matrix, orig_hic, Xs, Xt
    return hic_matrix