import numpy as np
import matplotlib.pyplot as plt
from Chromatin3D.Data_Tools.Data_Calculation import create_sphere_surface
import plotly.graph_objects as go


def plot_structure_in_sphere(synthetic_biological_structure: np.ndarray) -> None:

    unit_sphere_surface = create_sphere_surface()
    synthetic_biological_structure_scatter = go.Scatter3d(x=synthetic_biological_structure[:,0], 
                                                          y=synthetic_biological_structure[:,1], 
                                                          z=synthetic_biological_structure[:,2],
        marker=dict(size=4, color=np.asarray(range(len(synthetic_biological_structure[:,0]))), colorscale='Viridis'),
        line=dict(color='darkblue', width=2))
    layout = go.Layout(width=900, height=900,)
    fig = go.Figure(data=[unit_sphere_surface, synthetic_biological_structure_scatter], layout=layout)

    fig.show()

def plot_hic(hic: np.ndarray) -> None:
    fig, axs = plt.subplots(1, 1, figsize=(10,10))

    axs.imshow(hic, cmap="hot", interpolation='nearest')
    axs.tick_params(axis='both', which='major', labelsize=30, width=4)

    #plt.savefig('synthetic_biological_hic_example.png')
    plt.show()

def plot_optimal_transport(Xs: np.ndarray, Xt: np.ndarray, I1te: np.ndarray) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(15,5))
    counts, bins = np.histogram(Xs, bins=30)#source
    axs[0].hist(bins[:-1], bins, weights=counts)
    axs[0].set_title('Source Histogram', fontstyle='italic')
    counts, bins = np.histogram(Xt, bins=30)#target
    axs[1].hist(bins[:-1], bins, weights=counts)
    axs[1].set_title('Target Histogram', fontstyle='italic')
    counts, bins = np.histogram(I1te, bins=30)#transported source
    axs[2].hist(bins[:-1], bins, weights=counts)
    axs[2].set_title('Transported source Histogram', fontstyle='italic')