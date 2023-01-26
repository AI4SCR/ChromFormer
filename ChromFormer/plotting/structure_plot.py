import numpy as np
import pandas
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_sphere_surface(x_0=0, y_0=0, z_0=0, radius=1):
    """Creates a sphere surface for the plotting"""
    x, y, z = create_sphere_coordinates(x_0, y_0, z_0, radius)
    return go.Surface(x=x, y=y, z=z, opacity=0.1)


def create_sphere_coordinates(x_0=0, y_0=0, z_0=0, radius=1):
    """Generate the sphere coordinates for the plotting

    Args:
        x: integer
        y: integer
        z: integer

    Return:
        x: x integer coordinate
        y: y integer coordinate
        z: z integer coordinate
    """
    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi, 100)

    x = radius * np.outer(np.cos(theta), np.sin(phi)) + x_0
    y = radius * np.outer(np.sin(theta), np.sin(phi)) + y_0
    z = radius * np.outer(np.ones(100), np.cos(phi)) + z_0

    return x, y, z


def structure_in_sphere(synthetic_biological_structure: np.ndarray) -> go.Figure:
    """Plots a structure in a spherical surface

    Args:
        synthetic_biological_structure: array of the structure to plot in a spherical surface
    """
    unit_sphere_surface = create_sphere_surface()
    synthetic_biological_structure_scatter = go.Scatter3d(
        x=synthetic_biological_structure[:, 0],
        y=synthetic_biological_structure[:, 1],
        z=synthetic_biological_structure[:, 2],
        marker=dict(
            size=4,
            color=np.asarray(range(len(synthetic_biological_structure[:, 0]))),
            colorscale="Viridis",
        ),
        line=dict(color="darkblue", width=2),
    )
    layout = go.Layout(
        width=900,
        height=900,
    )
    fig = go.Figure(
        data=[unit_sphere_surface, synthetic_biological_structure_scatter],
        layout=layout,
    )
    return fig


def structures_in_sphere_multiview(synthetic_biological_structures: list,
                                   cameras: list = ({'eye': dict(x=1, y=1, z=0.5)}, {'eye': dict(x=-1, y=1, z=0.5)}),
                                   subplot_titles=None) \
        -> go.Figure:
    """Plots a structure in a spherical surface

    Args:
        synthetic_biological_structure: array of the structure to plot in a spherical surface
    """

    from plotly.subplots import make_subplots
    subplot_titles = tuple(str(camera) for camera in cameras) if subplot_titles is True else subplot_titles
    fig = make_subplots(rows=len(synthetic_biological_structures), cols=len(cameras),
                        specs=[[{'is_3d': True}] * len(cameras)] * len(synthetic_biological_structures),
                        subplot_titles=subplot_titles)

    for idx_structure, synthetic_biological_structure in enumerate(synthetic_biological_structures):
        unit_sphere_surface = create_sphere_surface()
        synthetic_biological_structure_scatter = go.Scatter3d(
            x=synthetic_biological_structure[:, 0],
            y=synthetic_biological_structure[:, 1],
            z=synthetic_biological_structure[:, 2],
            marker=dict(
                size=4,
                color=np.asarray(range(len(synthetic_biological_structure[:, 0]))),
                colorscale="Viridis",
            ),
            line=dict(color="darkblue", width=2),
        )

        for idx_camera, camera in enumerate(cameras):
            fig.add_trace(unit_sphere_surface, row=idx_structure + 1, col=idx_camera + 1)
            fig.add_trace(synthetic_biological_structure_scatter, row=idx_structure + 1, col=idx_camera + 1)
            scene = getattr(fig.layout, f'scene{idx_structure * len(cameras) + idx_camera + 1}')
            scene.camera = camera

    fig.update_traces(showlegend=False)
    fig.update(layout_showlegend=False)
    fig.update(layout_coloraxis_showscale=False)
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(height=500 * len(synthetic_biological_structures), width=500 * len(cameras))
    return fig


def structures_in_sphere(synthetic_biological_structures: list,
                         param_grid_df: pandas.DataFrame = None,
                         ncols: int = 3,
                         camera=None) -> go.Figure:
    """Plots a structure in a spherical surface

    Args:
        synthetic_biological_structures: list of the structures to plot in a spherical surface
    """

    if camera is None:
        camera = {'eye': dict(x=1, y=1, z=0.5)}

    subplot_titles = []
    for row in param_grid_df.T.to_dict().values():
        subtitle = ''
        for col, val in row.items():
            subtitle = subtitle + f'{col}:{val} '
        subplot_titles.append(subtitle.strip())

    nrows = int(np.ceil(len(synthetic_biological_structures) / ncols))
    fig = make_subplots(rows=nrows, cols=ncols,
                        specs=[[{'is_3d': True}] * ncols] * nrows,
                        subplot_titles=subplot_titles)
    idx = 0
    for row in range(nrows):
        for col in range(ncols):
            if idx > len(synthetic_biological_structures):
                break
            synthetic_biological_structure = synthetic_biological_structures[idx]
            unit_sphere_surface = create_sphere_surface()
            synthetic_biological_structure_scatter = go.Scatter3d(
                x=synthetic_biological_structure[:, 0],
                y=synthetic_biological_structure[:, 1],
                z=synthetic_biological_structure[:, 2],
                marker=dict(
                    size=4,
                    color=np.asarray(range(len(synthetic_biological_structure[:, 0]))),
                    colorscale="Viridis",
                ),
                line=dict(color="darkblue", width=2)
            )

            fig.add_trace(unit_sphere_surface, row=row + 1, col=col + 1)
            fig.add_trace(synthetic_biological_structure_scatter, row=row + 1, col=col + 1)
            scene = getattr(fig.layout, f'scene{idx+1}')
            scene.camera = camera
            idx += 1

    fig.update_traces(showlegend=False)
    fig.update(layout_showlegend=False)
    fig.update(layout_coloraxis_showscale=False)
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(height=400 * nrows, width=400 * ncols)
    return fig



