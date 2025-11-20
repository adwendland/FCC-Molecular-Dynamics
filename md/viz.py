# ------------------------------------------------------------
# Browser-native 3D atom visualizer (Plotly)
# ------------------------------------------------------------
import plotly.graph_objects as go
import numpy as np

def visualize_atoms_3d(positions, box, symbol="X"):
    pos = np.asarray(positions)
    x, y, z = pos.T
    Lx, Ly, Lz = box

    fig = go.Figure()

    # Atoms
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(size=4, opacity=0.9),
            name=symbol,
        )
    )

    # Simulation box wireframe
    corners = np.array([
        [0, 0, 0],
        [Lx, 0, 0],
        [Lx, Ly, 0],
        [0, Ly, 0],
        [0, 0, Lz],
        [Lx, 0, Lz],
        [Lx, Ly, Lz],
        [0, Ly, Lz],
    ])

    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]

    for i, j in edges:
        fig.add_trace(
            go.Scatter3d(
                x=[corners[i,0], corners[j,0]],
                y=[corners[i,1], corners[j,1]],
                z=[corners[i,2], corners[j,2]],
                mode="lines",
                line=dict(width=2),
                showlegend=False,
            )
        )

    fig.update_layout(
        title=f"{symbol} Atom Configuration",
        scene=dict(
            xaxis=dict(range=[0, Lx]),
            yaxis=dict(range=[0, Ly]),
            zaxis=dict(range=[0, Lz]),
            aspectmode="data"
        ),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig