import numpy as np


def make_fcc_lattice(a, nx, ny, nz):
# Build an FCC lattice with lattice constant a (Å)
# Return: positions, box



    basis = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.5, 0.5],
    [0.5, 0.0, 0.5],
    [0.5, 0.5, 0.0]
    ])


    positions = []


    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                origin = a * np.array([i, j, k], dtype=float)
                for b in basis:
                    positions.append(origin + a * b)


    positions = np.array(positions)
    box = np.array([nx * a, ny * a, nz * a])
    return positions, box