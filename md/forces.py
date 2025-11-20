import numpy as np


def lj_forces(positions, box, pairs, epsilon=1.0, sigma=1.0, rcut=2.5):
    pos = positions
    N = pos.shape[0]
    forces = np.zeros_like(pos)
    potential_energy = 0.0

    # Precompute constants
    rcut2 = rcut * rcut # rcut is cutoff radius
    box_inv = 1.0 / box  # for minimum image

    for (i, j) in pairs:
        # Compute interatomic distance
        # Minimum image displacement vector
        rij = pos[i] - pos[j]
        rij -= np.round(rij / box) * box
        r2 = np.dot(rij, rij)

        if r2 < rcut2:
            # Make LJ a little cheaper
            # inv_r6 = (sigma/r)^6, inv_r12 = (sigma/r)^12
            inv_r2 = (sigma * sigma) / r2
            inv_r6 = inv_r2 ** 3
            inv_r12 = inv_r6 ** 2

            # Compute LJ potential
            # U = 4eps * [(sigma/r)^12 - (sigma/r)^6]
            vij = 4.0 * epsilon * (inv_r12 - inv_r6)
            potential_energy += vij

            # Compute LJ forces
            # F = -\nabla_r U       
            fij_over_r2 = 24.0 * epsilon * (2.0 * inv_r12 - inv_r6) / r2
            fij = fij_over_r2 * rij

            # Newton's 3rd Law
            # Forces are equal and opposite
            forces[i] += fij
            forces[j] -= fij
    return forces, potential_energy
