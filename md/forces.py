import numpy as np


def lj_forces(
    positions,
    box,
    pairs,
    epsilon=1.0,
    sigma=1.0,
    rcut=2.5,
):
    pos = np.asarray(positions)
    box = np.asarray(box, dtype=float)

    forces = np.zeros_like(pos)
    potential_energy = 0.0
    virial = 0.0

    rcut2 = rcut * rcut

    for i, j in pairs:
        rij = pos[i] - pos[j]
        rij -= box * np.round(rij / box)

        r2 = np.dot(rij, rij)

        if 0.0 < r2 < rcut2:
            inv_r2 = (sigma * sigma) / r2
            inv_r6 = inv_r2**3
            inv_r12 = inv_r6**2

            vij = 4.0 * epsilon * (inv_r12 - inv_r6)
            potential_energy += vij

            fij_over_r2 = (
                24.0
                * epsilon
                * (2.0 * inv_r12 - inv_r6)
                / r2
            )
            fij = fij_over_r2 * rij

            forces[i] += fij
            forces[j] -= fij

            virial += np.dot(rij, fij)

    return forces, potential_energy, virial