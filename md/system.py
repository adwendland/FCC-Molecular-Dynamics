# md/system.py
import numpy as np
import copy
from md.neighborlist import NeighborList

kB = 8.617333262145e-5  # eV/K

class System:
    """
    Represents the MD simulation state.
    Holds:
        - positions
        - velocities
        - forces
        - mass
        - simulation box
        - neighbor list

    Coordinates are in internal MD units.
    """

    def __init__(self, positions, mass, box, symbol="X", skin=0.3, cutoff=2.5):
        self.N = positions.shape[0]
        self.mass = mass
        self.inv_mass = 1.0 / mass
        self.symbol = symbol

        # Initialize positions, velocities, forces
        self.pos = positions.copy()
        self.vel = np.zeros_like(positions)
        self.force = np.zeros_like(positions)

        # Intialize energies
        self.kinetic_energy = 0.0
        self.potential_energy = 0.0
        self.total_energy = 0.0     

        # Simulation box
        self.box = np.array(box, dtype=float)

        # Build neighbor list
        self.cutoff = cutoff
        self.skin = skin
        self.nl = NeighborList(
            cutoff=cutoff,
            skin=skin,
            positions=self.pos,
            box=self.box
        )

        # Track when to rebuild
        self.displacement = np.zeros(self.N)

    def copy(self):
        return copy.deepcopy(self)

    # --- Boundary conditions ---
    def apply_pbc(self):
        """Wrap atoms back into the simulation box."""
        self.pos = self.pos % self.box # Minimum image

    def volume(self):
        """
        Return simulation box volume.

        Assumes an orthorhombic box (diagonal lengths in self.box).
        Units: Å^3
        """
        return float(np.prod(self.box))


    # --- Force evaluation wrapper ---
    def compute_forces(self, force_fn):
        """
        Compute forces using a force function (e.g., lj_forces).
        force_fn must have the signature:

            force, potential_energy = force_fn(pos, box, pairs)

        where pairs is the neighbor list pair array.
        """
        # Update neighbor list
        self.nl.update(self.pos)
        pairs = self.nl.pairs

        # Update forces
        forces, potential_energy = force_fn(self.pos, self.box, pairs)

        self.force[:, :] = forces
        self.potential_energy = float(potential_energy)

        return self.potential_energy
        
    # --- Kinetic energy ---
    # K = (1/2) m v^2   (total KE in eV)
    def update_kinetic_energy(self):
        if np.isscalar(self.mass):
            self.kinetic_energy = 0.5 * self.mass * np.sum(self.vel**2)
        else:
            m = self.mass.reshape(-1, 1)
            self.kinetic_energy = 0.5 * np.sum(m * self.vel**2)

        return self.kinetic_energy

    def update_energies(self):
        """Update kinetic and total energy for scalar or per-particle masses."""
        self.update_kinetic_energy()
        self.total_energy = self.kinetic_energy + self.potential_energy
        return self.total_energy


    # --- Temperature ---
    # KE = (dof/2) * k_B * T,  dof = 3N - 3
    def temperature(self):
        """Return instantaneous temperature in K."""
        dof = 3 * self.N - 3
        if dof <= 0:
            return 0.0

        KE = self.update_kinetic_energy()
        return (2.0 * KE) / (dof * kB)

    # --- Momentum removal ---
    # Removes net drift if developed accidentally
    def remove_drift(self):
        """Remove center of mass drift and update kinetic/total energy."""
        vcm = np.mean(self.vel, axis=0)
        self.vel -= vcm
        self.update_energies()
