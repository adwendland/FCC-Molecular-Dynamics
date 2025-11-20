# md/neighborlist.py
import numpy as np


class NeighborList:
    """
    Simple Verlet (skin) neighbor list for MD simulations.
    Rebuilds only when atoms have moved more than half the skin distance.
    """

    def __init__(self, cutoff, skin, positions, box):
        self.cutoff = cutoff
        self.skin = skin
        self.cutoff_skin = cutoff + skin

        self.box = np.array(box, dtype=float)
        self.pos_old = positions.copy()

        self.N = positions.shape[0]
        self.pairs = np.empty((0, 2), dtype=np.int32)

        self._build(positions)

    # ------------------------------------------------------------
    # Distance with periodic boundary conditions
    # ------------------------------------------------------------
    def _pbc_diff(self, pos_i, pos_j):
        """Compute r_i - r_j with periodic boundary conditions."""
        dr = pos_i - pos_j
        dr -= np.round(dr / self.box) * self.box
        return dr

    # ------------------------------------------------------------
    # Build neighbor list from scratch
    # ------------------------------------------------------------
    def _build(self, pos):
        cutoff2 = self.cutoff_skin ** 2
        pairs = []

        for i in range(self.N):
            for j in range(i + 1, self.N):
                dr = self._pbc_diff(pos[i], pos[j])
                if dr.dot(dr) < cutoff2:
                    pairs.append((i, j))

        self.pairs = np.asarray(pairs, dtype=np.int32)
        self.pos_old[:] = pos.copy()

    # ------------------------------------------------------------
    # Check if rebuild is necessary
    # ------------------------------------------------------------
    def _needs_rebuild(self, pos):
        """Check if any particle moved more than half the skin distance."""
        dr = pos - self.pos_old
        dr -= np.round(dr / self.box) * self.box
        disp = np.sqrt(np.sum(dr * dr, axis=1))
        return np.any(disp > 0.5 * self.skin)

    # ------------------------------------------------------------
    # Public update function
    # ------------------------------------------------------------
    def update(self, positions):
        """
        Rebuild neighbor list if needed.
        """
        if self._needs_rebuild(positions):
            self._build(positions)
