import numpy as np
from md.integrator import step_nve

def compute_msd(positions_traj, box):
    """
    Compute mean squared displacement MSD(t).
    MSD tells average distance displaced from start.

    positions_traj : array, shape (n_frames, N, 3)
        Trajectory of positions (Å).
    box : array-like, shape (3,)
        Box lengths (Å) for unwrapping with minimum image.

    Returns
    -------
    times : array, shape (n_frames,)
        Time indices (in units of your MD timestep, if you like).
    msd : array, shape (n_frames,)
        MSD(t) in Å^2.
    """
    pos = np.asarray(positions_traj)
    box = np.asarray(box, dtype=float)
    n_frames, N, _ = pos.shape

    # use frame 0 as reference
    r0 = pos[0]
    msd = np.zeros(n_frames, dtype=float)

    for t in range(n_frames):
        dr = pos[t] - r0
        dr -= box * np.round(dr / box)  # minimum image
        msd[t] = np.mean(np.sum(dr**2, axis=1))

    times = np.arange(n_frames, dtype=float)  # multiply by dt outside if you want
    return times, msd