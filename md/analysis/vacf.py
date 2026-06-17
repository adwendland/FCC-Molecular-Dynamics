import numpy as np
from md.integrator import step_nve

def compute_vacf(velocities_traj):
    """
    Compute velocity autocorrelation function VACF(t).
    Tells how the velocity of a particle at one point in time relates to another.

    velocities_traj : array, shape (n_frames, N, 3)
        Velocities (in internal units consistent with your forces & dt).

    Returns
    -------
    times : array, shape (n_frames,)
        Time indices (0, 1, 2, ...).
    vacf : array, shape (n_frames,)
        VACF(t) = < v(0) · v(t) > averaged over atoms and time origins.
    """
    v = np.asarray(velocities_traj)
    n_frames, N, _ = v.shape

    # subtract COM drift frame-by-frame (just in case)
    v = v - v.mean(axis=1, keepdims=True)

    vacf = np.zeros(n_frames, dtype=float)
    n_origins = np.zeros(n_frames, dtype=float)

    # time–origin averaging
    for t0 in range(n_frames):
        v0 = v[t0]
        max_tau = n_frames - t0
        dots = np.sum(v0 * v[t0:], axis=2)  # shape (max_tau, N)
        vacf[:max_tau] += dots.mean(axis=1)
        n_origins[:max_tau] += 1.0

    vacf /= np.maximum(n_origins, 1.0)
    times = np.arange(n_frames, dtype=float)
    return times, vacf