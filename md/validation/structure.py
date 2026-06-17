import numpy as np

from md.analysis import compute_rdf, compute_coordination_number


def find_local_maxima(y):
    """
    Return indices of simple local maxima.
    """
    y = np.asarray(y)
    return np.where((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]))[0] + 1


def test_fcc_rdf_peaks(
    positions_traj,
    box,
    lattice_constant,
    r_max=None,
    n_bins=200,
    n_peaks=3,
):
    """
    Validate FCC structure by comparing RDF peak locations to expected
    FCC neighbor-shell radii.

    Expected FCC shell locations:
        r1 = a / sqrt(2)
        r2 = a
        r3 = sqrt(3/2) a
    """

    box = np.asarray(box, dtype=float)

    if r_max is None:
        r_max = 0.45 * np.min(box)

    r, g_r = compute_rdf(
        positions_traj=positions_traj,
        box=box,
        r_max=r_max,
        n_bins=n_bins,
    )

    peak_indices = find_local_maxima(g_r)

    # ignore tiny noisy peaks near r = 0
    peak_indices = peak_indices[r[peak_indices] > 0.5]

    # sort peaks by height, then keep strongest few
    peak_indices = peak_indices[np.argsort(g_r[peak_indices])[::-1]]
    peak_indices = peak_indices[:n_peaks]

    # sort back by radius
    peak_indices = peak_indices[np.argsort(r[peak_indices])]

    measured_peaks = r[peak_indices]

    expected_peaks = np.array(
        [
            lattice_constant / np.sqrt(2.0),
            lattice_constant,
            np.sqrt(3.0 / 2.0) * lattice_constant,
        ]
    )

    m = min(len(measured_peaks), len(expected_peaks))

    abs_errors = np.abs(measured_peaks[:m] - expected_peaks[:m])
    rel_errors = abs_errors / expected_peaks[:m]

    return {
        "r": r,
        "g_r": g_r,
        "measured_peaks": measured_peaks,
        "expected_peaks": expected_peaks,
        "absolute_errors": abs_errors,
        "relative_errors": rel_errors,
        "max_relative_error": np.max(rel_errors) if len(rel_errors) > 0 else np.nan,
    }


def test_coordination_number(
    positions_traj,
    box,
    lattice_constant,
    n_bins=200,
    r_max=None,
):
    """
    Validate FCC nearest-neighbor coordination number.

    For ideal FCC:
        CN = 12

    We integrate the RDF up to a cutoff between the first and second shells.
    """

    box = np.asarray(box, dtype=float)
    N = positions_traj.shape[1]
    rho = N / np.prod(box)

    if r_max is None:
        first_shell = lattice_constant / np.sqrt(2.0)
        second_shell = lattice_constant
        r_max = 0.5 * (first_shell + second_shell)

    rdf_r_max = 0.45 * np.min(box)

    r, g_r = compute_rdf(
        positions_traj=positions_traj,
        box=box,
        r_max=rdf_r_max,
        n_bins=n_bins,
    )

    CN = compute_coordination_number(
        r=r,
        g_r=g_r,
        rho=rho,
        r_max=r_max,
    )

    expected_CN = 12.0
    abs_error = abs(CN - expected_CN)
    rel_error = abs_error / expected_CN

    return {
        "r": r,
        "g_r": g_r,
        "coordination_number": CN,
        "expected_coordination_number": expected_CN,
        "absolute_error": abs_error,
        "relative_error": rel_error,
        "coordination_cutoff": r_max,
    }