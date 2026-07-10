"""Small end-to-end scientific smoke tests.

These use real lattice, neighbor-list, force, integration, and validation code,
but deliberately tiny systems and short trajectories so normal CI stays fast.
"""

import numpy as np
import pytest

import md.integrator as integrator
from md.analysis.rdf import compute_rdf
from md.validation.energy import test_relative_energy_drift as run_energy_drift
from md.validation.momentum import test_momentum_conservation as run_momentum_check


@pytest.mark.scientific
def test_short_nve_run_has_finite_state(monkeypatch, small_ni_system, ni_parameters):
    monkeypatch.setattr(integrator, "_HAVE_CPP", False)
    p = ni_parameters
    for _ in range(5):
        integrator.step_nve(small_ni_system, 0.001, p["epsilon"], p["sigma"], p["rcut"])
    assert np.all(np.isfinite(small_ni_system.pos))
    assert np.all(np.isfinite(small_ni_system.vel))
    assert np.all(np.isfinite(small_ni_system.force))
    assert np.isfinite(small_ni_system.total_energy)


@pytest.mark.scientific
def test_short_nve_momentum_conservation(monkeypatch, small_ni_system, ni_parameters):
    monkeypatch.setattr(integrator, "_HAVE_CPP", False)
    p = ni_parameters
    result = run_momentum_check(
        small_ni_system, dt=0.001, n_steps=8,
        epsilon=p["epsilon"], sigma=p["sigma"], rcut=p["rcut"],
    )
    assert result["max_rel_drift"] < 1e-10


@pytest.mark.scientific
def test_short_nve_energy_drift_is_small(monkeypatch, small_ni_system, ni_parameters):
    monkeypatch.setattr(integrator, "_HAVE_CPP", False)
    p = ni_parameters
    result = run_energy_drift(
        small_ni_system, dt=0.0005, n_steps=8,
        epsilon=p["epsilon"], sigma=p["sigma"], rcut=p["rcut"],
    )
    assert result["max_abs_rel"] < 1e-6


@pytest.mark.scientific
def test_static_fcc_rdf_has_nearest_neighbor_signal(small_ni_system, ni_parameters):
    trajectory = small_ni_system.pos[None, :, :]
    r, g_r = compute_rdf(
        trajectory,
        small_ni_system.box,
        r_max=0.49 * np.min(small_ni_system.box),
        n_bins=300,
    )
    expected = ni_parameters["a"] / np.sqrt(2.0)
    window = np.abs(r - expected) < 0.05
    assert np.any(window)
    assert np.max(g_r[window]) > 0.0
