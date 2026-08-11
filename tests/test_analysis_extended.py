"""Additional deterministic tests for scientific-analysis helpers."""

from types import SimpleNamespace

import numpy as np
import pytest

from md.analysis.rdf import compute_coordination_number, compute_rdf
from md.analysis.structure_factor import compute_structure_factor
from md.analysis.thermodynamics import compute_heat_capacity_from_energy, compute_pressure, kB
from md.analysis.transport import (
    compute_diffusion_from_msd,
    compute_diffusion_from_vacf,
    compute_thermal_conductivity,
    compute_viscosity,
)
from md.analysis.vacf import compute_vacf


def test_rdf_returns_requested_number_of_bins():
    trajectory = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    r, g_r = compute_rdf(trajectory, box=[10.0] * 3, r_max=2.0, n_bins=20)
    assert r.shape == (20,)
    assert g_r.shape == (20,)
    assert np.all(np.isfinite(g_r))


def test_rdf_counts_a_pair_in_the_correct_shell():
    trajectory = np.array([[[0.0, 0.0, 0.0], [1.05, 0.0, 0.0]]])
    r, g_r = compute_rdf(trajectory, box=[10.0] * 3, r_max=2.0, n_bins=20)
    peak = np.argmax(g_r)
    assert r[peak] == pytest.approx(1.05, abs=0.051)
    assert g_r[peak] > 0.0


def test_rdf_uses_periodic_minimum_image():
    trajectory = np.array([[[0.1, 0.0, 0.0], [9.9, 0.0, 0.0]]])
    r, g_r = compute_rdf(trajectory, box=[10.0] * 3, r_max=1.0, n_bins=20)
    assert r[np.argmax(g_r)] == pytest.approx(0.2, abs=0.026)


def test_coordination_number_for_constant_gr_matches_analytic_integral():
    r = np.linspace(0.0, 2.0, 2001)
    rho = 0.25
    cn = compute_coordination_number(r, np.ones_like(r), rho=rho, r_max=2.0)
    expected = 4.0 * np.pi * rho * 2.0**3 / 3.0
    assert cn == pytest.approx(expected, rel=2e-4)


def test_coordination_number_ignores_radii_above_cutoff():
    r = np.linspace(0.0, 4.0, 4001)
    g_r = np.ones_like(r)
    cn = compute_coordination_number(r, g_r, rho=1.0, r_max=1.0)
    assert cn == pytest.approx(4.0 * np.pi / 3.0, rel=5e-4)


def test_structure_factor_zero_wavevector_limit():
    r = np.linspace(0.1, 2.0, 1000)
    g_r = np.full_like(r, 2.0)
    rho = 0.1
    _, s_k = compute_structure_factor([0.0], r, g_r, rho)
    expected = 1.0 + 4.0 * np.pi * rho * np.trapezoid(r**2, r)
    assert s_k[0] == pytest.approx(expected)


def test_structure_factor_preserves_k_values():
    k = np.array([0.0, 0.5, 1.5])
    returned_k, s_k = compute_structure_factor(k, [0.5, 1.0], [1.0, 1.0], rho=0.2)
    np.testing.assert_array_equal(returned_k, k)
    np.testing.assert_allclose(s_k, 1.0)


def test_pressure_for_force_free_particles_is_ideal_kinetic_term():
    system = SimpleNamespace(
        pos=np.zeros((2, 3)),
        force=np.zeros((2, 3)),
        kinetic_energy=0.0,
        virial=0.0,
        volume=lambda: 8.0,
    )

    system.update_kinetic_energy = lambda: setattr(
        system,
        "kinetic_energy",
        12.0,
    )

    assert compute_pressure(system) == pytest.approx(1.0)


def test_pressure_includes_virial_term():
    system = SimpleNamespace(
        kinetic_energy=0.0,
        virial=3.0,
        volume=lambda: 3.0,
    )

    system.update_kinetic_energy = lambda: setattr(
        system,
        "kinetic_energy",
        0.0,
    )

    assert compute_pressure(system) == pytest.approx(1.0 / 3.0)


def test_heat_capacity_matches_population_variance_formula():
    energies = np.array([1.0, 2.0, 3.0])
    temperature = 250.0
    expected = np.var(energies) / (kB * temperature**2)
    assert compute_heat_capacity_from_energy(energies, temperature) == pytest.approx(expected)


def test_viscosity_integrates_constant_correlation():
    times = np.array([0.0, 1.0, 2.0])
    eta, integral = compute_viscosity(times, np.full(3, 2.0), V=5.0, T=100.0)
    assert integral == pytest.approx(4.0)
    assert eta == pytest.approx(5.0 * 4.0 / (kB * 100.0))


def test_thermal_conductivity_integrates_constant_correlation():
    times = np.array([0.0, 1.0, 2.0])
    kappa, integral = compute_thermal_conductivity(times, np.full(3, 2.0), V=5.0, T=100.0)
    assert integral == pytest.approx(4.0)
    assert kappa == pytest.approx(4.0 / (kB * 100.0**2 * 5.0))


def test_diffusion_from_vacf_respects_tmax():
    times = np.arange(5.0)
    vacf = np.ones(5)
    diffusion, integral = compute_diffusion_from_vacf(times, vacf, t_max=2.0)
    assert integral == pytest.approx(2.0)
    assert diffusion == pytest.approx(2.0 / 3.0)


def test_diffusion_from_msd_default_window_recovers_linear_slope():
    times = np.arange(12.0)
    msd = 3.0 + 1.2 * times
    diffusion, slope = compute_diffusion_from_msd(times, msd)
    assert slope == pytest.approx(1.2)
    assert diffusion == pytest.approx(0.2)


def test_vacf_removes_uniform_center_of_mass_motion():
    velocities = np.ones((4, 3, 3))
    _, vacf = compute_vacf(velocities)
    np.testing.assert_allclose(vacf, 0.0, atol=1e-15)


def test_vacf_times_are_frame_indices():
    velocities = np.zeros((6, 2, 3))
    times, _ = compute_vacf(velocities)
    np.testing.assert_array_equal(times, np.arange(6, dtype=float))
