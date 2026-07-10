import numpy as np
import pytest

from md.analysis.msd import compute_msd
from md.analysis.structure_factor import compute_structure_factor
from md.analysis.thermodynamics import compute_heat_capacity_from_energy
from md.analysis.transport import compute_diffusion_from_msd, compute_diffusion_from_vacf
from md.analysis.vacf import compute_vacf


def test_stationary_trajectory_has_zero_msd():
    trajectory = np.zeros((5, 3, 3))
    times, msd = compute_msd(trajectory, box=[10.0] * 3)
    np.testing.assert_array_equal(times, np.arange(5, dtype=float))
    np.testing.assert_allclose(msd, 0.0)


def test_msd_uses_periodic_minimum_image():
    trajectory = np.array([[[9.9, 0.0, 0.0]], [[0.1, 0.0, 0.0]]])
    _, msd = compute_msd(trajectory, box=[10.0] * 3)
    assert msd[1] == pytest.approx(0.2**2)


def test_constant_relative_velocities_have_constant_vacf():
    frame = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    velocities = np.repeat(frame[None, :, :], 4, axis=0)
    _, vacf = compute_vacf(velocities)
    np.testing.assert_allclose(vacf, 1.0)


def test_diffusion_from_linear_msd():
    times = np.linspace(0.0, 10.0, 20)
    expected_D = 0.25
    msd = 6.0 * expected_D * times + 2.0
    D, slope = compute_diffusion_from_msd(times, msd, t_min=0.0, t_max=10.0)

    assert slope == pytest.approx(6.0 * expected_D)
    assert D == pytest.approx(expected_D)


def test_diffusion_from_constant_vacf():
    times = np.array([0.0, 1.0, 2.0])
    vacf = np.full(3, 3.0)
    D, integral = compute_diffusion_from_vacf(times, vacf)

    assert integral == pytest.approx(6.0)
    assert D == pytest.approx(2.0)


def test_constant_energy_has_zero_heat_capacity():
    assert compute_heat_capacity_from_energy([5.0, 5.0, 5.0], T=300.0) == pytest.approx(0.0)


def test_ideal_g_of_r_gives_structure_factor_one():
    r = np.linspace(0.1, 5.0, 100)
    k = np.array([0.0, 1.0, 2.0])
    _, structure_factor = compute_structure_factor(k, r, np.ones_like(r), rho=0.1)
    np.testing.assert_allclose(structure_factor, 1.0)
