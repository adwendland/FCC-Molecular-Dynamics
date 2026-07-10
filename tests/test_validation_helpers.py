"""Fast unit tests for validation calculations using deterministic stand-ins."""

from types import SimpleNamespace

import numpy as np
import pytest

import md.validation.convergence as convergence
import md.validation.energy as energy
import md.validation.momentum as momentum
import md.validation.structure as structure
import md.validation.thermostat as thermostat


class FakeEnergySystem:
    def __init__(self):
        self.total_energy = 10.0
        self._count = 0

    def remove_drift(self):
        pass

    def update_energies(self):
        return self.total_energy


class FakeMomentumSystem:
    def __init__(self, mass=2.0):
        self.mass = mass
        self.vel = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])


class FakeThermostatSystem:
    def __init__(self, mass=2.0):
        self.mass = mass
        self.N = 2
        self.vel = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        self.kinetic_energy = 2.0
        self._temperature = 300.0

    def temperature(self):
        return self._temperature


def test_energy_validation_returns_expected_samples(monkeypatch):
    system = FakeEnergySystem()

    def fake_step(s, dt, **kwargs):
        if dt:
            s._count += 1
            s.total_energy = 10.0 + 0.01 * s._count

    monkeypatch.setattr(energy, "step_nve", fake_step)
    result = energy.test_relative_energy_drift(system, 0.5, 5, 1, 1, 2.5, sample_every=2)
    np.testing.assert_allclose(result["times"], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(result["energies"], [10.01, 10.03, 10.05])
    assert result["max_abs_rel"] == pytest.approx(0.005)
    assert result["drift_slope"] == pytest.approx(0.02)


def test_energy_validation_single_sample_has_zero_slope(monkeypatch):
    monkeypatch.setattr(energy, "step_nve", lambda *a, **k: None)
    result = energy.test_relative_energy_drift(FakeEnergySystem(), 1.0, 1, 1, 1, 2.5)
    assert result["drift_slope"] == 0.0
    assert result["max_abs_rel"] == 0.0


def test_total_momentum_scalar_mass():
    system = FakeMomentumSystem(mass=3.0)
    system.vel = np.array([[1.0, 2.0, 0.0], [2.0, -1.0, 1.0]])
    np.testing.assert_allclose(momentum.total_momentum(system), [9.0, 3.0, 3.0])


def test_total_momentum_particle_masses():
    system = FakeMomentumSystem(mass=np.array([2.0, 4.0]))
    system.vel = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    np.testing.assert_allclose(momentum.total_momentum(system), [2.0, 8.0, 0.0])


def test_momentum_scale_scalar_and_array_mass():
    scalar = FakeMomentumSystem(mass=2.0)
    array = FakeMomentumSystem(mass=np.array([2.0, 3.0]))
    assert momentum.momentum_scale(scalar) == pytest.approx(4.0)
    assert momentum.momentum_scale(array) == pytest.approx(5.0)


def test_momentum_validation_detects_known_drift(monkeypatch):
    system = FakeMomentumSystem(mass=2.0)

    def fake_step(s, dt, **kwargs):
        if dt:
            s.vel[0, 0] += 0.1

    monkeypatch.setattr(momentum, "step_nve", fake_step)
    result = momentum.test_momentum_conservation(system, 0.5, 3, 1, 1, 2.5)
    np.testing.assert_allclose(result["times"], [0.0, 0.5, 1.0])
    assert result["max_abs_drift"] == pytest.approx(0.6)
    assert result["max_rel_drift"] == pytest.approx(0.15)


def test_minimum_image_displacement():
    dx = np.array([[9.8, -9.8, 0.0]])
    corrected = convergence.minimum_image_displacement(dx, np.array([10.0] * 3))
    np.testing.assert_allclose(corrected, [[-0.2, 0.2, 0.0]])


def test_rms_position_error_uses_particle_mean_and_periodicity():
    pos = np.array([[9.9, 0.0, 0.0], [0.0, 0.0, 0.0]])
    ref = np.array([[0.1, 0.0, 0.0], [0.0, 0.0, 0.0]])
    error = convergence.rms_position_error(pos, ref, np.array([10.0] * 3))
    assert error == pytest.approx(np.sqrt(0.04 / 2.0))


def test_timestep_refinement_recovers_second_order(monkeypatch):
    class FakeSystem:
        def __init__(self):
            self.pos = np.zeros((1, 3))
            self.box = np.array([100.0] * 3)

    def fake_step(system, dt, **kwargs):
        # Accumulated global error at fixed final time is proportional to dt^2.
        system.pos[0, 0] += dt + dt**3

    monkeypatch.setattr(convergence, "step_nve", fake_step)
    result = convergence.test_timestep_refinement(FakeSystem, dt=0.4, n_steps=10, epsilon=1, sigma=1, rcut=2.5)
    assert result["order"] == pytest.approx(2.0, rel=0.15)
    assert result["reference_dt"] == pytest.approx(0.05)
    assert list(result["errors"]) == [0.4, 0.2, 0.1]


def test_find_local_maxima():
    np.testing.assert_array_equal(structure.find_local_maxima([0, 2, 1, 3, 2]), [1, 3])


def test_find_local_maxima_ignores_plateaus_and_endpoints():
    assert structure.find_local_maxima([3, 2, 2, 1]).size == 0


def test_fcc_peak_validation_with_mocked_rdf(monkeypatch):
    a = 4.0
    expected = np.array([a / np.sqrt(2), a, np.sqrt(1.5) * a])
    r = np.linspace(0.0, 6.0, 1201)
    g = np.zeros_like(r)
    for peak in expected:
        g[np.argmin(abs(r - peak))] = 10.0
    monkeypatch.setattr(structure, "compute_rdf", lambda **kwargs: (r, g))

    result = structure.test_fcc_rdf_peaks(np.zeros((1, 4, 3)), [20.0] * 3, a, n_peaks=3)
    np.testing.assert_allclose(result["measured_peaks"], expected, atol=0.006)
    assert result["max_relative_error"] < 0.002


def test_fcc_peak_validation_handles_no_peaks(monkeypatch):
    r = np.linspace(0.0, 4.0, 100)
    monkeypatch.setattr(structure, "compute_rdf", lambda **kwargs: (r, np.zeros_like(r)))
    result = structure.test_fcc_rdf_peaks(np.zeros((1, 4, 3)), [10.0] * 3, 3.5)
    assert result["measured_peaks"].size == 0
    assert np.isnan(result["max_relative_error"])


def test_coordination_validation_uses_density_and_expected_value(monkeypatch):
    captured = {}
    monkeypatch.setattr(structure, "compute_rdf", lambda **kwargs: (np.array([1.0]), np.array([2.0])))

    def fake_cn(r, g_r, rho, r_max):
        captured.update(rho=rho, r_max=r_max)
        return 11.5

    monkeypatch.setattr(structure, "compute_coordination_number", fake_cn)
    trajectory = np.zeros((2, 32, 3))
    result = structure.test_coordination_number(trajectory, [8.0] * 3, lattice_constant=4.0)
    assert captured["rho"] == pytest.approx(32 / 512)
    assert result["coordination_number"] == 11.5
    assert result["absolute_error"] == pytest.approx(0.5)
    assert result["relative_error"] == pytest.approx(0.5 / 12.0)


def test_temperature_stability_summarizes_samples(monkeypatch):
    system = FakeThermostatSystem()
    counter = {"n": 0}

    def fake_step(s, dt, **kwargs):
        if dt:
            counter["n"] += 1
            s._temperature = 290.0 + 10.0 * counter["n"]
            s.kinetic_energy = float(counter["n"])
            s.vel[:] = counter["n"]

    monkeypatch.setattr(thermostat, "step_nvt_berendsen", fake_step)
    result = thermostat.test_temperature_stability(system, 0.5, 5, 320.0, 2.0, 1, 1, 2.5, sample_every=2)
    np.testing.assert_allclose(result["times"], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(result["temperatures"], [300.0, 320.0, 340.0])
    assert result["mean_temperature"] == pytest.approx(320.0)
    assert result["relative_temperature_error"] == pytest.approx(0.0)
    assert result["velocities"].shape == (3, 2, 3)


def test_component_equipartition_for_balanced_velocities(monkeypatch):
    system = FakeThermostatSystem()
    velocities = np.array([
        [[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]],
        [[2.0, 2.0, 2.0], [-2.0, -2.0, -2.0]],
    ])
    fake_result = {
        "velocities": velocities,
        "mean_temperature": 300.0,
        "kinetic_energies": np.array([6.0, 24.0]),
    }
    monkeypatch.setattr(thermostat, "test_temperature_stability", lambda **kwargs: fake_result.copy())
    result = thermostat.test_component_equipartition(system, 1, 1, 300, 1, 1, 1, 2.5)
    assert result["mean_kinetic_x"] == pytest.approx(result["mean_kinetic_y"])
    assert result["mean_kinetic_y"] == pytest.approx(result["mean_kinetic_z"])
    assert result["relative_component_equipartition_error"] >= 0.0


def test_total_kinetic_energy_error_is_zero_for_consistent_data(monkeypatch):
    system = FakeThermostatSystem()
    expected = 1.5 * system.N * thermostat.kB * 300.0
    fake_result = {
        "mean_temperature": 300.0,
        "kinetic_energies": np.array([expected, expected]),
        "velocities": np.zeros((2, 2, 3)),
    }
    monkeypatch.setattr(thermostat, "test_temperature_stability", lambda **kwargs: fake_result.copy())
    result = thermostat.test_total_kinetic_energy(system, 1, 1, 300, 1, 1, 1, 2.5)
    assert result["relative_equipartition_error"] == pytest.approx(0.0)
