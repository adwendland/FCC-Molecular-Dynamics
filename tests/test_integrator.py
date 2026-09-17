import numpy as np
import pytest

import md.integrator as integrator
from md.system import System
from md.analysis.analysis_driver import build_system
from md.forces import lj_forces

def require_cpp():
    try:
        integrator.resolve_backend("cpp")
    except RuntimeError:
        pytest.skip("C++ backend unavailable")


def make_test_system():
    system, a, sigma, eps, rcut = build_system(
        metal="Ni",
        nx=4,
        ny=4,
        nz=4,
        T0=300.0,
        seed=123,
        thermal_displacement=0.01,
    )

    system.nl.update(system.pos)

    return system, eps, sigma, rcut


def test_python_cpp_force_agreement():
    require_cpp()

    system, eps, sigma, rcut = make_test_system()

    pairs = system.nl.pairs

    # Python
    f_py, pe_py, vir_py = lj_forces(
        system.pos,
        system.box,
        pairs,
        epsilon=eps,
        sigma=sigma,
        rcut=rcut,
    )

    # C++
    pairs64 = np.ascontiguousarray(
        pairs, dtype=np.int64
    )

    f_cpp = np.zeros_like(system.pos)

    pe_cpp, vir_cpp = integrator.md_cpp.lj_forces_cpp(
        system.pos,
        f_cpp,
        system.box,
        pairs64,
        float(eps),
        float(sigma),
        float(rcut),
    )

    np.testing.assert_allclose(
        f_cpp,
        f_py,
        rtol=1e-12,
        atol=1e-12,
    )

    assert pe_cpp == pytest.approx(
        pe_py,
        rel=1e-12,
        abs=1e-12,
    )

    assert vir_cpp == pytest.approx(
        vir_py,
        rel=1e-12,
        abs=1e-12,
    )


def test_python_cpp_one_step_agreement():
    require_cpp()

    initial, eps, sigma, rcut = make_test_system()

    py_system = initial.copy()
    cpp_system = initial.copy()

    dt = 0.01

    integrator.step_nve(
        py_system,
        dt,
        epsilon=eps,
        sigma=sigma,
        rcut=rcut,
        backend="python",
    )

    integrator.step_nve(
        cpp_system,
        dt,
        epsilon=eps,
        sigma=sigma,
        rcut=rcut,
        backend="cpp",
    )

    np.testing.assert_allclose(
        cpp_system.pos,
        py_system.pos,
        rtol=1e-11,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        cpp_system.vel,
        py_system.vel,
        rtol=1e-11,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        cpp_system.force,
        py_system.force,
        rtol=1e-11,
        atol=1e-12,
    )

    assert cpp_system.potential_energy == pytest.approx(
        py_system.potential_energy,
        rel=1e-11,
        abs=1e-12,
    )

    assert cpp_system.total_energy == pytest.approx(
        py_system.total_energy,
        rel=1e-11,
        abs=1e-12,
    )


def test_python_cpp_short_trajectory_agreement():
    require_cpp()

    initial, eps, sigma, rcut = make_test_system()

    py_system = initial.copy()
    cpp_system = initial.copy()

    dt = 0.01
    n_steps = 100

    for _ in range(n_steps):

        integrator.step_nve(
            py_system,
            dt,
            epsilon=eps,
            sigma=sigma,
            rcut=rcut,
            backend="python",
        )

        integrator.step_nve(
            cpp_system,
            dt,
            epsilon=eps,
            sigma=sigma,
            rcut=rcut,
            backend="cpp",
        )

    np.testing.assert_allclose(
        cpp_system.pos,
        py_system.pos,
        rtol=1e-8,
        atol=1e-10,
    )

    np.testing.assert_allclose(
        cpp_system.vel,
        py_system.vel,
        rtol=1e-8,
        atol=1e-10,
    )

    assert cpp_system.total_energy == pytest.approx(
        py_system.total_energy,
        rel=1e-8,
        abs=1e-10,
    )
    

def make_two_particle_system():
    positions = np.array([[4.5, 5.0, 5.0], [5.5, 5.0, 5.0]])
    system = System(positions, mass=10.0, box=[10.0] * 3, cutoff=2.5, skin=0.3)
    system.vel[:] = [[0.0, 0.01, 0.0], [0.0, -0.01, 0.0]]
    return system


def test_python_velocity_verlet_preserves_total_momentum(monkeypatch):
    monkeypatch.setattr(integrator, "_HAVE_CPP", False)
    system = make_two_particle_system()
    initial_momentum = system.mass * system.vel.sum(axis=0)

    for _ in range(20):
        integrator.velocity_verlet(system, dt=0.001)

    final_momentum = system.mass * system.vel.sum(axis=0)
    np.testing.assert_allclose(final_momentum, initial_momentum, atol=1e-11)
    assert np.all(np.isfinite(system.pos))
    assert np.all(np.isfinite(system.vel))


def test_simple_rescale_reaches_target_temperature():
    system = make_two_particle_system()
    target = 300.0
    integrator.simple_rescale_thermostat(system, target)
    assert system.temperature() == pytest.approx(target)


def test_berendsen_moves_temperature_toward_target():
    system = make_two_particle_system()
    initial = system.temperature()
    target = 2.0 * initial

    integrator.berendsen_thermostat(system, T_target=target, tau_T=1.0, dt=0.1)
    assert initial < system.temperature() < target


def test_cpp_neighbor_list_updates_after_drift():
    pytest.importorskip("md.md_cpp")

    if not integrator.cpp_backend_available():
        pytest.skip("C++ extension unavailable")

    # Initially atoms 1 and 2 are outside the
    # neighbor-list radius (cutoff + skin = 2.8).
    positions = np.array([
        [5.0, 5.0, 5.0],
        [7.0, 5.0, 5.0],
        [10.0, 5.0, 5.0],
    ])

    initial = System(
        positions,
        mass=1.0,
        box=[30.0] * 3,
        cutoff=2.5,
        skin=0.3,
    )

    # Move atom 2 toward atom 1.
    initial.vel[2, 0] = -0.8

    py_system = initial.copy()
    cpp_system = initial.copy()

    kwargs = dict(
        dt=1.0,
        epsilon=0.001,
        sigma=1.0,
        rcut=2.5,
    )

    integrator.step_nve(
        py_system, **kwargs, backend="python"
    )

    integrator.step_nve(
        cpp_system, **kwargs, backend="cpp"
    )

    # The new pair must be present after the step.
    assert [1, 2] in cpp_system.nl.pairs.tolist()

    # Both implementations must agree.
    np.testing.assert_allclose(
        cpp_system.pos, py_system.pos,
        rtol=1e-10, atol=1e-12
    )

    np.testing.assert_allclose(
        cpp_system.vel, py_system.vel,
        rtol=1e-10, atol=1e-12
    )

    np.testing.assert_allclose(
        cpp_system.force, py_system.force,
        rtol=1e-10, atol=1e-12
    )

    np.testing.assert_allclose(
        cpp_system.potential_energy,
        py_system.potential_energy,
        rtol=1e-10, atol=1e-12
    )
