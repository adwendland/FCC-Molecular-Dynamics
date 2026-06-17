import time
import numpy as np

from md.constants import get_lattice_constant, get_sigma, get_eps, get_mass_internal
from md.lattice import make_fcc_lattice
from md.system import System

from md.validation import (
    test_relative_energy_drift,
    test_timestep_refinement,
    test_momentum_conservation,
    test_temperature_stability,
    test_equipartition,
    test_fcc_rdf_peaks,
    test_coordination_number,
)

AVAILABLE_TESTS = {
    "energy_drift",
    "timestep_refinement",
    "momentum",
    "temperature_stability",
    "equipartition",
    "rdf_peaks",
    "coordination_number",
}

kB = 8.617333262145e-5


def initialize_velocities(system, T, seed=123):
    rng = np.random.default_rng(seed)
    N = system.N
    m = system.mass

    if np.isscalar(m):
        std = np.sqrt(kB * T / m)
    else:
        std = np.sqrt(kB * T / m)[:, None]

    system.vel = rng.normal(0.0, 1.0, size=(N, 3)) * std
    system.remove_drift()
    system.update_energies()


def run_validation_suite(
    metal="Ni",
    nx=4,
    ny=4,
    nz=4,
    T0=300.0,
    dt=0.001,
    n_steps=5000,
    sample_every=10,
    refinement_dt=0.01,
    refinement_steps=1000,
    tests=None,
):
    start_time = time.perf_counter()

    if tests is None:
        tests = set(AVAILABLE_TESTS)
    else:
        tests = set(tests)

    unknown_tests = tests - AVAILABLE_TESTS
    if unknown_tests:
        raise ValueError(f"Unknown validation tests: {sorted(unknown_tests)}")

    a = get_lattice_constant(metal)
    mass = get_mass_internal(metal)
    sigma = get_sigma(metal)
    eps = get_eps(metal)
    rcut = 2.5 * sigma

    positions, box = make_fcc_lattice(a, nx, ny, nz)

    system = System(
        positions,
        mass,
        box,
        symbol=metal,
        cutoff=rcut,
        skin=0.3,
    )

    initialize_velocities(system, T0)

    results = {}

    results["metadata"] = {
        "metal": metal,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "N": system.N,
        "T0": T0,
        "dt": dt,
        "n_steps": n_steps,
        "sample_every": sample_every,
        "lattice_constant": a,
        "sigma": sigma,
        "epsilon": eps,
        "rcut": rcut,
        "refinement_dt": refinement_dt,
        "refinement_steps": refinement_steps,
        "tests_requested": sorted(tests),
    }

    results["tolerances"] = {
        "energy_drift": 5e-3,
        "convergence_order_min": 1.5,
        "convergence_order_max": 2.5,
        "momentum_drift": 1e-8,
        "temperature_relative_error": 5e-2,
        "equipartition_relative_error": 5e-2,
        "coordination_relative_error": 5e-2,
        "rdf_first_peak_relative_error": 5e-2,
    }

    tol = results["tolerances"]
    summary = {}

    # -------------------------
    # Energy drift
    # -------------------------
    if "energy_drift" in tests:
        results["energy_drift"] = test_relative_energy_drift(
            system.copy(),
            dt=dt,
            n_steps=n_steps,
            epsilon=eps,
            sigma=sigma,
            rcut=rcut,
            sample_every=sample_every,
        )

        e = results["energy_drift"]
        energies = None

        for key in ["total_energy", "total_energies", "energies", "E_total"]:
            if key in e:
                energies = np.asarray(e[key])
                break

        if energies is not None and len(energies) > 0:
            e["initial_energy"] = float(energies[0])
            e["final_energy"] = float(energies[-1])
        else:
            e["initial_energy"] = np.nan
            e["final_energy"] = np.nan

        summary["Energy Drift"] = e["max_abs_rel"] < tol["energy_drift"]

    # -------------------------
    # Timestep refinement
    # -------------------------
    if "timestep_refinement" in tests:
        results["timestep_refinement"] = test_timestep_refinement(
            system_builder_fn=system.copy,
            dt=refinement_dt,
            n_steps=refinement_steps,
            epsilon=eps,
            sigma=sigma,
            rcut=rcut,
        )

        r = results["timestep_refinement"]

        summary["Timestep Refinement"] = (
            tol["convergence_order_min"]
            <= r["order"]
            <= tol["convergence_order_max"]
        )

    # -------------------------
    # Momentum conservation
    # -------------------------
    if "momentum" in tests:
        results["momentum"] = test_momentum_conservation(
            system.copy(),
            dt=dt,
            n_steps=n_steps,
            epsilon=eps,
            sigma=sigma,
            rcut=rcut,
            sample_every=sample_every,
        )

        m = results["momentum"]
        summary["Momentum Conservation"] = m["max_rel_drift"] < tol["momentum_drift"]

    # -------------------------
    # Temperature stability
    # -------------------------
    if "temperature_stability" in tests:
        results["temperature_stability"] = test_temperature_stability(
            system.copy(),
            dt=dt,
            n_steps=n_steps,
            T_target=T0,
            tau_T=100 * dt,
            epsilon=eps,
            sigma=sigma,
            rcut=rcut,
            sample_every=sample_every,
        )

        temp = results["temperature_stability"]

        summary["Temperature Stability"] = (
            temp["relative_temperature_error"]
            < tol["temperature_relative_error"]
        )

    # -------------------------
    # Equipartition
    # -------------------------
    if "equipartition" in tests:
        results["equipartition"] = test_equipartition(
            system.copy(),
            dt=dt,
            n_steps=n_steps,
            T_target=T0,
            tau_T=100 * dt,
            epsilon=eps,
            sigma=sigma,
            rcut=rcut,
            sample_every=sample_every,
        )

        eq = results["equipartition"]

        summary["Equipartition"] = (
            eq["relative_equipartition_error"]
            < tol["equipartition_relative_error"]
        )

    # -------------------------
    # Structural tests
    # -------------------------
    positions_traj = np.array([system.pos.copy()])

    if "rdf_peaks" in tests:
        results["rdf_peaks"] = test_fcc_rdf_peaks(
            positions_traj=positions_traj,
            box=box,
            lattice_constant=a,
            n_bins=200,
        )

        rdf = results["rdf_peaks"]
        measured_peaks = rdf["measured_peaks"]
        expected_peaks = rdf["expected_peaks"]

        if len(measured_peaks) > 0 and len(expected_peaks) > 0:
            measured_first = float(measured_peaks[0])
            expected_first = float(expected_peaks[0])
            first_peak_rel_error = abs(measured_first - expected_first) / expected_first
        else:
            measured_first = np.nan
            expected_first = np.nan
            first_peak_rel_error = np.nan

        results["rdf_first_peak"] = {
            "measured": measured_first,
            "expected": expected_first,
            "relative_error": first_peak_rel_error,
        }

        summary["RDF First Peak"] = (
            np.isfinite(first_peak_rel_error)
            and first_peak_rel_error < tol["rdf_first_peak_relative_error"]
        )

    if "coordination_number" in tests:
        results["coordination_number"] = test_coordination_number(
            positions_traj=positions_traj,
            box=box,
            lattice_constant=a,
            n_bins=200,
        )

        cn = results["coordination_number"]

        summary["Coordination Number"] = (
            cn["relative_error"] < tol["coordination_relative_error"]
        )

    results["summary"] = summary
    results["runtime_seconds"] = time.perf_counter() - start_time
    results["simulation_time_fs"] = n_steps * dt

    return results


def _status(passed):
    return "PASS" if passed else "FAIL"


def _line(width=58):
    print("-" * width)


def _section(title, width=58):
    print()
    print(title)
    _line(width)


def print_validation_report(results):
    meta = results["metadata"]
    tol = results["tolerances"]
    summary = results["summary"]

    n_passed = sum(summary.values())
    n_total = len(summary)
    overall_pass = n_passed == n_total and n_total > 0

    runtime = results["runtime_seconds"]
    if runtime < 60:
        runtime_str = f"{runtime:.2f} s"
    else:
        runtime_str = f"{runtime / 60:.2f} min"

    sim_time_fs = results["simulation_time_fs"]
    sim_time_ps = sim_time_fs / 1000.0

    print()
    print("=" * 58)
    print("FCC Molecular Dynamics Validation Report")
    print("=" * 58)

    _section("System")
    print(f"{'Metal':28s}: {meta['metal']}")
    print(f"{'Lattice size':28s}: {meta['nx']} x {meta['ny']} x {meta['nz']}")
    print(f"{'Atoms':28s}: {meta['N']}")
    print(f"{'Target temperature':28s}: {meta['T0']:.2f} K")
    print(f"{'Production time step':28s}: {meta['dt']:.4f} fs")
    print(f"{'Simulation time':28s}: {sim_time_fs:.4f} fs ({sim_time_ps:.4f} ps)")
    print(f"{'Wall runtime':28s}: {runtime_str}")
    print(f"{'Production steps':28s}: {meta['n_steps']}")
    print(f"{'Lattice constant':28s}: {meta['lattice_constant']:.6f} Å")
    print(f"{'Cutoff radius':28s}: {meta['rcut']:.6f} Å")

    if "energy_drift" in results or "momentum" in results:
        _section("Conservation Tests")

        if "energy_drift" in results:
            e = results["energy_drift"]
            print(f"{'Energy Drift':28s}: {_status(summary['Energy Drift'])}")
            print(f"  {'Initial Energy':26s}: {e['initial_energy']:.6e} eV")
            print(f"  {'Final Energy':26s}: {e['final_energy']:.6e} eV")
            print(f"  {'Max Relative Drift':26s}: {e['max_abs_rel']:.3e}")
            print(f"  {'Tolerance':26s}: {tol['energy_drift']:.3e}")

        if "momentum" in results:
            m = results["momentum"]
            print()
            print(f"{'Momentum Conservation':28s}: {_status(summary['Momentum Conservation'])}")
            print(f"  {'Max Normalized Drift':26s}: {m['max_rel_drift']:.3e}")
            print(f"  {'Tolerance':26s}: {tol['momentum_drift']:.3e}")

    if "timestep_refinement" in results:
        _section("Numerical Convergence Tests")

        r = results["timestep_refinement"]
        print(f"{'Timestep Refinement':28s}: {_status(summary['Timestep Refinement'])}")
        print(f"  {'Refinement dt':26s}: {meta['refinement_dt']:.4f} fs")
        print(f"  {'Refinement steps':26s}: {meta['refinement_steps']}")
        print(f"  {'Observed Order':26s}: {r['order']:.4f}")
        print(
            f"  {'Expected Range':26s}: "
            f"[{tol['convergence_order_min']:.1f}, {tol['convergence_order_max']:.1f}]"
        )

        print()
        print("  dt (fs)        Position Error")
        print("  ------------------------------")

        for dt_test, err in sorted(r["errors"].items(), reverse=True):
            print(f"  {float(dt_test):10.4f}    {float(err):14.3e}")

        print(f"  {'Reference dt':14s}: {r['reference_dt']:.4f} fs")

    if "temperature_stability" in results or "equipartition" in results:
        _section("Thermodynamic Tests")

        if "temperature_stability" in results:
            temp = results["temperature_stability"]
            print(f"{'Temperature Stability':28s}: {_status(summary['Temperature Stability'])}")
            print(f"  {'Mean Temperature':26s}: {temp['mean_temperature']:.4f} K")
            print(f"  {'Relative Error':26s}: {temp['relative_temperature_error']:.3e}")
            print(f"  {'Tolerance':26s}: {tol['temperature_relative_error']:.3e}")

        if "equipartition" in results:
            eq = results["equipartition"]
            print()
            print(f"{'Equipartition':28s}: {_status(summary['Equipartition'])}")
            print(f"  {'Relative Error':26s}: {eq['relative_equipartition_error']:.3e}")
            print(f"  {'Tolerance':26s}: {tol['equipartition_relative_error']:.3e}")

    if "coordination_number" in results or "rdf_peaks" in results:
        _section("Structural Tests")

        if "coordination_number" in results:
            cn = results["coordination_number"]
            print(f"{'Coordination Number':28s}: {_status(summary['Coordination Number'])}")
            print(f"  {'Measured CN':26s}: {cn['coordination_number']:.4f}")
            print(f"  {'Expected CN':26s}: 12")
            print(f"  {'Relative Error':26s}: {cn['relative_error']:.3e}")
            print(f"  {'Tolerance':26s}: {tol['coordination_relative_error']:.3e}")

        if "rdf_peaks" in results:
            rdf = results["rdf_first_peak"]
            print()
            print(f"{'RDF First Peak':28s}: {_status(summary['RDF First Peak'])}")
            print(f"  {'Measured':26s}: {rdf['measured']:.4f} Å")
            print(f"  {'Expected':26s}: {rdf['expected']:.4f} Å")
            print(f"  {'Relative Error':26s}: {rdf['relative_error']:.3e}")
            print(f"  {'Tolerance':26s}: {tol['rdf_first_peak_relative_error']:.3e}")

    _section("Overall Result")
    print(f"{n_passed} / {n_total} tests passed")
    print()
    print(f"Validation Status: {_status(overall_pass)}")
    print("=" * 58)
    print()