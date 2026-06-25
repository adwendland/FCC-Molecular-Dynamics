"""Performance benchmarks for the FCC molecular dynamics code.

These are timing benchmarks, not validation tests.  They are meant to answer:
    - how fast is the force kernel?
    - how fast is neighbor-list construction?
    - how fast is full time integration?
    - how does runtime scale with system size?

The output is intentionally backend-agnostic so the same suite can later compare
serial NumPy, C++/pybind11, OpenMP, Numba, multiprocessing, etc.
"""

from __future__ import annotations

import os
import platform
import statistics
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from md.analysis.analysis_driver import build_system
from md.forces import lj_forces
from md.integrator import step_nve


@dataclass(frozen=True)
class TimingStats:
    """Repeated timing summary."""

    mean_seconds: float
    median_seconds: float
    min_seconds: float
    max_seconds: float
    std_seconds: float
    repeats: int


def _time_repeated(fn: Callable[[], None], repeats: int = 5, warmup: int = 2) -> TimingStats:
    """Time a zero-argument function with warmup and repeated measurements."""
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if warmup < 0:
        raise ValueError("warmup must be nonnegative")

    for _ in range(warmup):
        fn()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)

    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return TimingStats(
        mean_seconds=float(statistics.mean(times)),
        median_seconds=float(statistics.median(times)),
        min_seconds=float(min(times)),
        max_seconds=float(max(times)),
        std_seconds=float(std),
        repeats=int(repeats),
    )


def _stats_dict(stats: TimingStats) -> dict:
    return {
        "mean_seconds": stats.mean_seconds,
        "median_seconds": stats.median_seconds,
        "min_seconds": stats.min_seconds,
        "max_seconds": stats.max_seconds,
        "std_seconds": stats.std_seconds,
        "repeats": stats.repeats,
    }


def _environment_metadata() -> dict:
    """Basic reproducibility metadata for performance reports."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
    }


def benchmark_neighbor_build(system, repeats: int = 5, warmup: int = 2) -> dict:
    """Benchmark rebuilding the Verlet neighbor list from scratch."""
    def build_once():
        system.nl._build(system.pos)

    stats = _time_repeated(build_once, repeats=repeats, warmup=warmup)
    n_pairs = int(len(system.nl.pairs))
    candidate_pairs = system.N * (system.N - 1) // 2

    return {
        "benchmark": "neighbor_build",
        "N": int(system.N),
        "pairs": n_pairs,
        "candidate_pairs": int(candidate_pairs),
        **_stats_dict(stats),
        "builds_per_second": float(1.0 / stats.mean_seconds),
        "candidate_pairs_per_second": float(candidate_pairs / stats.mean_seconds),
    }


def benchmark_force_evaluation(
    system,
    epsilon: float,
    sigma: float,
    rcut: float,
    repeats: int = 5,
    warmup: int = 2,
) -> dict:
    """Benchmark one LJ force evaluation using the current neighbor list."""
    system.nl.update(system.pos)
    pairs = system.nl.pairs
    n_pairs = int(len(pairs))

    def force_once():
        lj_forces(system.pos, system.box, pairs, epsilon=epsilon, sigma=sigma, rcut=rcut)

    stats = _time_repeated(force_once, repeats=repeats, warmup=warmup)

    return {
        "benchmark": "force_evaluation",
        "N": int(system.N),
        "pairs": n_pairs,
        **_stats_dict(stats),
        "force_calls_per_second": float(1.0 / stats.mean_seconds),
        "pair_evaluations_per_second": float(n_pairs / stats.mean_seconds),
    }


def benchmark_integrator(
    system,
    dt: float,
    n_steps: int,
    epsilon: float,
    sigma: float,
    rcut: float,
    repeats: int = 3,
    warmup_steps: int = 5,
) -> dict:
    """Benchmark full NVE velocity-Verlet throughput."""
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")

    def run_steps():
        s = system.copy()
        for _ in range(n_steps):
            step_nve(s, dt, epsilon=epsilon, sigma=sigma, rcut=rcut)

    # Warmup with a shorter run so compilation/import/cache effects do not dominate.
    for _ in range(max(0, warmup_steps)):
        s = system.copy()
        step_nve(s, dt, epsilon=epsilon, sigma=sigma, rcut=rcut)

    stats = _time_repeated(run_steps, repeats=repeats, warmup=0)
    steps_per_second = n_steps / stats.mean_seconds
    atom_steps_per_second = system.N * steps_per_second
    simulated_ns_per_day = steps_per_second * dt * 86400.0e-6  # fs/step -> ns/day

    return {
        "benchmark": "integrator_nve",
        "N": int(system.N),
        "dt_fs": float(dt),
        "n_steps_per_repeat": int(n_steps),
        **_stats_dict(stats),
        "steps_per_second": float(steps_per_second),
        "atom_steps_per_second": float(atom_steps_per_second),
        "simulated_ns_per_day": float(simulated_ns_per_day),
    }


def benchmark_one_size(
    metal: str = "Ni",
    nx: int = 4,
    ny: int = 4,
    nz: int = 4,
    T0: float = 300.0,
    dt: float = 0.001,
    n_steps: int = 100,
    repeats: int = 5,
    warmup: int = 2,
    seed: int = 123,
    thermal_displacement: float = 0.01,
) -> dict:
    """Run all standard benchmarks for one system size."""
    system, a, sigma, eps, rcut = build_system(
        metal=metal,
        nx=nx,
        ny=ny,
        nz=nz,
        T0=T0,
        seed=seed,
        thermal_displacement=thermal_displacement,
    )

    # Ensure force state and neighbor list are initialized consistently.
    system.nl.update(system.pos)

    metadata = {
        "metal": metal,
        "nx": int(nx),
        "ny": int(ny),
        "nz": int(nz),
        "N": int(system.N),
        "T0": float(T0),
        "dt_fs": float(dt),
        "lattice_constant": float(a),
        "sigma": float(sigma),
        "epsilon": float(eps),
        "rcut": float(rcut),
        "skin": float(system.skin),
    }

    return {
        "metadata": metadata,
        "neighbor_build": benchmark_neighbor_build(system.copy(), repeats=repeats, warmup=warmup),
        "force_evaluation": benchmark_force_evaluation(
            system.copy(), eps, sigma, rcut, repeats=repeats, warmup=warmup
        ),
        "integrator_nve": benchmark_integrator(
            system.copy(), dt, n_steps, eps, sigma, rcut, repeats=max(1, repeats // 2), warmup_steps=warmup
        ),
    }


def run_size_scaling_benchmarks(
    sizes=((3, 3, 3), (4, 4, 4), (5, 5, 5), (6, 6, 6)),
    metal: str = "Ni",
    T0: float = 300.0,
    dt: float = 0.001,
    n_steps: int = 100,
    repeats: int = 5,
    warmup: int = 2,
    seed: int = 123,
    thermal_displacement: float = 0.01,
    backend: str = "auto",
) -> dict:
    """Run the standard size-scaling performance suite."""
    print(" Starting performance suite...")
    t0 = time.perf_counter()

    cases = []
    for nx, ny, nz in sizes:
        cases.append(
            benchmark_one_size(
                metal=metal,
                nx=nx,
                ny=ny,
                nz=nz,
                T0=T0,
                dt=dt,
                n_steps=n_steps,
                repeats=repeats,
                warmup=warmup,
                seed=seed,
                thermal_displacement=thermal_displacement,
            )
        )

    return {
        "suite": "FCC MD Performance Suite",
        "backend": backend,
        "environment": _environment_metadata(),
        "parameters": {
            "metal": metal,
            "T0": float(T0),
            "dt_fs": float(dt),
            "n_steps": int(n_steps),
            "repeats": int(repeats),
            "warmup": int(warmup),
            "seed": int(seed),
            "thermal_displacement": float(thermal_displacement),
            "sizes": [list(s) for s in sizes],
        },
        "cases": cases,
        "runtime_seconds": float(time.perf_counter() - t0),
    }
