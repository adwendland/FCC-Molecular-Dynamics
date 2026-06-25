"""Driver, reporting, and file output for MD performance benchmarks."""

from __future__ import annotations

import csv
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from .benchmarks import run_size_scaling_benchmarks

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_output_root(output_root, default_subdir):
    if output_root is None:
        return PROJECT_ROOT / "outputs" / default_subdir
    output_root = Path(output_root)
    if output_root.is_absolute():
        return output_root
    return PROJECT_ROOT / output_root


def _json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def run_performance_suite(
    metal="Ni",
    sizes=((3, 3, 3), (4, 4, 4), (5, 5, 5), (6, 6, 6)),
    T0=300.0,
    dt=0.001,
    n_steps=100,
    repeats=5,
    warmup=2,
    seed=123,
    thermal_displacement=0.01,
    backend="auto",
    save_outputs=False,
):
    results = run_size_scaling_benchmarks(
        sizes=sizes,
        metal=metal,
        T0=T0,
        dt=dt,
        n_steps=n_steps,
        repeats=repeats,
        warmup=warmup,
        seed=seed,
        thermal_displacement=thermal_displacement,
        backend=backend,
    )

    if save_outputs:
        results["saved_files"] = save_performance_data(results)
    else:
        results["saved_files"] = {}

    return results


def _runtime_str(seconds):
    return f"{seconds:.2f} s" if seconds < 60 else f"{seconds / 60:.2f} min"


def print_performance_report(results):
    params = results["parameters"]
    env = results["environment"]

    print()
    print("=" * 70)
    print("FCC Molecular Dynamics Performance Report")
    print("=" * 70)

    print("\nSystem")
    print("-" * 70)
    print(f"{'Metal':32s}: {params['metal']}")
    print(f"{'Target temperature':32s}: {params['T0']:.2f} K")
    print(f"{'Time step':32s}: {params['dt_fs']:.4f} fs")
    print(f"{'Integrator steps per repeat':32s}: {params['n_steps']}")
    print(f"{'Benchmark repeats':32s}: {params['repeats']}")
    print(f"{'Backend label':32s}: {results['backend']}")
    print(f"{'Total benchmark runtime':32s}: {_runtime_str(results['runtime_seconds'])}")

    print("\nEnvironment")
    print("-" * 70)
    print(f"{'Python':32s}: {env['python']}")
    print(f"{'Platform':32s}: {env['platform']}")
    print(f"{'CPU count':32s}: {env['cpu_count']}")
    print(f"{'OMP_NUM_THREADS':32s}: {env['omp_num_threads']}")
    print(f"{'MKL_NUM_THREADS':32s}: {env['mkl_num_threads']}")
    print(f"{'OPENBLAS_NUM_THREADS':32s}: {env['openblas_num_threads']}")

    print("\nKernel Timing")
    print("-" * 70)
    print(
        f"{'Size':>8s} {'Atoms':>8s} {'Pairs':>10s} "
        f"{'NL build (ms)':>14s} {'Force (ms)':>12s} {'Step (ms)':>12s}"
    )
    print("-" * 70)

    for case in results["cases"]:
        m = case["metadata"]
        size = f"{m['nx']}x{m['ny']}x{m['nz']}"
        nb = case["neighbor_build"]
        force = case["force_evaluation"]
        integ = case["integrator_nve"]
        step_ms = 1000.0 / integ["steps_per_second"]
        print(
            f"{size:>8s} {m['N']:8d} {force['pairs']:10d} "
            f"{1000.0 * nb['mean_seconds']:14.3f} "
            f"{1000.0 * force['mean_seconds']:12.3f} "
            f"{step_ms:12.3f}"
        )

    print("\nIntegrator Throughput")
    print("-" * 70)
    print(
        f"{'Size':>8s} {'Atoms':>8s} {'steps/s':>12s} "
        f"{'atom-steps/s':>16s} {'ns/day':>12s}"
    )
    print("-" * 70)

    for case in results["cases"]:
        m = case["metadata"]
        integ = case["integrator_nve"]
        size = f"{m['nx']}x{m['ny']}x{m['nz']}"
        print(
            f"{size:>8s} {m['N']:8d} "
            f"{integ['steps_per_second']:12.2f} "
            f"{integ['atom_steps_per_second']:16.2e} "
            f"{integ['simulated_ns_per_day']:12.4f}"
        )

    print("=" * 70)
    print("  Performance suite complete.")
    print("=" * 70)
    print()


def make_performance_output_dir(results, output_root=None, run_name=None):
    params = results["parameters"]
    if run_name is None:
        first = params["sizes"][0]
        last = params["sizes"][-1]
        run_name = (
            f"{params['metal']}_performance_"
            f"{first[0]}x{first[1]}x{first[2]}_to_"
            f"{last[0]}x{last[1]}x{last[2]}"
        )

    output_dir = _resolve_output_root(output_root, "performance") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _flatten_case(case):
    m = case["metadata"]
    nb = case["neighbor_build"]
    force = case["force_evaluation"]
    integ = case["integrator_nve"]

    return {
        "nx": m["nx"],
        "ny": m["ny"],
        "nz": m["nz"],
        "N": m["N"],
        "pairs": force["pairs"],
        "neighbor_build_mean_s": nb["mean_seconds"],
        "neighbor_build_min_s": nb["min_seconds"],
        "force_mean_s": force["mean_seconds"],
        "force_min_s": force["min_seconds"],
        "pair_evaluations_per_second": force["pair_evaluations_per_second"],
        "integrator_mean_s": integ["mean_seconds"],
        "steps_per_second": integ["steps_per_second"],
        "atom_steps_per_second": integ["atom_steps_per_second"],
        "simulated_ns_per_day": integ["simulated_ns_per_day"],
    }


def save_performance_data(results, output_root=None, run_name=None):
    output_dir = make_performance_output_dir(results, output_root=output_root, run_name=run_name)
    saved = {}

    report_path = output_dir / "performance_report.txt"
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print_performance_report(results)
    report_path.write_text(buffer.getvalue(), encoding="utf-8")
    saved["report"] = report_path

    json_path = output_dir / "performance_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, default=_json_safe)
    saved["json"] = json_path

    csv_path = output_dir / "performance_summary.csv"
    rows = [_flatten_case(case) for case in results["cases"]]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    saved["summary_csv"] = csv_path

    return saved
