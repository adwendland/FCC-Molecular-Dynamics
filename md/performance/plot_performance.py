"""Generate figures for the FCC MD Python/C++ performance report.

Usage:
    python plot_performance.py python_results.json cpp_results.json

Figures are automatically saved to:
    <project root>/figures/performance/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

# This file lives in:
# MD_FCC/md/performance/plot_performance.py
#
# Therefore parents[2] is the MD_FCC project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "figures" / "performance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def load_json(path):
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def cases_by_n(data):
    return {
        int(case["metadata"]["N"]): case
        for case in data["cases"]
    }


def save_figure(filename):
    path = OUTPUT_DIR / filename

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved: {path}")


# ---------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Plot Python vs C++ FCC MD performance results."
)

parser.add_argument(
    "python_json",
    help="Path to Python performance_results.json",
)

parser.add_argument(
    "cpp_json",
    help="Path to C++ performance_results.json",
)

args = parser.parse_args()


# ---------------------------------------------------------
# Load benchmark data
# ---------------------------------------------------------

python_data = load_json(args.python_json)
cpp_data = load_json(args.cpp_json)

py = cases_by_n(python_data)
cpp = cases_by_n(cpp_data)

Ns = sorted(set(py) & set(cpp))

if not Ns:
    raise RuntimeError(
        "Python and C++ benchmark files contain no matching system sizes."
    )


# ---------------------------------------------------------
# Extract timing data
# ---------------------------------------------------------

py_force = np.array([
    py[n]["force_evaluation"]["median_seconds"]
    for n in Ns
])

cpp_force = np.array([
    cpp[n]["force_evaluation"]["median_seconds"]
    for n in Ns
])

py_integrator = np.array([
    py[n]["integrator_nve"]["median_seconds"]
    for n in Ns
])

cpp_integrator = np.array([
    cpp[n]["integrator_nve"]["median_seconds"]
    for n in Ns
])

force_speedup = py_force / cpp_force
integrator_speedup = py_integrator / cpp_integrator


# =========================================================
# Figure 1: Python-to-C++ speedup
# =========================================================

plt.figure(figsize=(7, 4.5))

plt.plot(
    Ns,
    force_speedup,
    marker="o",
    label="LJ force kernel",
)

plt.plot(
    Ns,
    integrator_speedup,
    marker="o",
    label="NVE integration",
)

plt.xlabel("Number of atoms")
plt.ylabel("Speedup (Python / C++)")
plt.title("Python-to-C++ Performance Speedup")
plt.legend()

save_figure("speedup_vs_atoms.png")


# =========================================================
# Figure 2: NVE integration runtime
# =========================================================

plt.figure(figsize=(7, 4.5))

plt.plot(
    Ns,
    py_integrator,
    marker="o",
    label="Python",
)

plt.plot(
    Ns,
    cpp_integrator,
    marker="o",
    label="C++",
)

plt.yscale("log")

plt.xlabel("Number of atoms")
plt.ylabel("Median runtime for 500 NVE steps (s)")
plt.title("NVE Integration Runtime")
plt.legend()

save_figure("integrator_runtime_vs_atoms.png")


# =========================================================
# Figure 3: LJ force throughput
# =========================================================

py_pair_rate = np.array([
    py[n]["force_evaluation"]["pair_evaluations_per_second"]
    for n in Ns
])

cpp_pair_rate = np.array([
    cpp[n]["force_evaluation"]["pair_evaluations_per_second"]
    for n in Ns
])

plt.figure(figsize=(7, 4.5))

plt.plot(
    Ns,
    py_pair_rate,
    marker="o",
    label="Python",
)

plt.plot(
    Ns,
    cpp_pair_rate,
    marker="o",
    label="C++",
)

plt.yscale("log")

plt.xlabel("Number of atoms")
plt.ylabel("Neighbor-pair evaluations per second")
plt.title("Lennard-Jones Force Throughput")
plt.legend()

save_figure("force_throughput_vs_atoms.png")


# =========================================================
# Figure 4: Neighbor-list construction scaling
# =========================================================

candidate_pairs = np.array([
    py[n]["neighbor_build"]["candidate_pairs"]
    for n in Ns
])

neighbor_time = np.array([
    py[n]["neighbor_build"]["median_seconds"]
    for n in Ns
])

plt.figure(figsize=(7, 4.5))

plt.plot(
    candidate_pairs,
    neighbor_time,
    marker="o",
)

plt.xlabel("Candidate atom pairs")
plt.ylabel("Median neighbor-list build time (s)")
plt.title("Neighbor-List Construction Scaling")

save_figure("neighbor_build_scaling.png")


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print()
print("Performance plots complete.")
print(f"Output directory: {OUTPUT_DIR}")