import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from md.analysis.analysis_driver import (
    run_analysis_suite,
    print_analysis_report,
)


def print_saved_files(saved_files):
    if not saved_files:
        return

    print("\nSaved files:")
    print("-" * 50)
    for name, path in saved_files.items():
        print(f"{name:20s}: {path}")


def main():
    print()
    print("Starting analysis...")

    results = run_analysis_suite(
        metal="Ni",
        nx=4,
        ny=4,
        nz=4,
        T0=8000.0,
        ensemble="nve",
        dt=0.1,
        n_equil_steps=20000,
        n_steps=100000,
        sample_every=10,
        analyses=None,  # None means run all analyses
        # analyses=["thermo", "rdf", "coordination_number"],
        # analyses=["msd", "vacf", "diffusion_msd", "diffusion_vacf"],
        xyz_file=None,
        xyz_every=None,
        save_outputs=True,
    )

    print_analysis_report(results)
    print_saved_files(results.get("saved_files", {}))

    print()
    print("=" * 58)
    print("  Done.")
    print("=" * 58)
    print()


if __name__ == "__main__":
    main()
