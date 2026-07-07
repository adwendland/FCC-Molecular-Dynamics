import sys
import matplotlib.pyplot as plt
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
    metal = "Ni"
    ensemble = "nvt"
    T0 = 12000.0
    prod_steps = 100000
    equil_steps = 20000
    dt = 0.1
    sample_every = 500

    nx = 5
    ny = 5
    nz = 5

    save_outputs = True
    save_plots = True
    show_plots = True

    print()
    print("Starting analysis...")

    results = run_analysis_suite(
        metal=metal,
        nx=nx,
        ny=ny,
        nz=nz,
        T0=T0,
        ensemble=ensemble,
        dt=dt,
        n_equil_steps=equil_steps,
        n_steps=prod_steps,
        sample_every=sample_every,
        analyses=None,  # None means run all analyses
        # analyses=["thermo", "rdf", "coordination_number"],
        # analyses=["msd", "vacf", "diffusion_msd", "diffusion_vacf"],
        xyz_file=None,
        xyz_every=None,
        save_outputs=save_outputs,
        save_plots=save_plots,
        show_plots=show_plots,
    )

    print_analysis_report(results)
    # print_saved_files(results.get("saved_files", {}))

    print()
    print("=" * 58)
    print("  Done.")
    print("=" * 58)
    print()

    if show_plots:
        plt.show(block=True)


if __name__ == "__main__":
    main()
