import sys
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from md.validation.validation_driver import (
    run_validation_suite,
    print_validation_report,
)


def print_saved_files(saved_files):
    if not saved_files:
        return

    print("\nSaved files:")
    print("-" * 50)
    for name, path in saved_files.items():
        print(f"{name:24s}: {path}")


def main():
    metal = "Ni"
    T0 = 4000.0
    sim_steps = 100000
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
    print("Starting validation...")

    results = run_validation_suite(
        metal=metal,
        nx=nx,
        ny=ny,
        nz=nz,
        T0=T0,
        dt=dt,
        n_steps=sim_steps,
        n_equil_steps=equil_steps,
        refinement_steps=500,
        refinement_dt=0.04,
        sample_every=sample_every,
        tests=None,  # None means run all validation tests
        # tests=["energy_drift", "timestep_refinement"],
        save_outputs=save_outputs,
        save_plots=save_plots,
        show_plots=show_plots,
    )

    print_validation_report(results)
    print_saved_files(results.get("saved_files", {}))

    print()
    print("=" * 58)
    print("  Done.")
    print("=" * 58)
    print()

    if show_plots:
        plt.show()


if __name__ == "__main__":
    main()
