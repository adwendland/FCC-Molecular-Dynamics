import sys
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
    print()
    print("Starting validation...")

    results = run_validation_suite(
        metal="Ni",
        nx=4,
        ny=4,
        nz=4,
        T0=4000.0,
        dt=0.1,
        n_steps=100000,
        refinement_steps=1000,
        refinement_dt=0.02,
        sample_every=10,
        tests=None,  # None means run all validation tests
        # tests=["energy_drift", "timestep_refinement"],
        save_outputs=True,
    )

    print_validation_report(results)
    print_saved_files(results.get("saved_files", {}))

    print()
    print("=" * 58)
    print("  Done.")
    print("=" * 58)
    print()


if __name__ == "__main__":
    main()
