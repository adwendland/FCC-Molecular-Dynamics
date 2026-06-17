import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from MD_FCC.md.validation.validation_driver import run_validation_suite, print_validation_report


def main():
    print("Starting validation...")

    results = run_validation_suite(
        metal="Ni",
        nx=4,
        ny=4,
        nz=4,
        T0=300.0,
        dt=0.1,
        n_steps=1000,
        sample_every=10,
        tests = None
    )

    print_validation_report(results)
    print("Done.")


if __name__ == "__main__":
    main()