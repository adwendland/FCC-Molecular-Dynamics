import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from md.analysis.analysis_driver import (
    run_analysis_suite,
    print_analysis_report,
    save_analysis_data,
)


def main():
    print()
    print("Starting analysis...")

    results = run_analysis_suite(
        metal="Ni",
        nx=4,
        ny=4,
        nz=4,
        T0=8000.0,
        ensemble="nvt",
        dt=0.1,
        n_equil_steps=20000,
        n_steps=100000,
        sample_every=50,
        analyses=None,  # None means run all analyses
        # analyses=["thermo", "rdf", "coordination_number"],
        # analyses=["msd", "vacf", "diffusion_msd", "diffusion_vacf"],
        xyz_file=None,
        xyz_every=None,
    )

    print_analysis_report(results)

    
    # saved_files = save_analysis_data(results, prefix="ni_analysis")
    saved_files = None
    if saved_files:
        print("Saved data files:")
        for filename in saved_files:
            print(f"  {filename}")
    

    print("Done.")


if __name__ == "__main__":
    main()