import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from md.performance import run_performance_suite, print_performance_report


if __name__ == "__main__":
    results = run_performance_suite(
        metal="Cu",
        sizes=((3, 3, 3), (4, 4, 4), (5, 5, 5), (6, 6, 6)),
        T0=300.0,
        dt=0.01,
        n_steps=250,
        repeats=10,
        warmup=2,
        backend="serial-baseline",
        save_outputs=False,
    )
    print_performance_report(results)
    print("Saved files:")
    for name, path in results["saved_files"].items():
        print(f"  {name}: {path}")
