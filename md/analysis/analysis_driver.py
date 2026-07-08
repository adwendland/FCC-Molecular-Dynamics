# analysis_driver.py

import time
import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np

from md.constants import get_lattice_constant, get_sigma, get_eps, get_mass_internal
from md.lattice import make_fcc_lattice
from md.system import System
from md.integrator import step_nve, step_nvt_berendsen
from md.utils import write_xyz
from md.plotting import (
    plot_temperature,
    plot_energy,
    plot_msd,
    plot_rdf,
    plot_structure_factor,
    plot_vacf,
)

from .rdf import compute_rdf, compute_coordination_number
from .structure_factor import compute_structure_factor
from .msd import compute_msd
from .vacf import compute_vacf
from .transport import compute_diffusion_from_msd, compute_diffusion_from_vacf
from .thermodynamics import compute_pressure, compute_heat_capacity_from_energy


AVAILABLE_ANALYSES = {
    "thermo",
    "rdf",
    "msd",
    "vacf",
    "diffusion_msd",
    "diffusion_vacf",
    "coordination_number",
    "heat_capacity",
    "structure_factor",
}

kB = 8.617333262145e-5
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def initialize_velocities(system, T, seed=123):
    rng = np.random.default_rng(seed)
    N = system.N
    m = system.mass

    if np.isscalar(m):
        std = np.sqrt(kB * T / m)
    else:
        std = np.sqrt(kB * T / m)[:, None]

    system.vel = rng.normal(0.0, 1.0, size=(N, 3)) * std
    system.remove_drift()
    system.update_energies()


def build_system(
    metal="Ni",
    nx=4,
    ny=4,
    nz=4,
    T0=300.0,
    seed=123,
    thermal_displacement=0.01,
):
    a = get_lattice_constant(metal)
    mass = get_mass_internal(metal)
    sigma = get_sigma(metal)
    eps = get_eps(metal)
    rcut = 2.5 * sigma

    positions, box = make_fcc_lattice(a, nx, ny, nz)

    if thermal_displacement > 0.0:
        rng = np.random.default_rng(seed)
        positions += thermal_displacement * a * rng.normal(size=positions.shape)
        positions %= box

    system = System(
        positions,
        mass,
        box,
        symbol=metal,
        cutoff=rcut,
        skin=0.3,
    )

    initialize_velocities(system, T0, seed=seed)
    return system, a, sigma, eps, rcut


def run_simulation(
    system,
    dt,
    n_steps,
    sample_every,
    ensemble,
    T_target,
    tau_T,
    epsilon,
    sigma,
    rcut,
    xyz_file=None,
    xyz_every=None,
):
    if ensemble not in {"nve", "nvt"}:
        raise ValueError("ensemble must be 'nve' or 'nvt'")

    n_samples = n_steps // sample_every + 1

    positions_traj = np.zeros((n_samples, system.N, 3))
    velocities_traj = np.zeros((n_samples, system.N, 3))
    pressure_traj = np.zeros(n_samples)
    kinetic_traj = np.zeros(n_samples)
    potential_traj = np.zeros(n_samples)
    energy_traj = np.zeros(n_samples)
    temp_traj = np.zeros(n_samples)
    step_traj = np.zeros(n_samples, dtype=int)

    system.update_energies()

    positions_traj[0] = system.pos.copy()
    velocities_traj[0] = system.vel.copy()
    pressure_traj[0] = compute_pressure(system)
    kinetic_traj[0] = system.kinetic_energy
    potential_traj[0] = system.potential_energy
    energy_traj[0] = system.total_energy
    temp_traj[0] = system.temperature()
    step_traj[0] = 0

    if xyz_file is not None:
        write_xyz(system, step=0, filename=xyz_file)

    sample_idx = 0

    for step in range(1, n_steps + 1):
        if ensemble == "nve":
            step_nve(system, dt, epsilon=epsilon, sigma=sigma, rcut=rcut)
        else:
            step_nvt_berendsen(
                system,
                dt,
                T_target=T_target,
                tau_T=tau_T,
                epsilon=epsilon,
                sigma=sigma,
                rcut=rcut,
            )

        if xyz_file is not None and xyz_every is not None and step % xyz_every == 0:
            write_xyz(system, step=step, filename=xyz_file)

        if step % sample_every == 0:
            sample_idx += 1
            system.update_energies()

            positions_traj[sample_idx] = system.pos.copy()
            velocities_traj[sample_idx] = system.vel.copy()
            pressure_traj[sample_idx] = compute_pressure(system)
            kinetic_traj[sample_idx] = system.kinetic_energy
            potential_traj[sample_idx] = system.potential_energy
            energy_traj[sample_idx] = system.total_energy
            temp_traj[sample_idx] = system.temperature()
            step_traj[sample_idx] = step

    step_traj = step_traj[: sample_idx + 1]

    return {
        "positions_traj": positions_traj[: sample_idx + 1],
        "velocities_traj": velocities_traj[: sample_idx + 1],
        "pressure_traj": pressure_traj[: sample_idx + 1],
        "kinetic_traj": kinetic_traj[: sample_idx + 1],
        "potential_traj": potential_traj[: sample_idx + 1],
        "energy_traj": energy_traj[: sample_idx + 1],
        "temp_traj": temp_traj[: sample_idx + 1],
        "step_traj": step_traj,
        "time_traj": step_traj * dt,
    }


def _resolve_output_root(output_root, default_subdir):
    if output_root is None:
        return PROJECT_ROOT / "outputs" / default_subdir

    output_root = Path(output_root)
    if output_root.is_absolute():
        return output_root

    return PROJECT_ROOT / output_root


def make_analysis_output_dir(results, output_root=None, run_name=None):
    meta = results["metadata"]

    if run_name is None:
        run_name = (
            f"{meta['metal']}_{meta['ensemble']}_"
            f"{int(meta['T0'])}K_{meta['nx']}x{meta['ny']}x{meta['nz']}"
        )

    output_dir = _resolve_output_root(output_root, "analysis") / run_name
    plot_dir = output_dir / "plots"

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    return output_dir, plot_dir



def write_xyz_trajectory(path, positions_traj, symbol, box=None, steps=None, times=None):
    """Write a multi-frame extended XYZ trajectory.

    The output is OVITO-friendly. ``positions_traj`` must have shape
    ``(n_frames, n_atoms, 3)``. If ``box`` is provided, the second line of each
    frame includes an extended-XYZ Lattice field.
    """
    path = Path(path)
    positions_traj = np.asarray(positions_traj)

    if positions_traj.ndim != 3 or positions_traj.shape[2] != 3:
        raise ValueError("positions_traj must have shape (n_frames, n_atoms, 3)")

    n_frames, n_atoms, _ = positions_traj.shape

    if steps is None:
        steps = np.arange(n_frames)
    if times is None:
        times = [None] * n_frames

    with open(path, "w", encoding="utf-8") as f:
        for frame_idx, positions in enumerate(positions_traj):
            f.write(f"{n_atoms}\n")

            fields = [f"Frame={frame_idx}", f"Step={int(steps[frame_idx])}"]
            if times[frame_idx] is not None:
                fields.append(f"Time_fs={float(times[frame_idx]):.8f}")

            if box is not None:
                Lx, Ly, Lz = np.asarray(box, dtype=float)
                fields.append(
                    f'Lattice="{Lx:.8f} 0 0  0 {Ly:.8f} 0  0 0 {Lz:.8f}"'
                )
                fields.append('Properties=species:S:1:pos:R:3')
            else:
                fields.append('Properties=species:S:1:pos:R:3')

            f.write(" ".join(fields) + "\n")

            for x, y, z in positions:
                f.write(f"{symbol} {x:.8f} {y:.8f} {z:.8f}\n")


def write_xyz_snapshot(path, positions, symbol, box=None, step=None, time=None):
    """Write a one-frame extended XYZ snapshot."""
    path = Path(path)
    positions = np.asarray(positions)

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (n_atoms, 3)")

    n_atoms = positions.shape[0]

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{n_atoms}\n")

        fields = ["Final configuration"]
        if step is not None:
            fields.append(f"Step={int(step)}")
        if time is not None:
            fields.append(f"Time_fs={float(time):.8f}")
        if box is not None:
            Lx, Ly, Lz = np.asarray(box, dtype=float)
            fields.append(
                f'Lattice="{Lx:.8f} 0 0  0 {Ly:.8f} 0  0 0 {Lz:.8f}"'
            )
        fields.append('Properties=species:S:1:pos:R:3')
        f.write(" ".join(fields) + "\n")

        for x, y, z in positions:
            f.write(f"{symbol} {x:.8f} {y:.8f} {z:.8f}\n")


def generate_analysis_plots(
    results,
    output_root=None,
    run_name=None,
    save_plots=False,
    show_plots=False,
):
    meta = results["metadata"]

    if not save_plots and not show_plots:
        return {}

    saved = {}
    plot_dir = None

    if save_plots:
        _, plot_dir = make_analysis_output_dir(
            results,
            output_root=output_root,
            run_name=run_name,
        )

    def maybe_path(filename):
        if save_plots:
            return plot_dir / filename
        return None

    traj = results["trajectory"]

    plot_temperature(
        traj["time_traj"],
        traj["temp_traj"],
        path=maybe_path("temperature.png"),
        show=show_plots,
        meta=meta,
    )
    if save_plots:
        saved["temperature_plot"] = plot_dir / "temperature.png"

    plot_energy(
        traj["time_traj"],
        traj["kinetic_traj"],
        traj["potential_traj"],
        traj["energy_traj"],
        path=maybe_path("energy.png"),
        show=show_plots,
        meta=meta,
    )
    if save_plots:
        saved["energy_plot"] = plot_dir / "energy.png"

    if "rdf" in results:
        plot_rdf(
            results["rdf"]["r"],
            results["rdf"]["g_r"],
            path=maybe_path("rdf.png"),
            show=show_plots,
            meta=meta,
        )
        if save_plots:
            saved["rdf_plot"] = plot_dir / "rdf.png"

    if "structure_factor" in results:
        plot_structure_factor(
            results["structure_factor"]["k"],
            results["structure_factor"]["S_k"],
            path=maybe_path("structure_factor.png"),
            show=show_plots,
            meta=meta,
        )
        if save_plots:
            saved["structure_factor_plot"] = plot_dir / "structure_factor.png"

    if "msd" in results:
        plot_msd(
            results["msd"]["time"],
            results["msd"]["msd"],
            path=maybe_path("msd.png"),
            show=show_plots,
            meta=meta,
        )
        if save_plots:
            saved["msd_plot"] = plot_dir / "msd.png"

    if "vacf" in results:
        plot_vacf(
            results["vacf"]["time"],
            results["vacf"]["vacf"],
            path=maybe_path("vacf.png"),
            show=show_plots,
            meta=meta,
        )
        if save_plots:
            saved["vacf_plot"] = plot_dir / "vacf.png"

    return saved


def run_analysis_suite(
    metal="Ni",
    nx=4,
    ny=4,
    nz=4,
    T0=300.0,
    ensemble="nve",
    dt=0.01,
    n_equil_steps=0,
    n_steps=5000,
    sample_every=10,
    analyses=None,
    seed=123,
    thermal_displacement=0.01,
    r_max_factor=0.45,
    n_bins=200,
    k_min=0.1,
    k_max=12.0,
    n_k=300,
    xyz_file=None,
    xyz_every=None,
    output_root=None,
    run_name=None,
    save_outputs=False,
    save_trajectory=False,
    save_plots=False,
    show_plots=False,
):
    start_time = time.perf_counter()

    if analyses is None:
        analyses = set(AVAILABLE_ANALYSES)
    else:
        analyses = set(analyses)

    unknown_analyses = analyses - AVAILABLE_ANALYSES
    if unknown_analyses:
        raise ValueError(f"Unknown analyses: {sorted(unknown_analyses)}")

    system, a, sigma, eps, rcut = build_system(
        metal=metal,
        nx=nx,
        ny=ny,
        nz=nz,
        T0=T0,
        seed=seed,
        thermal_displacement=thermal_displacement,
    )

    tau_T = 100 * dt

    if n_equil_steps > 0:
        print("Starting equilibration steps...")

    for _ in range(n_equil_steps):
        step_nvt_berendsen(
            system,
            dt,
            T_target=T0,
            tau_T=tau_T,
            epsilon=eps,
            sigma=sigma,
            rcut=rcut,
        )

    print("Starting production steps...")

    traj = run_simulation(
        system=system,
        dt=dt,
        n_steps=n_steps,
        sample_every=sample_every,
        ensemble=ensemble,
        T_target=T0,
        tau_T=tau_T,
        epsilon=eps,
        sigma=sigma,
        rcut=rcut,
        xyz_file=xyz_file,
        xyz_every=xyz_every,
    )

    print("Performing analysis...")
    print()

    positions_traj = traj["positions_traj"]
    velocities_traj = traj["velocities_traj"]
    pressure_traj = traj["pressure_traj"]
    energy_traj = traj["energy_traj"]
    temp_traj = traj["temp_traj"]
    time_traj = traj["time_traj"]

    results = {
        "metadata": {
            "metal": metal,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "N": system.N,
            "T0": T0,
            "ensemble": ensemble,
            "dt": dt,
            "n_equil_steps": n_equil_steps,
            "n_steps": n_steps,
            "time_ps": n_steps*dt/1000.0,
            "sample_every": sample_every,
            "n_samples": len(time_traj),
            "lattice_constant": a,
            "sigma": sigma,
            "epsilon": eps,
            "rcut": rcut,
            "analyses_requested": sorted(analyses),
        },
        "trajectory": traj,
        "summary": {},
    }

    rho = system.N / system.volume()
    dt_sample = dt * sample_every

    if "thermo" in analyses:
        half = max(1, len(temp_traj) // 2)
        results["thermo"] = {
            "mean_temperature": float(np.mean(temp_traj[half:])),
            "std_temperature": float(np.std(temp_traj[half:])),
            "mean_pressure": float(np.mean(pressure_traj[half:])),
            "std_pressure": float(np.std(pressure_traj[half:])),
            "mean_total_energy": float(np.mean(energy_traj[half:])),
            "std_total_energy": float(np.std(energy_traj[half:])),
        }
        results["summary"].update(results["thermo"])

    need_rdf = bool({"rdf", "coordination_number", "structure_factor"} & analyses)

    if need_rdf:
        r_max = r_max_factor * min(system.box)
        r, g_r = compute_rdf(positions_traj, system.box, r_max, n_bins)
        results["rdf"] = {
            "r": r,
            "g_r": g_r,
            "r_max": r_max,
            "n_bins": n_bins,
        }

    if "msd" in analyses or "diffusion_msd" in analyses:
        t_msd, msd = compute_msd(positions_traj, system.box)
        t_msd *= dt_sample
        results["msd"] = {
            "time": t_msd,
            "msd": msd,
        }

    if "vacf" in analyses or "diffusion_vacf" in analyses:
        t_vacf, vacf = compute_vacf(velocities_traj)
        t_vacf *= dt_sample
        results["vacf"] = {
            "time": t_vacf,
            "vacf": vacf,
        }

    if "diffusion_msd" in analyses:
        D_msd, slope = compute_diffusion_from_msd(
            results["msd"]["time"],
            results["msd"]["msd"],
        )
        results["diffusion_msd"] = {
            "D": float(D_msd),
            "slope": float(slope),
        }
        results["summary"]["D_msd"] = float(D_msd)

    if "diffusion_vacf" in analyses:
        D_vacf, integral = compute_diffusion_from_vacf(
            results["vacf"]["time"],
            results["vacf"]["vacf"],
        )
        results["diffusion_vacf"] = {
            "D": float(D_vacf),
            "integral": float(integral),
        }
        results["summary"]["D_vacf"] = float(D_vacf)

    if "coordination_number" in analyses:
        r = results["rdf"]["r"]
        g_r = results["rdf"]["g_r"]

        idx_peak = int(np.argmax(g_r))
        idx_min = idx_peak + int(np.argmin(g_r[idx_peak:]))
        r_cn = float(r[idx_min])

        CN = compute_coordination_number(r, g_r, rho, r_cn)

        results["coordination_number"] = {
            "coordination_number": float(CN),
            "r_cut_cn": r_cn,
            "expected_fcc": 12.0,
        }
        results["summary"]["coordination_number"] = float(CN)

    if "heat_capacity" in analyses:
        half = max(1, len(energy_traj) // 2)
        E_tail = energy_traj[half:]
        T_mean = float(np.mean(temp_traj[half:]))

        Cv = compute_heat_capacity_from_energy(E_tail, T_mean)

        results["heat_capacity"] = {
            "Cv": float(Cv),
            "T_mean": T_mean,
        }
        results["summary"]["Cv"] = float(Cv)

    if "structure_factor" in analyses:
        k_values = np.linspace(k_min, k_max, n_k)
        k_values, S_k = compute_structure_factor(
            k_values,
            results["rdf"]["r"],
            results["rdf"]["g_r"],
            rho,
        )

        results["structure_factor"] = {
            "k": k_values,
            "S_k": S_k,
        }

    results["runtime_seconds"] = time.perf_counter() - start_time
    results["simulation_time_fs"] = n_steps * dt

    results["saved_files"] = {}

    if save_outputs or save_trajectory:
        saved_files = save_analysis_data(
            results,
            output_root=output_root,
            run_name=run_name,
            save_data=save_outputs,
            save_trajectory=save_trajectory,
        )
        results["saved_files"].update(saved_files)

    if save_plots or show_plots:
        saved_plots = generate_analysis_plots(
            results,
            output_root=output_root,
            run_name=run_name,
            save_plots=save_plots,
            show_plots=show_plots,
        )
        results["saved_files"].update(saved_plots)

    return results


def _json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_analysis_data(
    results,
    output_root=None,
    run_name=None,
    save_data=True,
    save_trajectory=False,
):
    output_dir, _ = make_analysis_output_dir(
        results,
        output_root=output_root,
        run_name=run_name,
    )

    saved = {}

    if save_data:
        if "thermo" in results:
            traj = results["trajectory"]
            path = output_dir / "thermo.dat"

            np.savetxt(
                path,
                np.column_stack(
                    (
                        traj["time_traj"],
                        traj["temp_traj"],
                        traj["pressure_traj"],
                        traj["kinetic_traj"],
                        traj["potential_traj"],
                        traj["energy_traj"],
                    )
                ),
                header=(
                    "time_fs temperature_K pressure_eV_per_A3 "
                    "kinetic_energy_eV potential_energy_eV total_energy_eV"
                ),
            )
            saved["thermo"] = path

        if "rdf" in results:
            path = output_dir / "rdf.dat"
            np.savetxt(
                path,
                np.column_stack((results["rdf"]["r"], results["rdf"]["g_r"])),
                header="r_A g_r",
            )
            saved["rdf"] = path

        if "msd" in results:
            path = output_dir / "msd.dat"
            np.savetxt(
                path,
                np.column_stack((results["msd"]["time"], results["msd"]["msd"])),
                header="time_fs msd_A2",
            )
            saved["msd"] = path

        if "vacf" in results:
            path = output_dir / "vacf.dat"
            np.savetxt(
                path,
                np.column_stack((results["vacf"]["time"], results["vacf"]["vacf"])),
                header="time_fs vacf",
            )
            saved["vacf"] = path

        if "structure_factor" in results:
            path = output_dir / "structure_factor.dat"
            np.savetxt(
                path,
                np.column_stack(
                    (
                        results["structure_factor"]["k"],
                        results["structure_factor"]["S_k"],
                    )
                ),
                header="k_1_per_A S_k",
            )
            saved["structure_factor"] = path

        summary_path = output_dir / "analysis_summary.json"

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": results.get("metadata", {}),
                    "summary": results.get("summary", {}),
                    "thermo": results.get("thermo", {}),
                    "coordination_number": results.get("coordination_number", {}),
                    "heat_capacity": results.get("heat_capacity", {}),
                    "diffusion_msd": results.get("diffusion_msd", {}),
                    "diffusion_vacf": results.get("diffusion_vacf", {}),
                    "runtime_seconds": results.get("runtime_seconds"),
                    "simulation_time_fs": results.get("simulation_time_fs"),
                },
                f,
                indent=4,
                default=_json_safe,
            )

        saved["summary_json"] = summary_path

        report_path = output_dir / "analysis_report.txt"
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            print_analysis_report(results)

        report_path.write_text(buffer.getvalue(), encoding="utf-8")
        saved["report"] = report_path

    if save_trajectory and "trajectory" in results:
        traj = results["trajectory"]
        meta = results["metadata"]
        symbol = meta.get("metal", "X")
        box = np.asarray(
            [
                meta["nx"] * meta["lattice_constant"],
                meta["ny"] * meta["lattice_constant"],
                meta["nz"] * meta["lattice_constant"],
            ],
            dtype=float,
        )

        npz_path = output_dir / "trajectory.npz"
        xyz_path = output_dir / "trajectory.xyz"
        final_xyz_path = output_dir / "final.xyz"

        np.savez_compressed(npz_path, **traj)

        write_xyz_trajectory(
            xyz_path,
            traj["positions_traj"],
            symbol=symbol,
            box=box,
            steps=traj.get("step_traj"),
            times=traj.get("time_traj"),
        )

        write_xyz_snapshot(
            final_xyz_path,
            traj["positions_traj"][-1],
            symbol=symbol,
            box=box,
            step=traj.get("step_traj", [None])[-1],
            time=traj.get("time_traj", [None])[-1],
        )

        saved["trajectory_npz"] = npz_path
        saved["trajectory_xyz"] = xyz_path
        saved["final_xyz"] = final_xyz_path

    return saved


def _line(width=58):
    print("-" * width)


def _section(title, width=58):
    print()
    print(title)
    _line(width)


def print_analysis_report(results):
    meta = results["metadata"]
    runtime = results["runtime_seconds"]
    runtime_str = f"{runtime:.2f} s" if runtime < 60 else f"{runtime / 60:.2f} min"

    sim_time_fs = results["simulation_time_fs"]
    sim_time_ps = sim_time_fs / 1000.0

    print()
    print("=" * 58)
    print("FCC Molecular Dynamics Analysis Report")
    print("=" * 58)

    _section("System")
    print(f"{'Metal':28s}: {meta['metal']}")
    print(f"{'Lattice size':28s}: {meta['nx']} x {meta['ny']} x {meta['nz']}")
    print(f"{'Atoms':28s}: {meta['N']}")
    print(f"{'Ensemble':28s}: {meta['ensemble'].upper()}")
    print(f"{'Target temperature':28s}: {meta['T0']:.2f} K")
    print(f"{'Production time step':28s}: {meta['dt']:.4f} fs")
    print(f"{'Simulation time':28s}: {sim_time_fs:.4f} fs ({sim_time_ps:.4f} ps)")
    print(f"{'Wall runtime':28s}: {runtime_str}")
    print(f"{'Equilibration steps':28s}: {meta['n_equil_steps']}")
    print(f"{'Production steps':28s}: {meta['n_steps']}")
    print(f"{'Samples':28s}: {meta['n_samples']}")
    print(f"{'Lattice constant':28s}: {meta['lattice_constant']:.6f} Å")
    print(f"{'Cutoff radius':28s}: {meta['rcut']:.6f} Å")

    if "thermo" in results:
        t = results["thermo"]
        _section("Thermodynamics")
        print(f"{'Mean Temperature':28s}: {t['mean_temperature']:.4f} K")
        print(f"{'Std Temperature':28s}: {t['std_temperature']:.4f} K")
        print(f"{'Mean Pressure':28s}: {t['mean_pressure']:.6e} eV/Å³")
        print(f"{'Std Pressure':28s}: {t['std_pressure']:.6e} eV/Å³")
        print(f"{'Mean Total Energy':28s}: {t['mean_total_energy']:.6e} eV")
        print(f"{'Std Total Energy':28s}: {t['std_total_energy']:.6e} eV")

    if "coordination_number" in results or "rdf" in results or "structure_factor" in results:
        _section("Structure")

        if "rdf" in results:
            rdf = results["rdf"]
            i_peak = int(np.argmax(rdf["g_r"]))
            print(f"{'RDF bins':28s}: {rdf['n_bins']}")
            print(f"{'RDF r_max':28s}: {rdf['r_max']:.4f} Å")
            print(f"{'RDF largest peak at':28s}: {rdf['r'][i_peak]:.4f} Å")

        if "coordination_number" in results:
            cn = results["coordination_number"]
            print(f"{'Coordination Number':28s}: {cn['coordination_number']:.4f}")
            print(f"{'CN cutoff radius':28s}: {cn['r_cut_cn']:.4f} Å")
            print(f"{'FCC reference CN':28s}: {cn['expected_fcc']:.0f}")

        if "structure_factor" in results:
            sk = results["structure_factor"]
            i_peak = int(np.argmax(sk["S_k"]))
            print(f"{'S(k) points':28s}: {len(sk['k'])}")
            print(f"{'S(k) largest peak at':28s}: {sk['k'][i_peak]:.4f} 1/Å")

    if "msd" in results or "vacf" in results or "diffusion_msd" in results or "diffusion_vacf" in results:
        _section("Dynamics / Transport")

        if "msd" in results:
            msd = results["msd"]
            print(f"{'MSD(t_final)':28s}: {msd['msd'][-1]:.6e} Å²")

        if "vacf" in results:
            vacf = results["vacf"]
            print(f"{'VACF(0)':28s}: {vacf['vacf'][0]:.6e}")
            print(f"{'VACF(t_final)':28s}: {vacf['vacf'][-1]:.6e}")

        if "diffusion_msd" in results:
            d = results["diffusion_msd"]
            print(f"{'D from MSD':28s}: {d['D']:.6e} Å²/fs")
            print(f"{'MSD slope':28s}: {d['slope']:.6e} Å²/fs")

        if "diffusion_vacf" in results:
            d = results["diffusion_vacf"]
            print(f"{'D from VACF':28s}: {d['D']:.6e} Å²/fs")
            print(f"{'VACF integral':28s}: {d['integral']:.6e}")

    if "heat_capacity" in results:
        _section("Fluctuation Quantities")
        cv = results["heat_capacity"]
        print(f"{'Heat Capacity Cv':28s}: {cv['Cv']:.6e} eV/K")
        print(f"{'Mean T used for Cv':28s}: {cv['T_mean']:.4f} K")

    _section("Output")
    print(f"{'Analyses requested':28s}: {', '.join(meta['analyses_requested'])}")

    if results.get("saved_files"):
        print()
        print(f"{'Saved files':28s}:")
        print("-" * 50)
        for name, path in results["saved_files"].items():
            print(f"  {name:26s}: {path}")

    print("=" * 58)
    print()