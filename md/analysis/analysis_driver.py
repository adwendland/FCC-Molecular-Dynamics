import time
import numpy as np

from md.constants import get_lattice_constant, get_sigma, get_eps, get_mass_internal
from md.lattice import make_fcc_lattice
from md.system import System
from md.integrator import step_nve, step_nvt_berendsen
from md.utils import write_xyz

from .rdf import (
    compute_rdf,
    compute_coordination_number,
)

from .structure_factor import (
    compute_structure_factor,
)

from .msd import (
    compute_msd,
)

from .vacf import (
    compute_vacf
)

from .transport import (
    compute_diffusion_from_msd,
    compute_diffusion_from_vacf,
)

from .thermodynamics import (
    compute_pressure,
    compute_heat_capacity_from_energy,
)

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

        center_now = np.mean(positions, axis=0)
        center_target = box / 2.0
        positions += center_target - center_now
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
    energy_traj = np.zeros(n_samples)
    temp_traj = np.zeros(n_samples)
    step_traj = np.zeros(n_samples, dtype=int)

    system.update_energies()
    positions_traj[0] = system.pos.copy()
    velocities_traj[0] = system.vel.copy()
    pressure_traj[0] = compute_pressure(system)
    energy_traj[0] = system.total_energy
    temp_traj[0] = system.temperature()

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
            energy_traj[sample_idx] = system.total_energy
            temp_traj[sample_idx] = system.temperature()
            step_traj[sample_idx] = step

    return {
        "positions_traj": positions_traj[: sample_idx + 1],
        "velocities_traj": velocities_traj[: sample_idx + 1],
        "pressure_traj": pressure_traj[: sample_idx + 1],
        "energy_traj": energy_traj[: sample_idx + 1],
        "temp_traj": temp_traj[: sample_idx + 1],
        "step_traj": step_traj[: sample_idx + 1],
        "time_traj": step_traj[: sample_idx + 1] * dt,
    }


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
    return results


def save_analysis_data(results, prefix="analysis"):
    saved = []

    if "rdf" in results:
        filename = f"{prefix}_rdf.dat"
        np.savetxt(filename, np.column_stack((results["rdf"]["r"], results["rdf"]["g_r"])))
        saved.append(filename)

    if "msd" in results:
        filename = f"{prefix}_msd.dat"
        np.savetxt(filename, np.column_stack((results["msd"]["time"], results["msd"]["msd"])))
        saved.append(filename)

    if "vacf" in results:
        filename = f"{prefix}_vacf.dat"
        np.savetxt(filename, np.column_stack((results["vacf"]["time"], results["vacf"]["vacf"])))
        saved.append(filename)

    if "structure_factor" in results:
        filename = f"{prefix}_structure_factor.dat"
        np.savetxt(
            filename,
            np.column_stack((results["structure_factor"]["k"], results["structure_factor"]["S_k"])),
        )
        saved.append(filename)

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
    print("=" * 58)
    print()