import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import font as tkfont
import threading
import queue

import numpy as np
import matplotlib.pyplot as plt

from md.system import System
from md.constants import get_lattice_constant, get_mass_internal, get_amu, get_sigma, get_eps
from md.forces import lj_forces
from md.lattice import make_fcc_lattice
from md.integrator import step_nve, step_nvt_berendsen
from md.utils import write_xyz
from md.analysis import (
    compute_rdf,
    compute_msd,
    compute_vacf,
    compute_pressure,
    compute_heat_capacity_from_energy,
    compute_coordination_from_rdf,
    compute_structure_factor,
    compute_diffusion_from_msd,
    compute_diffusion_from_vacf,
)

kB = 8.617333262145e-5  # eV/K


def initialize_velocities(system, T, seed=123):
    rng = np.random.default_rng(seed)
    N = system.N
    m = system.mass

    if np.isscalar(m):
        std = np.sqrt(kB * T / m)
    else:
        std = np.sqrt(kB * T / m)[:, None]

    v = rng.normal(0.0, 1.0, size=(N, 3)) * std
    v -= v.mean(axis=0)
    system.vel = v
    system.kinetic_energy()


class MDGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FCC Molecular Dynamics")

        # To start maximized on Windows; otherwise use a big default size
        #try:
            #self.state("zoomed")
        #except tk.TclError:
            #self.geometry("1000x700")

        self.sim_thread = None
        self.log_queue = queue.Queue()

        self._build_widgets()
        # periodically pull log messages from worker thread
        self.after(100, self._poll_log_queue)

    # --------------------------------------------------------
    # UI layout 
    # --------------------------------------------------------
    def _build_widgets(self):
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill="both", expand=True)

        # Left column: parameters
        params_frame = ttk.LabelFrame(main_frame, text="Simulation parameters", padding=10)
        for i in range(2):
            params_frame.columnconfigure(i, pad=10)

        params_frame.pack(side="left", fill="y")

        row = 0

        # Metal
        ttk.Label(params_frame, text="Metal:").grid(row=row, column=0, sticky="w")
        self.metal_var = tk.StringVar(value="Cu")
        metals = ["Ag", "Al", "Au", "Cu", "Ni", "Pb", "Pd", "Pt"]
        self.metal_menu = ttk.Combobox(
            params_frame,
            textvariable=self.metal_var,
            values=metals,
            width=8,
            state="readonly",
        )
        self.metal_menu.grid(row=row, column=1, sticky="w")
        row += 1

        # nx, ny, nz
        for label, attr, default in [
            ("nx:", "nx_var", 3),
            ("ny:", "ny_var", 3),
            ("nz:", "nz_var", 3),
        ]:
            ttk.Label(params_frame, text=label).grid(row=row, column=0, sticky="w")
            var = tk.StringVar(value=str(default))
            setattr(self, attr, var)
            ttk.Entry(params_frame, textvariable=var, width=8).grid(row=row, column=1, sticky="w")
            row += 1

        # Temperature
        ttk.Label(params_frame, text="T target (K):").grid(row=row, column=0, sticky="w")
        self.T_var = tk.StringVar(value="300.0")
        ttk.Entry(params_frame, textvariable=self.T_var, width=8).grid(row=row, column=1, sticky="w")
        row += 1

        # Ensemble selector
        ttk.Label(params_frame, text="Ensemble:").grid(row=row, column=0, sticky="w")
        self.ensemble_var = tk.StringVar(value="NVT (Berendsen)")
        self.ensemble_menu = ttk.Combobox(
            params_frame,
            textvariable=self.ensemble_var,
            values=["NVT (Berendsen)", "NVE"],
            width=15,
            state="readonly",
        )
        self.ensemble_menu.grid(row=row, column=1, sticky="w")
        row += 1

        # dt
        ttk.Label(params_frame, text="dt (fs):").grid(row=row, column=0, sticky="w")
        self.dt_var = tk.StringVar(value="0.001")
        ttk.Entry(params_frame, textvariable=self.dt_var, width=8).grid(row=row, column=1, sticky="w")
        row += 1

        # Equil steps
        ttk.Label(params_frame, text="Equilibration steps:").grid(row=row, column=0, sticky="w")
        self.nsteps_equil_var = tk.StringVar(value="2000")
        ttk.Entry(params_frame, textvariable=self.nsteps_equil_var, width=8).grid(
            row=row, column=1, sticky="w"
        )
        row += 1

        # Production steps
        ttk.Label(params_frame, text="Production steps:").grid(row=row, column=0, sticky="w")
        self.nsteps_var = tk.StringVar(value="2000")
        ttk.Entry(params_frame, textvariable=self.nsteps_var, width=8).grid(row=row, column=1, sticky="w")
        row += 1

        # Sample interval
        ttk.Label(params_frame, text="Sample interval:").grid(row=row, column=0, sticky="w")
        self.sample_interval_var = tk.StringVar(value="10")
        ttk.Entry(params_frame, textvariable=self.sample_interval_var, width=8).grid(
            row=row, column=1, sticky="w"
        )
        row += 1

        # .xyz output interval
        ttk.Label(params_frame, text=".xyz interval:").grid(row=row, column=0, sticky="w")
        self.output_interval_var = tk.StringVar(value="100")
        ttk.Entry(params_frame, textvariable=self.output_interval_var, width=8).grid(
            row=row, column=1, sticky="w"
        )
        row += 1

        # Trajectory filename
        ttk.Label(params_frame, text="traj file:").grid(row=row, column=0, sticky="w")
        self.traj_var = tk.StringVar(value="traj_gui.xyz")
        ttk.Entry(params_frame, textvariable=self.traj_var, width=14).grid(
            row=row, column=1, sticky="w"
        )
        row += 1

        # Run button
        self.run_button = ttk.Button(params_frame, text="Run simulation", command=self.on_run_clicked)
        self.run_button.grid(row=row, column=0, columnspan=2, pady=(10, 0), sticky="ew")

        # Right column: log output
        log_frame = ttk.LabelFrame(main_frame, text="Log / summary", padding=5)
        log_frame.pack(side="right", fill="both", expand=True)

        # Use grid inside log_frame so we can add scrollbars
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        mono = tkfont.Font(family="Consolas", size=10)  # or any monospace available

        self.log_text = tk.Text(
            log_frame,
            wrap="none",          # no word wrapping, keep columns tidy
            state="disabled",
            font=mono,
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")

        # Scrollbars
        yscroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        xscroll = ttk.Scrollbar(log_frame, orient="horizontal", command=self.log_text.xview)
        self.log_text.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)

        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")

    # --------------------------------------------------------
    # Logging helpers
    # --------------------------------------------------------
    def _append_log(self, msg):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def log(self, msg):
        self._append_log(msg)

    def log_threadsafe(self, msg):
        self.log_queue.put(msg)

    def _poll_log_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self._append_log(msg)
        except queue.Empty:
            pass
        self.after(100, self._poll_log_queue)

    # --------------------------------------------------------
    # Button callback
    # --------------------------------------------------------
    def on_run_clicked(self):
        if self.sim_thread and self.sim_thread.is_alive():
            messagebox.showerror("Busy", "A simulation is already running.")
            return

        try:
            params = self._read_params()
        except ValueError as e:
            messagebox.showerror("Input error", str(e))
            return

        self.run_button.configure(state="disabled")

        self.log("Starting simulation...")
        self.log(
            f"Metal={params['metal']}, "
            f"lattice={params['nx']}x{params['ny']}x{params['nz']}, "
            f"T={params['T_target']} K, ensemble={params['ensemble']}"
        )

        self.sim_thread = threading.Thread(
            target=self._run_simulation,
            args=(params,),
            daemon=True,
        )
        self.sim_thread.start()

    # --------------------------------------------------------
    # Read parameters from widgets
    # --------------------------------------------------------
    def _read_params(self):
        try:
            nx = int(self.nx_var.get())
            ny = int(self.ny_var.get())
            nz = int(self.nz_var.get())
            T = float(self.T_var.get())
            dt = float(self.dt_var.get())
            nsteps_equil = int(self.nsteps_equil_var.get())
            nsteps = int(self.nsteps_var.get())
            sample_interval = int(self.sample_interval_var.get())
            output_interval = int(self.output_interval_var.get())
        except Exception:
            raise ValueError("Failed to parse one or more numeric parameters.")

        if nx <= 0 or ny <= 0 or nz <= 0:
            raise ValueError("nx, ny, nz must be positive integers.")
        if dt <= 0:
            raise ValueError("dt must be > 0.")
        if nsteps_equil < 0 or nsteps <= 0:
            raise ValueError("Steps must be non-negative / positive.")
        if sample_interval <= 0 or output_interval <= 0:
            raise ValueError("Intervals must be positive integers.")

        return {
            "metal": self.metal_var.get(),
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "T_target": T,
            "ensemble": self.ensemble_var.get(),
            "dt": dt,
            "nsteps_equil": nsteps_equil,
            "nsteps": nsteps,
            "sample_interval": sample_interval,
            "output_interval": output_interval,
            "traj_file": self.traj_var.get(),
        }

    # --------------------------------------------------------
    # Core MD + analysis (runs in worker thread)
    # --------------------------------------------------------
    def _run_simulation(self, p):
        try:
            metal = p["metal"]
            nx = p["nx"]
            ny = p["ny"]
            nz = p["nz"]
            T_target = p["T_target"]
            ensemble = p["ensemble"]
            dt = p["dt"]
            nsteps_equil = p["nsteps_equil"]
            nsteps = p["nsteps"]
            sample_interval = p["sample_interval"]
            output_interval = p["output_interval"]
            traj_file = p["traj_file"]

            a = get_lattice_constant(metal)
            mass = get_mass_internal(metal)
            epsilon = get_eps(metal)
            sigma = get_sigma(metal)
            rcut = 2.5 * sigma

            pos, box = make_fcc_lattice(a, nx, ny, nz)
            # Center the lattice inside the box

            pos, box = make_fcc_lattice(a, nx, ny, nz)

            # --- Proper centering: move geometric center to the middle of the box ---
            center_now = np.mean(pos, axis=0)
            center_target = box / 2.0
            pos += (center_target - center_now)



            system = System(pos, mass, box, symbol=metal, cutoff=rcut, skin=0.3)

            initialize_velocities(system, T_target)
            system.remove_drift()

            pe0 = system.compute_forces(
                lambda pos, box, pairs: lj_forces(pos, box, pairs,
                                                  epsilon=epsilon, sigma=sigma, rcut=rcut)
            )
            system.potential_energy = pe0

            # Equilibration
            if ensemble.startswith("NVT"):
                self.log_threadsafe(f"Equilibrating for {nsteps_equil} steps in {ensemble}...")
            else:
                self.log_threadsafe(f"Equilibrating in NVT (Berendsen) for {nsteps_equil} steps before switching to NVE...")
            for _ in range(nsteps_equil):
                step_nvt_berendsen(
                    system, dt, T_target, 100 * dt,
                    epsilon=epsilon, sigma=sigma, rcut=rcut
                )


            write_xyz(system, step=0, filename=traj_file)

            # Production run
            n_samples = nsteps // sample_interval + 1
            N_atoms = system.N

            positions_traj = np.zeros((n_samples, N_atoms, 3))
            velocities_traj = np.zeros((n_samples, N_atoms, 3))
            pressure_traj = np.zeros(n_samples)
            energy_traj = np.zeros(n_samples)
            temp_traj = np.zeros(n_samples)

            sample_idx = 0
            positions_traj[0] = system.pos.copy()
            velocities_traj[0] = system.vel.copy()
            pressure_traj[0] = compute_pressure(system)
            KE0 = system.kinetic_energy()
            PE0 = system.potential_energy
            energy_traj[0] = KE0 + PE0
            temp_traj[0] = system.temperature()

            self.log_threadsafe("Starting production run...")
            self.log_threadsafe("# step        KE (eV)        PE (eV)     E_tot (eV)      T (K)")

            for step in range(1, nsteps + 1):
                if ensemble.startswith("NVT"):
                    step_nvt_berendsen(
                        system, dt, T_target, 100 * dt,
                        epsilon=epsilon, sigma=sigma, rcut=rcut
                    )
                else:
                    step_nve(system, dt, epsilon=epsilon, sigma=sigma, rcut=rcut)

                if step % output_interval == 0:
                    write_xyz(system, step, traj_file)

                if step % sample_interval == 0:
                    sample_idx += 1
                    positions_traj[sample_idx] = system.pos.copy()
                    velocities_traj[sample_idx] = system.vel.copy()
                    pressure_traj[sample_idx] = compute_pressure(system)

                    KE = system.kinetic_energy()
                    PE = system.potential_energy
                    energy_traj[sample_idx] = KE + PE
                    temp_traj[sample_idx] = system.temperature()

                if step % max(1, nsteps // 10) == 0:
                    KE = system.kinetic_energy()
                    PE = system.potential_energy
                    Etot = KE + PE
                    Tcur = system.temperature()
                    self.log_threadsafe(
                        f"{step:6d}  {KE: .6e}  {PE: .6e}  {Etot: .6e}  {Tcur: .4f}"
                    )

            # Trim arrays
            positions_traj = positions_traj[: sample_idx + 1]
            velocities_traj = velocities_traj[: sample_idx + 1]
            pressure_traj = pressure_traj[: sample_idx + 1]
            energy_traj = energy_traj[: sample_idx + 1]
            temp_traj = temp_traj[: sample_idx + 1]

            dt_sample = dt * sample_interval

            # Analysis
            self.log_threadsafe("Computing RDF, MSD, VACF, CN, Cv, diffusion...")

            r_max = 0.45 * min(system.box)
            n_bins = 200
            r, g_r = compute_rdf(positions_traj, system.box, r_max, n_bins)

            t_msd, msd = compute_msd(positions_traj, system.box)
            t_msd *= dt_sample

            t_vacf, vacf = compute_vacf(velocities_traj)
            t_vacf *= dt_sample

            D_msd, _ = compute_diffusion_from_msd(t_msd, msd)
            D_vacf, _ = compute_diffusion_from_vacf(t_vacf, vacf)

            rho = system.N / system.volume()
            idx_peak = np.argmax(g_r)
            idx_min = idx_peak + np.argmin(g_r[idx_peak:])
            r_cn = r[idx_min]
            CN = compute_coordination_from_rdf(r, g_r, rho, r_cn)

            E_tail = energy_traj[len(energy_traj) // 2:]
            T_mean = temp_traj[len(temp_traj) // 2:].mean()
            Cv = compute_heat_capacity_from_energy(E_tail, T_mean)

            k_vals = np.linspace(0.1, 12.0, 300)
            k_vals, S_k = compute_structure_factor(k_vals, r, g_r, rho)

            # Summary
            self.log_threadsafe("")
            self.log_threadsafe("=== Summary ===")
            self.log_threadsafe(f"Mean T: {T_mean:.3f} K")
            self.log_threadsafe(f"Mean P: {np.mean(pressure_traj):.5e} eV/Å³")
            self.log_threadsafe(f"CN (FCC ideal ~12): {CN:.3f}")
            self.log_threadsafe(f"Cv: {Cv:.5e} eV/K")
            self.log_threadsafe(f"D (MSD):  {D_msd:.5e}")
            self.log_threadsafe(f"D (VACF): {D_vacf:.5e}")
            self.log_threadsafe("=================")

            # Finish up on main thread
            def finish_on_main():
                self._make_plots(
                    r,
                    g_r,
                    r_cn,
                    t_msd,
                    msd,
                    t_vacf,
                    vacf,
                    k_vals,
                    S_k,
                    temp_traj,
                    pressure_traj,
                )
                self.run_button.configure(state="normal")
                messagebox.showinfo("Done", "Simulation and analysis complete.\nPlots have been generated.")

            self.after(0, finish_on_main)
            self.log_threadsafe("Simulation and analysis complete.")

        except Exception as e:
            # Capture the message BEFORE leaving the except block
            err_msg = f"{e}"
            self.log_threadsafe(f"Error: {err_msg}")

            def err():
                # Now we use err_msg, which is safe to capture
                messagebox.showerror("Error", f"Simulation failed:\n{err_msg}")
                self.run_button.configure(state="normal")

            self.after(0, err)


    # --------------------------------------------------------
    # Plot helper
    # --------------------------------------------------------
    def _make_plots(
        self,
        r,
        g_r,
        r_cn,
        t_msd,
        msd,
        t_vacf,
        vacf,
        k_vals,
        S_k,
        temp_traj,
        pressure_traj,
    ):
        plt.figure(figsize=(7, 5))
        plt.plot(r, g_r)
        plt.xlabel("r (Å)")
        plt.ylabel("g(r)")
        plt.title("RDF")
        plt.axvline(r_cn, ls="--", alpha=0.6)
        plt.tight_layout()

        plt.figure(figsize=(7, 5))
        plt.plot(t_msd, msd)
        plt.xlabel("t")
        plt.ylabel("MSD (Å²)")
        plt.title("MSD")
        plt.tight_layout()

        # Normalize VACF
        vacf_norm = vacf / vacf[0]

        plt.figure(figsize=(7,5))
        plt.plot(t_vacf, vacf_norm)
        plt.xlabel("t")
        plt.ylabel("Normalized VACF")
        plt.title("Normalized VACF")
        plt.tight_layout()


        plt.figure(figsize=(7, 5))
        plt.plot(k_vals, S_k)
        plt.xlabel("k (1/Å)")
        plt.ylabel("S(k)")
        plt.title("Structure Factor")
        plt.tight_layout()

        plt.figure(figsize=(7, 5))
        plt.plot(temp_traj)
        plt.xlabel("sample index")
        plt.ylabel("T (K)")
        plt.title("Temperature")
        plt.tight_layout()

        plt.figure(figsize=(7, 5))
        plt.plot(pressure_traj)
        plt.xlabel("sample index")
        plt.ylabel("P (eV/Å³)")
        plt.title("Pressure")
        plt.tight_layout()

        plt.show()


if __name__ == "__main__":
    app = MDGUI()
    app.mainloop()
