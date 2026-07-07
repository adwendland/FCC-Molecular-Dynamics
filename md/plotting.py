# plotting.py

from pathlib import Path
import os

import matplotlib.pyplot as plt

from md.system import System
from md.analysis.analysis_driver import *


def _save_or_show(path=None, show=False):
    plt.tight_layout()

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches="tight")

    if show:
        plt.show(block=False)
    else:
        plt.close()


# ==========================================================
# Analysis plots
# ==========================================================

def plot_temperature(time, temperature, meta, path=None, show=False,):
    plt.figure(figsize=(6, 4))
    plt.plot(time, temperature)

    plt.xlabel("Time (fs)")
    plt.ylabel("Temperature (K)")
    plt.title(
        f"Temperature (K) vs Time (fs)\n"
        f"{meta['metal']} ({meta['nx']}×{meta['ny']}×{meta['nz']}) | "
        f"{meta['ensemble']} | "
        f"{meta['T0']:.0f} K | "
        f"{meta['time_ps']:g} ps"
    )

    _save_or_show(path, show)


def plot_energy(time, kinetic, potential, total, meta, path=None, show=False):
    plt.figure(figsize=(6, 4))
    plt.plot(time, kinetic, label="Kinetic")
    plt.plot(time, potential, label="Potential")
    plt.plot(time, total, label="Total")

    plt.xlabel("Time (fs)")
    plt.ylabel("Energy (eV)")
    plt.title(
        f"Energy (eV) vs Time (fs)\n"
        f"{meta['metal']} ({meta['nx']}×{meta['ny']}×{meta['nz']}) | "
        f"{meta['ensemble']} | "
        f"{meta['T0']:.0f} K | "
        f"{meta['time_ps']:g} ps"
    )
    plt.legend()

    _save_or_show(path, show)


def plot_rdf(r, g_r, meta, path=None, show=False):
    plt.figure(figsize=(6, 4))
    plt.plot(r, g_r)

    plt.xlabel(r"$r$ ($\AA$)")
    plt.ylabel(r"$g(r)$")
    plt.title(
        f"Radial Distribution Function\n"
        f"{meta['metal']} ({meta['nx']}×{meta['ny']}×{meta['nz']}) | "
        f"{meta['ensemble']} | "
        f"{meta['T0']:.0f} K | "
        f"{meta['time_ps']:g} ps"
    )

    _save_or_show(path, show)


def plot_structure_factor(k, S_k, meta, path=None, show=False):
    plt.figure(figsize=(6, 4))
    plt.plot(k, S_k)

    plt.xlabel(r"$k$")
    plt.ylabel(r"$S(k)$")
    plt.title(
        f"Structure Factor\n"
        f"{meta['metal']} ({meta['nx']}×{meta['ny']}×{meta['nz']}) | "
        f"{meta['ensemble']} | "
        f"{meta['T0']:.0f} K | "
        f"{meta['time_ps']:g} ps"
    )

    _save_or_show(path, show)


def plot_msd(time, msd, meta, path=None, show=False):
    plt.figure(figsize=(6, 4))
    plt.plot(time, msd)

    plt.xlabel("Time (fs)")
    plt.ylabel(r"MSD ($\AA^2$)")
    plt.title(
        f"Mean Squared Displacement\n"
        f"{meta['metal']} ({meta['nx']}×{meta['ny']}×{meta['nz']}) | "
        f"{meta['ensemble']} | "
        f"{meta['T0']:.0f} K | "
        f"{meta['time_ps']:g} ps"
    )

    _save_or_show(path, show)


def plot_vacf(time, vacf, meta, path=None, show=False):
    plt.figure(figsize=(6, 4))
    plt.plot(time, vacf)

    plt.xlabel("Time (fs)")
    plt.ylabel("VACF")
    plt.title(
        f"Velocity Autocorrelation Function\n"
        f"{meta['metal']} ({meta['nx']}×{meta['ny']}×{meta['nz']}) | "
        f"{meta['ensemble']} | "
        f"{meta['T0']:.0f} K | "
        f"{meta['time_ps']:g} ps"
    )

    _save_or_show(path, show)



# ==========================================================
# Validation plots
# ==========================================================

def plot_energy_conservation(time, relative_drift, tolerance, meta,
                             path=None, show=False):
    plt.figure(figsize=(6, 4))
    plt.plot(time, relative_drift, label="Relative energy drift")
    plt.axhline(tolerance, ls="--", c="r", lw=1.2, alpha=0.6, label="Tolerance")
    plt.axhline(-tolerance, ls="--", c="r", lw=1.2, alpha=0.6,)
    plt.xlabel("Time (fs)")
    plt.ylabel("Relative energy drift")
    plt.title(
        f"Energy Conservation\n"
        f"{meta['metal']} ({meta['nx']}×{meta['ny']}×{meta['nz']}) | "
        f"{meta['T0']:.0f} K | "
        f"{meta['time_ps']:g} ps"
    )
    plt.legend()

    _save_or_show(path, show)


def plot_timestep_convergence(dt_values, errors, order, meta,
                              path=None, show=False):
    plt.figure(figsize=(6, 4))
    plt.loglog(dt_values, errors, "o-", lw=2, ms=6, label="Measured")
    dt_ref = dt_values[-1]
    err_ref = errors[-1]

    ref1 = err_ref * (dt_values / dt_ref)
    ref2 = err_ref * (dt_values / dt_ref) ** 2

    plt.loglog(dt_values, ref1, "--", lw=1.2, label=r"$\mathcal{O}(\Delta t)$")
    plt.loglog(dt_values, ref2, "--", lw=1.2, label=r"$\mathcal{O}(\Delta t^2)$")
    plt.gca().invert_xaxis()

    plt.text(
        0.03, 0.05,
        f"Observed order = {order:.2f}",
        transform=plt.gca().transAxes,
        va="bottom",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="0.8"),
    )
    plt.xlabel("Time step (fs)")
    plt.ylabel("Position error")
    plt.title(
        f"Time Step Convergence\n"
        f"{meta['metal']} ({meta['nx']}×{meta['ny']}×{meta['nz']}) | "
        f"{meta['T0']:.0f} K"
    )
    plt.legend()

    _save_or_show(path, show)


def plot_temperature_stability(time, temperature, target_temperature, meta,
                               path=None, show=False):
    plt.figure(figsize=(6, 4))
    plt.plot(time, temperature)
    plt.axhline(target_temperature, ls="--", c="r",
                label=f"Target ({target_temperature:.0f} K)")
    plt.xlabel("Time (fs)")
    plt.ylabel("Temperature (K)")
    plt.title(
        f"Temperature Stability\n"
        f"{meta['metal']} ({meta['nx']}×{meta['ny']}×{meta['nz']}) | "
        f"{meta['T0']:.0f} K | "
        f"{meta['time_ps']:g} ps"
    )
    plt.legend()

    _save_or_show(path, show)


def plot_momentum_conservation(time, momentum_norm, meta,
                               path=None, show=False):
    plt.figure(figsize=(6, 4))
    plt.plot(time, momentum_norm)

    plt.xlabel("Time (fs)")
    plt.ylabel(r"$|P|$")
    plt.title(
        f"Total Momentum\n"
        f"{meta['metal']} ({meta['nx']}×{meta['ny']}×{meta['nz']}) | "
        f"{meta['T0']:.0f} K | "
        f"{meta['time_ps']:g} ps"
    )

    _save_or_show(path, show)


def plot_total_kinetic_energy(time, kinetic_energy, expected_kinetic_energy, meta,
                            path=None, show=False):
    plt.figure(figsize=(6, 4))

    plt.plot(time, kinetic_energy, label="Measured")
    plt.axhline(expected_kinetic_energy, linestyle="--", label="Expected")

    plt.xlabel("Time (fs)")
    plt.ylabel("Total kinetic energy (eV)")
    plt.title(
        f"Total Kinetic Energy\n"
        f"{meta['metal']} ({meta['nx']}×{meta['ny']}×{meta['nz']}) | "
        f"{meta['T0']:.0f} K | "
        f"{meta['time_ps']:g} ps"
    )
    plt.legend()

    _save_or_show(path, show)


def plot_component_equipartition(time, kinetic_x, kinetic_y, kinetic_z,
                                 expected_component_energy, meta,
                                 path=None, show=False):
    plt.figure(figsize=(6, 4))

    plt.plot(time, kinetic_x, label=r"$K_x$")
    plt.plot(time, kinetic_y, label=r"$K_y$")
    plt.plot(time, kinetic_z, label=r"$K_z$")
    plt.axhline(expected_component_energy, linestyle="--", label="Expected")

    plt.xlabel("Time (fs)")
    plt.ylabel("Component kinetic energy (eV)")
    plt.title(
        f"Component Equipartition\n"
        f"{meta['metal']} ({meta['nx']}×{meta['ny']}×{meta['nz']}) | "
        f"{meta['T0']:.0f} K | "
        f"{meta['time_ps']:g} ps"
    )
    plt.legend()

    _save_or_show(path, show)


# Backward-compatible alias. Prefer plot_component_equipartition for validation.
def plot_equipartition(time, kinetic_x, kinetic_y, kinetic_z,
                       expected_component_energy, meta,
                       path=None, show=False):
    plot_component_equipartition(
        time, kinetic_x, kinetic_y, kinetic_z, expected_component_energy, meta,
        path=path, show=show,
    )


def plot_rdf_validation(r, g, meta, expected_peak=None,
                        path=None, show=False):
    plt.figure(figsize=(6, 4))
    plt.plot(r, g)

    if expected_peak is not None:
        plt.axvline(expected_peak,
                    ls="--",
                    c="r",
                    label="Expected FCC peak")
        plt.legend()

    plt.xlabel(r"$r$ ($\AA$)")
    plt.ylabel("g(r)")
    plt.title(
        f"RDF Validation\n"
        f"{meta['metal']} ({meta['nx']}×{meta['ny']}×{meta['nz']}) | "
        f"{meta['T0']:.0f} K | "
        f"{meta['time_ps']:g} ps"
    )

    _save_or_show(path, show)