# plot_analysis.py
#
# Plot RDF, MSD, VACF from MD simulation output files:
#   - rdf.dat
#   - msd.dat
#   - vacf.dat
#
# Run:  python plot_analysis.py

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Utility to load 2-column data safely
# ---------------------------------------------------------------------
def load_two_column_file(filename):
    try:
        data = np.loadtxt(filename)
        if data.ndim == 1:
            data = data[None, :]
        return data[:, 0], data[:, 1]
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None, None


# =============================
# 1. RDF plot
# =============================
r, g_r = load_two_column_file("rdf.dat")
if r is not None:
    plt.figure(figsize=(6,4))
    plt.plot(r, g_r, lw=2)
    plt.xlabel("r (Å)")
    plt.ylabel("g(r)")
    plt.title("Radial Distribution Function")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("rdf.png", dpi=300)
    print("Saved rdf.png")


# =============================
# 2. MSD plot
# =============================
t_msd, msd = load_two_column_file("msd.dat")
if t_msd is not None:
    plt.figure(figsize=(6,4))
    plt.plot(t_msd, msd, lw=2)
    plt.xlabel("t (simulation units)")
    plt.ylabel("MSD (Å$^2$)")
    plt.title("Mean Squared Displacement")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("msd.png", dpi=300)
    print("Saved msd.png")


# =============================
# 3. VACF plot
# =============================
t_vacf, vacf = load_two_column_file("vacf.dat")
if t_vacf is not None:
    plt.figure(figsize=(6,4))
    plt.plot(t_vacf, vacf, lw=2)
    plt.xlabel("t (simulation units)")
    plt.ylabel("VACF")
    plt.title("Velocity Autocorrelation Function")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("vacf.png", dpi=300)
    print("Saved vacf.png")

print("Done.")
