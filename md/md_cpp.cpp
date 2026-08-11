#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>

namespace py = pybind11;
using index_t = py::ssize_t;


// -------------------------------------------------------------
// Minimum-image PBC
// -------------------------------------------------------------
inline void pbc_diff(
    const double *ri,
    const double *rj,
    const double *box,
    double *dr_out
)
{
    for (int k = 0; k < 3; ++k) {
        double dr = ri[k] - rj[k];
        dr -= std::round(dr / box[k]) * box[k];
        dr_out[k] = dr;
    }
}


// -------------------------------------------------------------
// Lennard-Jones forces
// pos:   (N, 3)
// force: (N, 3)
// box:   (3,)
// pairs: (M, 2)
//
// Returns:
//     (potential_energy, virial)
//
// Virial convention:
//     W = sum_{i<j} r_ij · F_ij
// -------------------------------------------------------------
py::tuple lj_forces_cpp(
    py::array_t<double> pos_in,
    py::array_t<double> force_in,
    py::array_t<double> box_in,
    py::array_t<long long> pairs_in,
    double epsilon,
    double sigma,
    double rcut
)
{
    auto pos   = pos_in.unchecked<2>();
    auto force = force_in.mutable_unchecked<2>();
    auto box   = box_in.unchecked<1>();
    auto pairs = pairs_in.unchecked<2>();

    const index_t N = pos.shape(0);
    const index_t M = pairs.shape(0);

    // Zero forces
    for (index_t i = 0; i < N; ++i) {
        for (int k = 0; k < 3; ++k) {
            force(i, k) = 0.0;
        }
    }

    const double rcut2 = rcut * rcut;
    const double sig2 = sigma * sigma;

    const double box_arr[3] = {
        box(0),
        box(1),
        box(2)
    };

    double dr[3];
    double pe = 0.0;
    double virial = 0.0;

    // Loop over unique neighbor pairs
    for (index_t p = 0; p < M; ++p) {
        const index_t i = pairs(p, 0);
        const index_t j = pairs(p, 1);

        const double ri[3] = {
            pos(i, 0),
            pos(i, 1),
            pos(i, 2)
        };

        const double rj[3] = {
            pos(j, 0),
            pos(j, 1),
            pos(j, 2)
        };

        pbc_diff(ri, rj, box_arr, dr);

        const double r2 =
            dr[0] * dr[0]
            + dr[1] * dr[1]
            + dr[2] * dr[2];

        if (r2 <= 0.0 || r2 >= rcut2) {
            continue;
        }

        const double inv_r2_sigma = sig2 / r2;
        const double inv_r6 =
            inv_r2_sigma * inv_r2_sigma * inv_r2_sigma;
        const double inv_r12 = inv_r6 * inv_r6;

        // Potential energy
        const double v_ij =
            4.0 * epsilon * (inv_r12 - inv_r6);

        pe += v_ij;

        // Force vector on atom i due to atom j
        const double fij_over_r2 =
            24.0 * epsilon
            * (2.0 * inv_r12 - inv_r6)
            / r2;

        const double fx = fij_over_r2 * dr[0];
        const double fy = fij_over_r2 * dr[1];
        const double fz = fij_over_r2 * dr[2];

        force(i, 0) += fx;
        force(i, 1) += fy;
        force(i, 2) += fz;

        force(j, 0) -= fx;
        force(j, 1) -= fy;
        force(j, 2) -= fz;

        // Pair virial r_ij · F_ij
        virial += dr[0] * fx + dr[1] * fy + dr[2] * fz;
    }

    return py::make_tuple(pe, virial);
}


// -------------------------------------------------------------
// Velocity-Verlet
//
// Returns:
//     (potential_energy, virial)
// at the new positions.
// -------------------------------------------------------------
py::tuple velocity_verlet_lj_cpp(
    py::array_t<double> pos_in,
    py::array_t<double> vel_in,
    py::array_t<double> force_in,
    py::array_t<double> box_in,
    py::array_t<long long> pairs_in,
    double mass,
    double dt,
    double epsilon,
    double sigma,
    double rcut
)
{
    auto pos   = pos_in.mutable_unchecked<2>();
    auto vel   = vel_in.mutable_unchecked<2>();
    auto force = force_in.mutable_unchecked<2>();
    auto box   = box_in.unchecked<1>();

    const index_t N = pos.shape(0);

    // 1) Compute forces at current positions
    py::tuple old_result = lj_forces_cpp(
        pos_in,
        force_in,
        box_in,
        pairs_in,
        epsilon,
        sigma,
        rcut
    );

    // Old PE and virial are not needed after the first half-step
    (void)old_result;

    const double half_dt = 0.5 * dt;
    const double inv_m = 1.0 / mass;

    // 2) v(t + dt/2), x(t + dt)
    for (index_t i = 0; i < N; ++i) {
        for (int k = 0; k < 3; ++k) {
            vel(i, k) += half_dt * force(i, k) * inv_m;
            pos(i, k) += dt * vel(i, k);

            // Wrap positions into [0, L)
            const double L = box(k);
            double x = pos(i, k);
            x -= std::floor(x / L) * L;
            pos(i, k) = x;
        }
    }

    // 3) Compute forces and virial at new positions
    py::tuple new_result = lj_forces_cpp(
        pos_in,
        force_in,
        box_in,
        pairs_in,
        epsilon,
        sigma,
        rcut
    );

    // 4) v(t + dt)
    for (index_t i = 0; i < N; ++i) {
        for (int k = 0; k < 3; ++k) {
            vel(i, k) += half_dt * force(i, k) * inv_m;
        }
    }

    return new_result;
}


// -------------------------------------------------------------
// PYBIND11 MODULE
// -------------------------------------------------------------
PYBIND11_MODULE(md_cpp, m)
{
    m.doc() = "C++-accelerated MD integrator and LJ forces";

    m.def(
        "lj_forces_cpp",
        &lj_forces_cpp,
        py::arg("pos"),
        py::arg("force"),
        py::arg("box"),
        py::arg("pairs"),
        py::arg("epsilon"),
        py::arg("sigma"),
        py::arg("rcut")
    );

    m.def(
        "velocity_verlet_lj_cpp",
        &velocity_verlet_lj_cpp,
        py::arg("pos"),
        py::arg("vel"),
        py::arg("force"),
        py::arg("box"),
        py::arg("pairs"),
        py::arg("mass"),
        py::arg("dt"),
        py::arg("epsilon"),
        py::arg("sigma"),
        py::arg("rcut")
    );
}