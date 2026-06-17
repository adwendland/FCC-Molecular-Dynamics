import numpy as np
from md.integrator import step_nve


def minimum_image_displacement(dx, box):
    return dx - box * np.round(dx / box)


def rms_position_error(pos, ref_pos, box):
    dx = pos - ref_pos
    dx = minimum_image_displacement(dx, box)
    return np.sqrt(np.mean(np.sum(dx**2, axis=1)))


def test_timestep_refinement(
    system_builder_fn,
    dt,
    n_steps,
    epsilon,
    sigma,
    rcut,
):
    trajectories = {}

    for factor in [1, 2, 4]:
        dt_test = dt / factor
        system = system_builder_fn()

        for _ in range(n_steps * factor):
            step_nve(
                system,
                dt_test,
                epsilon=epsilon,
                sigma=sigma,
                rcut=rcut,
            )

        trajectories[dt_test] = {
            "pos": system.pos.copy(),
            "box": system.box.copy(),
        }

    dt1 = dt
    dt2 = dt / 2
    dt4 = dt / 4

    x1 = trajectories[dt1]["pos"]
    x2 = trajectories[dt2]["pos"]
    x4 = trajectories[dt4]["pos"]
    box = trajectories[dt4]["box"]

    e1 = rms_position_error(x1, x4, box)
    e2 = rms_position_error(x2, x4, box)

    if e1 > 0.0 and e2 > 0.0:
        order = np.log(e1 / e2) / np.log(2.0)
    else:
        order = np.nan

    return {
        "order": order,
        "errors": {
            dt1: e1,
            dt2: e2,
        },
        "reference_dt": dt4,
    }