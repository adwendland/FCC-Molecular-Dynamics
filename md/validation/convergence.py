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

    for factor in [1, 2, 4, 8]:
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
    dt8 = dt / 8

    x1 = trajectories[dt1]["pos"]
    x2 = trajectories[dt2]["pos"]
    x4 = trajectories[dt4]["pos"]
    x8 = trajectories[dt8]["pos"]
    box = trajectories[dt8]["box"]

    e1 = rms_position_error(x1, x8, box)
    e2 = rms_position_error(x2, x8, box)
    e4 = rms_position_error(x4, x8, box)

    dt_values = np.array([dt, dt/2, dt/4])
    errors = np.array([e1, e2, e4])

    order, intercept = np.polyfit(np.log(dt_values), np.log(errors), 1)

    return {
        "order": order,
        "errors": {
            dt1: e1,
            dt2: e2,
            dt4: e4,
        },
        "reference_dt": dt8,
    }