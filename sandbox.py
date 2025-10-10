import time
import jax.numpy as jnp
from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import (
    BastankhahGaussianDeficit,
    NiayifarGaussianDeficit,
    NOJDeficit,
)
from pixwake.turbulence import CrespoHernandez


def rect_grid_params(nx=3, ny=2):
    xs, ys = jnp.meshgrid(
        jnp.linspace(0.0, 500.0 * (nx - 1), nx), jnp.linspace(0.0, 500.0 * (ny - 1), ny)
    )
    xs = xs.ravel()
    ys = ys.ravel()
    ws = 10.0
    wd = 270.0
    k = 0.05
    ct_curve_array = jnp.stack([jnp.array([0.0, 20.0]), jnp.array([0.8, 0.8])], axis=1)
    power_curve_array = jnp.stack(
        [jnp.array([0.0, 20.0]), jnp.array([0.0, 3000.0])], axis=1
    )
    turbine = Turbine(
        rotor_diameter=100.0,
        hub_height=100.0,
        power_curve=Curve(
            wind_speed=power_curve_array[:, 0], values=power_curve_array[:, 1]
        ),
        ct_curve=Curve(wind_speed=ct_curve_array[:, 0], values=ct_curve_array[:, 1]),
    )
    return xs, ys, ws, wd, k, turbine


xs, ys, ws, wd, _, turbine = rect_grid_params(nx=5, ny=4)


for deficit_model, requires_ti in [
    (NOJDeficit(k=0.05), False),
    (BastankhahGaussianDeficit(), True),
    (NiayifarGaussianDeficit(), True),
]:
    s = time.time()
    turbulence_model = CrespoHernandez()
    turb_arg = (turbulence_model) if requires_ti else {}
    sim = WakeSimulation(turbine, deficit_model, *turb_arg, mapping_strategy="vmap")

    sim_args = {
        "wt_xs": xs,
        "wt_ys": ys,
        "ws_amb": jnp.full_like(xs, ws),
        "wd": jnp.full_like(xs, wd),
    }
    if requires_ti:
        sim_args["ti"] = 0.1

    res = sim(**sim_args)
    e = time.time()
    print(f"Simulation took {e - s:.2f} seconds")
    break
