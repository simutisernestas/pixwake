import jax.numpy as jnp
import pytest

from pixwake import Curve, Turbine
from pixwake.core import WakeSimulation
from pixwake.deficit import NiayifarGaussianDeficit
from pixwake.turbulence.crespo import CrespoHernandez

"""TODO: should be more comprehensive !!!"""


@pytest.fixture(params=[False, True], scope="module")
def wake_sim(request):
    turbine = Turbine(
        rotor_diameter=100.0,
        hub_height=100.0,
        power_curve=Curve(
            wind_speed=jnp.array([0.0, 20.0]), values=jnp.array([0.0, 3000.0])
        ),
        ct_curve=Curve(wind_speed=jnp.array([0.0, 20.0]), values=jnp.array([0.8, 0.8])),
    )
    model = NiayifarGaussianDeficit()

    if not request.param:
        sim = WakeSimulation(turbine, model)
    else:
        tmodel = CrespoHernandez()
        sim = WakeSimulation(turbine, model, tmodel)

    return sim


def test_turbulence_shape(wake_sim):
    n_turbines = 10
    xs = jnp.arange(n_turbines) * 500.0
    ys = jnp.zeros(n_turbines)
    ws = jnp.full(n_turbines, 10.0)
    wd = jnp.full(n_turbines, 270.0)
    ti = jnp.full(n_turbines, 0.1)

    # scalar ti
    _ = wake_sim(xs, ys, ws, wd, 0.1)
    # array ti
    _ = wake_sim(xs, ys, ws, wd, ti)

    with pytest.raises(AssertionError):
        # no ti
        _ = wake_sim(xs, ys, ws, wd)
    with pytest.raises(AssertionError):
        # wrong shape ti
        _ = wake_sim(xs, ys, ws, wd, jnp.array([0.1, 0.1]))


# TODO: check that sim throws proper errors with all deficit models!!
