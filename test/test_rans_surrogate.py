import time

import jax
import jax.numpy as jnp
import numpy as onp
from jax.test_util import check_grads
from py_wake.examples.data.dtu10mw import DTU10MW

from pixwake import Curve, RANSModel, Turbine, WakeSimulation


def get_rans_dependencies():
    turbine = DTU10MW()
    ct_xp = turbine.powerCtFunction.ws_tab
    ct_fp = turbine.powerCtFunction.power_ct_tab[1, :]
    pw_fp = turbine.powerCtFunction.power_ct_tab[0, :]
    D = turbine.diameter()
    return ct_xp, ct_fp, pw_fp, D


def test_rans_surrogate_aep():
    ct_xp, ct_fp, pw_fp, D = get_rans_dependencies()
    CUTOUT_WS = 25.0
    CUTIN_WS = 3.0

    onp.random.seed(42)
    T = 10
    WSS = jnp.asarray(onp.random.uniform(CUTIN_WS, CUTOUT_WS, T))
    WDS = jnp.asarray(onp.random.uniform(0, 360, T))

    wi, le = 10, 8
    xs, ys = jnp.meshgrid(  # example positions
        jnp.linspace(0, wi * 3 * D, wi),
        jnp.linspace(0, le * 3 * D, le),
    )
    xs, ys = xs.ravel(), ys.ravel()
    assert xs.shape[0] == (wi * le), xs.shape

    turbine = Turbine(
        rotor_diameter=D,
        hub_height=100.0,
        power_curve=Curve(wind_speed=ct_xp, values=pw_fp),
        ct_curve=Curve(wind_speed=ct_xp, values=ct_fp),
    )

    def aep(xx, yy):
        model = RANSModel(ambient_ti=0.1)
        sim = WakeSimulation(model, mapping_strategy="map", fpi_damp=0.8, fpi_tol=1e-3)
        return sim(xx, yy, WSS, WDS, turbine).aep()

    aep_and_grad = jax.jit(jax.value_and_grad(aep, argnums=(0, 1)))

    def block_all(res):
        if isinstance(res, tuple):
            return tuple(block_all(r) for r in res)
        else:
            return res.block_until_ready()

    res = aep_and_grad(xs, ys)
    block_all(res)
    s = time.time()
    res = aep_and_grad(xs, ys)
    block_all(res)
    print(f"AEP: {res[0]} in {time.time() - s:.3f} seconds")

    assert jnp.isfinite(res[0]).all(), "AEP should be finite"
    assert jnp.isfinite(res[1][0]).all(), "Gradient of x should be finite"
    assert jnp.isfinite(res[1][1]).all(), "Gradient of y should be finite"


def test_rans_surrogate_gradients():
    ct_xp, ct_fp, pw_fp, D = get_rans_dependencies()
    ws = 9.0
    wd = 90.0
    wi, le = 3, 2
    xs, ys = jnp.meshgrid(
        jnp.linspace(0, wi * 3 * D, wi),
        jnp.linspace(0, le * 3 * D, le),
    )
    xs, ys = xs.ravel(), ys.ravel()

    turbine = Turbine(
        rotor_diameter=D,
        hub_height=100.0,
        power_curve=Curve(wind_speed=ct_xp, values=pw_fp),
        ct_curve=Curve(wind_speed=ct_xp, values=ct_fp),
    )

    def sim(x, y):
        model = RANSModel(ambient_ti=0.1)
        simulation = WakeSimulation(model, fpi_damp=0.8, fpi_tol=1e-3)
        return simulation(
            x,
            y,
            jnp.full_like(x, ws),
            jnp.full_like(x, wd),
            turbine,
        ).effective_wind_speed.sum()

    check_grads(sim, (xs, ys), order=1, modes=["rev"], atol=1e-2, rtol=1e-2, eps=10)
