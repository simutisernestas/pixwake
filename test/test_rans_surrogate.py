import time

import jax
import jax.numpy as jnp
import numpy as onp
from py_wake.examples.data.dtu10mw import DTU10MW

from pixwake import simulate_case_rans, ws2aep


from jax.test_util import check_grads


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
    WSS = onp.random.uniform(CUTIN_WS, CUTOUT_WS, T)
    WDS = onp.random.uniform(0, 360, T)

    wi, le = 10, 8
    xs, ys = jnp.meshgrid(  # example positions
        jnp.linspace(0, wi * 3 * D, wi),
        jnp.linspace(0, le * 3 * D, le),
    )
    xs, ys = xs.ravel(), ys.ravel()
    assert xs.shape[0] == (wi * le), xs.shape

    def aep(xx, yy):
        def to_be_mapped(wr):
            return simulate_case_rans(
                xx,
                yy,
                wr[0],
                wr[1],
                D,
                jnp.stack([ct_xp, ct_fp], axis=1),
            )

        wind_resource = jnp.stack([WSS, WDS], axis=1)
        effective_wss = jax.lax.map(to_be_mapped, wind_resource)
        return ws2aep(effective_wss, jnp.stack([ct_xp, pw_fp], axis=1))

    aep_and_grad = jax.jit(jax.value_and_grad(aep, argnums=(0, 1)))

    def block_all(res):
        if isinstance(res, tuple):
            return tuple(block_all(r) for r in res)
        else:
            return res.block_until_ready()

    res = aep_and_grad(jnp.asarray(xs), jnp.asarray(ys))
    block_all(res)
    s = time.time()
    res = aep_and_grad(jnp.asarray(xs), jnp.asarray(ys))
    block_all(res)
    print(f"AEP: {res[0]} in {time.time() - s:.3f} seconds")

    assert jnp.isfinite(res[0]).all(), "AEP should be finite"
    assert jnp.isfinite(res[1][0]).all(), "Gradient of x should be finite"
    assert jnp.isfinite(res[1][1]).all(), "Gradient of y should be finite"


import pytest


@pytest.mark.xfail(reason="Gradients of the RANS model are not yet correct.")
def test_rans_surrogate_gradients():
    ct_xp, ct_fp, _, D = get_rans_dependencies()
    ws = 9.0
    wd = 90.0
    wi, le = 3, 2
    xs, ys = jnp.meshgrid(
        jnp.linspace(0, wi * 3 * D, wi),
        jnp.linspace(0, le * 3 * D, le),
    )
    xs, ys = xs.ravel(), ys.ravel()

    def sim(x, y):
        return simulate_case_rans(
            x, y, ws, wd, D, jnp.stack([ct_xp, ct_fp], axis=1)
        ).sum()

    check_grads(sim, (xs, ys), order=1, modes=["rev"], atol=1e-2, rtol=1e-2)
