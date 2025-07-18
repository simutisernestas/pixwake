import time

import jax
import jax.numpy as jnp
import numpy as onp
from py_wake.examples.data.dtu10mw import DTU10MW

from pixwake import simulate_case_rans, ws2aep

if __name__ == "__main__":
    turbine = DTU10MW()

    ct_xp = turbine.powerCtFunction.ws_tab
    ct_fp = turbine.powerCtFunction.power_ct_tab[1, :]
    pw_fp = turbine.powerCtFunction.power_ct_tab[0, :]
    D = turbine.diameter()
    CUTOUT_WS = 25.0
    CUTIN_WS = 3.0

    onp.random.seed(42)
    T = 100
    ws = onp.random.uniform(CUTIN_WS + 1, CUTOUT_WS - 1, T)
    wd = onp.random.uniform(0, 360, T)

    wi, le = 5, 5
    xs, ys = jnp.meshgrid(  # example positions
        jnp.linspace(0, wi * 3 * D, wi),
        jnp.linspace(0, le * 3 * D, le),
    )
    xs, ys = xs.ravel(), ys.ravel()
    assert xs.shape[0] == (wi * le), xs.shape

    grad_fn = jax.jit(
        jax.value_and_grad(
            lambda xx, yy: ws2aep(
                simulate_case_rans(
                    xx,
                    yy,
                    jnp.atleast_1d(ws),
                    jnp.atleast_1d(wd),
                    D,
                    jnp.stack([ct_xp, ct_fp], axis=1),
                ),
                jnp.stack([ct_xp, pw_fp], axis=1),
            ),
            argnums=(0, 1),
        )
    )

    # with jax.disable_jit():
    val, (dx, dy) = grad_fn(jnp.asarray(xs), jnp.asarray(ys))
    dx.block_until_ready()
    dy.block_until_ready()
    val.block_until_ready()

    s = time.time()
    val, (dx, dy) = grad_fn(jnp.asarray(xs), jnp.asarray(ys))
    dx.block_until_ready()
    dy.block_until_ready()
    val.block_until_ready()
    print(f"AEP: {val} in {time.time() - s:.3f} seconds")

    # exit()

    options = jax.profiler.ProfileOptions()
    options.host_tracer_level = 3
    with jax.profiler.trace(
        "/tmp/jax-trace", create_perfetto_link=True, profiler_options=options
    ):
        val, (dx, dy) = grad_fn(jnp.asarray(xs), jnp.asarray(ys))
        dx.block_until_ready()
        dy.block_until_ready()
        val.block_until_ready()
