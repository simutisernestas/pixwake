import time

import jax
import jax.numpy as jnp
import numpy as onp
from flax import serialization
from py_wake.examples.data.dtu10mw import DTU10MW

from pixwake import WakeDeficitModelFlax, rans_wake_step, simulate_case_rans, ws2aep

if __name__ == "__main__":
    # deficit_model = WakeDeficitModelFlax()
    # variables = deficit_model.init(jax.random.PRNGKey(0), jnp.ones((1, 6)))
    # with open("./data/rans_deficit_surrogate.msgpack", "rb") as f:
    #     bytes_data = f.read()
    # restored_variables = serialization.from_bytes(variables, bytes_data)

    # @jax.jit
    # def predict(x):
    #     return deficit_model.apply(restored_variables, x)

    # o = predict(jnp.ones((1_000_000, 6)))
    # o.block_until_ready()

    # onp.random.seed(42)
    # x = onp.random.uniform(0, 1, (1_000_000, 6))
    # x = jnp.asarray(x)

    # s = time.time()
    # for _ in range(10):
    #     out = predict(x)
    #     out.block_until_ready()
    # print(f"Prediction took {time.time() - s:.3f} seconds")
    # exit()

    turbine = DTU10MW()

    ct_xp = turbine.powerCtFunction.ws_tab
    ct_fp = turbine.powerCtFunction.power_ct_tab[1, :]
    pw_fp = turbine.powerCtFunction.power_ct_tab[0, :]
    D = turbine.diameter()
    CUTOUT_WS = 25.0
    CUTIN_WS = 3.0

    onp.random.seed(42)
    T = 200
    ws = onp.random.uniform(CUTIN_WS + 1, CUTOUT_WS - 1, T)
    wd = onp.random.uniform(0, 360, T)

    wi, le = 8, 8
    xs, ys = jnp.meshgrid(  # example positions
        jnp.linspace(0, wi * 3 * D, wi),
        jnp.linspace(0, le * 3 * D, le),
    )
    xs, ys = xs.ravel(), ys.ravel()
    assert xs.shape[0] == (wi * le), xs.shape

    import time

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
    val, (dx, dy) = grad_fn(jnp.asarray(xs), jnp.asarray(ys))
    dx.block_until_ready()
    dy.block_until_ready()
    val.block_until_ready()

    s = time.time()
    val, (dx, dy) = grad_fn(jnp.asarray(xs), jnp.asarray(ys))
    print(f"AEP: {val} in {time.time() - s:.3f} seconds")
