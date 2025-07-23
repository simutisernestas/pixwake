import time

import jax
import jax.numpy as jnp
import numpy as onp
from py_wake.examples.data.dtu10mw import DTU10MW

from pixwake import simulate_case_rans, ws2aep


def test_rans_surrogate_aep():
    turbine = DTU10MW()

    ct_xp = turbine.powerCtFunction.ws_tab
    ct_fp = turbine.powerCtFunction.power_ct_tab[1, :]
    pw_fp = turbine.powerCtFunction.power_ct_tab[0, :]
    D = turbine.diameter()
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

    if 0:
        options = jax.profiler.ProfileOptions()
        options.host_tracer_level = 3
        with jax.profiler.trace(
            "/tmp/jax-trace", create_perfetto_link=True, profiler_options=options
        ):
            res = aep_and_grad(jnp.asarray(xs), jnp.asarray(ys))
            block_all(res)

    if 0:
        ws_eff = simulate_case_rans(
            xs,
            ys,
            9.0,
            90.0,
            D,
            jnp.stack([ct_xp, ct_fp], axis=1),
        )
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 10))
        plt.scatter(xs, ys, c=ws_eff, cmap="viridis", s=100)
        plt.colorbar(label="Effective Wind Speed (m/s)")
        plt.title("Effective Wind Speed at Turbine Locations")
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.grid()
        plt.show()


if __name__ == "__main__":
    test_rans_surrogate_aep()
