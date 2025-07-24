import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as onp
from py_wake.examples.data.dtu10mw import DTU10MW

from pixwake import *



if __name__ == "__main__":
    turbine = DTU10MW()

    ct_xp = turbine.powerCtFunction.ws_tab  # [:-1]
    ct_fp = turbine.powerCtFunction.power_ct_tab[1, :]  # [:-1]
    pw_fp = turbine.powerCtFunction.power_ct_tab[0, :]  # [:-1]
    # print(ct_fp)
    # exit()

    coeffs = jnp.polyfit(ct_xp[:-1], pw_fp[:-1], 6)  # [a, b, c] for ax^2 + bx + c

    approx_pw_fp = jnp.polyval(coeffs, ct_xp[:-1])

    # # plot power and ct curve
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(ct_xp, pw_fp, label="Power Coefficient")
    # plt.plot(ct_xp, pw_fp, 'o')
    # plt.plot(ct_xp[:-1], approx_pw_fp, label="Polynomial Approximation")
    # plt.plot(ct_xp[:-1], approx_pw_fp, 'x')
    # plt.xlabel("Wind Speed (m/s)")
    # plt.ylabel("Power Coefficient")
    # plt.title("DTU 10MW Power Coefficient")
    # plt.grid()
    # plt.show()
    # exit()

    D = turbine.diameter()
    CUTOUT_WS = 25.0
    CUTIN_WS = 3.0

    onp.random.seed(42)
    T = 300
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

    exit()

    grad_fn = jax.jit(
        jax.value_and_grad(
            lambda xx, yy:
            # ws2aep(
            simulate_case_rans(
                xx,
                yy,
                jnp.atleast_1d(WSS),
                jnp.atleast_1d(WDS),
                D,
                jnp.stack([ct_xp, ct_fp], axis=1),
            ).mean(),
            #     jnp.stack([ct_xp, pw_fp], axis=1),
            #     inter=coeffs,
            # ),
            argnums=(0, 1),
        )
    )

    # out, (dx, dy) = grad_fn(
    #     jnp.asarray(xs),
    #     jnp.asarray(ys),
    #     jnp.asarray([10.0]),
    #     jnp.asarray([90.0]),
    # )
    # out.block_until_ready()
    # dx.block_until_ready()
    # dy.block_until_ready()

    # res = []
    # s = time.time()
    # for ws, wd in zip(WSS, WDS):
    #     print(f"Running for ws={ws:.2f}, wd={wd:.2f}")
    #     res.append(
    #         grad_fn(
    #             jnp.asarray(xs),
    #             jnp.asarray(ys),
    #             jnp.asarray(ws),
    #             jnp.asarray(wd),
    #         )
    #     )
    # print(res)
    # print(f"Total time: {time.time() - s:.3f} seconds")
    # exit()

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

    exit()

    options = jax.profiler.ProfileOptions()
    options.host_tracer_level = 3
    with jax.profiler.trace(
        "/tmp/jax-trace", create_perfetto_link=True, profiler_options=options
    ):
        val, (dx, dy) = grad_fn(jnp.asarray(xs), jnp.asarray(ys))
        dx.block_until_ready()
        dy.block_until_ready()
        val.block_until_ready()
