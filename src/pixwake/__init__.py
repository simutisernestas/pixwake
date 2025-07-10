from functools import partial
import jax
import jax.numpy as jnp
from jax import custom_vjp, vjp
from jax.lax import while_loop
from jax import config as jcfg
import tempfile

jcfg.update("jax_compilation_cache_dir", tempfile.gettempdir())
jcfg.update("jax_persistent_cache_min_entry_size_bytes", -1)
jcfg.update("jax_persistent_cache_min_compile_time_secs", 1e-2)
jcfg.update("jax_enable_x64", False)


def __get_eps():
    return (
        jax.numpy.finfo(jnp.float64).eps
        if jcfg.jax_enable_x64
        else jax.numpy.finfo(jnp.float32).eps
    )


# ------------------------
# 1) fixed_point with custom_vjp
# ------------------------
@partial(custom_vjp, nondiff_argnums=(0,), nondiff_argnames=["tol"])
def fixed_point(f, a, x_guess, tol=1e-6):
    def cond_fun(carry):
        x_prev, x = carry
        # iterate until max abs change < tol
        return jnp.max(jnp.abs(x_prev - x)) > tol

    def body_fun(carry):
        _, x = carry
        return x, f(a, x)

    _, x_star = while_loop(cond_fun, body_fun, (x_guess, f(a, x_guess)))
    return x_star


def fixed_point_fwd(f, a, x_guess, tol):
    x_star = fixed_point(f, a, x_guess, tol=tol)
    return x_star, (a, x_star)


def rev_iter(f, packed, u):
    a, x_star, x_star_bar = packed
    _, vjp_x = vjp(lambda x: f(a, x), x_star)
    return x_star_bar + vjp_x(u)[0]


def fixed_point_rev(f, _, res, x_star_bar):
    a, x_star = res
    # vjp wrt a at the fixed point
    _, vjp_a = vjp(lambda a: f(a, x_star), a)

    # run a second fixed-point solve in reverse
    (a_bar,) = vjp_a(
        fixed_point(partial(rev_iter, f), (a, x_star, x_star_bar), x_star_bar)
    )
    # fixed_point’s x_guess gets no gradient
    return a_bar, jnp.zeros_like(x_star)


fixed_point.defvjp(fixed_point_fwd, fixed_point_rev)


# ------------------------
# 2) NOJ turbine-wake iteration via fixed_point
# ------------------------
def wake_step(a, ws_eff):
    """
    Single update step g(a, ws_eff) -> ws_eff_new.
    a = (xs, ys, ws, wd, D, k, ct_xp, ct_fp)
    """
    xs, ys, ws, wd, D, k, ct_xp, ct_fp = a

    # geometry
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    wd_rad = jnp.deg2rad((270.0 - wd + 180.0) % 360.0)
    cos_a = jnp.cos(wd_rad)
    sin_a = jnp.sin(wd_rad)
    x_d = -(dx * cos_a + dy * sin_a)
    y_d = dx * sin_a - dy * cos_a
    wake_rad = (D / 2) + k * x_d

    # mask upstream turbines within wake cone
    mask = (x_d > 0) & (jnp.abs(y_d) < wake_rad)

    # interpolate CT curve
    ct = jnp.interp(ws_eff, ct_xp, ct_fp)

    # wake deficit formulation
    a_coef = ct * (0.2460 + ct * (0.0586 + ct * 0.0883))
    term = 2 * a_coef * ((D / 2) / jnp.maximum(wake_rad, __get_eps())) ** 2

    # combine deficits in quadrature
    deficits = jnp.sqrt(jnp.sum(jnp.where(mask, term**2, 0.0), axis=1) + __get_eps())

    # new effective speed
    return jnp.maximum(0.0, ws * (1.0 - deficits))


def simulate_case(xs, ys, ws, wd, D, k, ct_curve):
    """
    Solve for ws_eff := fixed_point( wake_step, a, init=full(ws) )
    where only xs, ys are differentiable.
    """
    # unpack CT curve breakpoints
    ct_xp, ct_fp = ct_curve[:, 0], ct_curve[:, 1]

    # bundle all parameters; only xs, ys will get a gradient
    a = (xs, ys, ws, wd, D, k, ct_xp, ct_fp)

    # initial guess: no deficits
    x0 = jnp.full_like(xs, ws)

    # run to convergence via our custom fixed_point
    return fixed_point(wake_step, a, x0)


batched_simulate_case = jax.vmap(simulate_case, (None, None, 0, 0, None, None, None))


def ws2power(ws_eff, pc):
    pc_xp, pc_fp = pc[:, 0], pc[:, 1]

    # ws_eff: (T, N)
    def per_case(ws_e):
        p = jnp.interp(ws_e, pc_xp, pc_fp)
        return p

    powers = jax.vmap(per_case)(ws_eff)
    return powers


def ws2aep(ws_eff, pc):
    wt_powers = ws2power(ws_eff, pc)  # kW
    wt_powers *= 1e3  # W
    aep = (wt_powers * (24 * 365) * 1e-9).sum()  # GWh
    return aep / ws_eff.shape[0]


# ------------------------
# 3) example: gradient wrt turbine positions
# ------------------------
if __name__ == "__main__":
    D = 100.0  # rotor diameter
    k = 0.05  # wake decay constant
    CUTOUT_WS = 25.0
    CUTIN_WS = 3.0

    import numpy as onp

    onp.random.seed(42)
    T = 9000
    ws = onp.random.uniform(CUTIN_WS + 1, CUTOUT_WS - 1, T)
    wd = onp.random.uniform(0, 360, T)

    wi, le = 10, 10
    xs, ys = jnp.meshgrid(  # example positions
        jnp.linspace(0, wi * 1e2, wi),
        jnp.linspace(0, le * 1e2, le),
    )
    xs, ys = xs.ravel(), ys.ravel()
    assert xs.shape[0] == (wi * le), xs.shape

    # realistic thrust coefficient (Ct) curve:
    #  - zero below cut-in
    #  - peaks around cut-in then gradually drops as pitch control kicks in
    #  - small residual Ct at cut-out (0.1)
    ct_vals = jnp.array(
        [
            0.00,
            0.00,
            0.00,  # 0,1,2 m/s
            0.80,  # 3 m/s (cut-in)
            0.79,
            0.77,
            0.75,  # 4,5,6 m/s
            0.72,
            0.68,
            0.64,  # 7,8,9 m/s
            0.62,
            0.61,
            0.60,  # 10,11,12 m/s (rated)
            0.55,
            0.50,
            0.45,  # 13,14,15 m/s
            0.40,
            0.35,
            0.30,  # 16,17,18 m/s
            0.25,
            0.20,
            0.18,  # 19,20,21 m/s
            0.15,
            0.12,
            0.10,  # 22,23,24 m/s
            0.10,  # 25 m/s (cut-out)
        ]
    )
    ct_curve = jnp.stack([jnp.arange(0.0, CUTOUT_WS + 1.0, 1.0), ct_vals], axis=1)

    # compute effective speeds
    ws_eff = batched_simulate_case(xs, ys, ws, wd, D, k, ct_curve)

    # grad of sum(ws_eff) wrt (xs, ys) only
    grad_fn = jax.jit(
        jax.value_and_grad(
            lambda xx, yy: jnp.sum(
                batched_simulate_case(xx, yy, ws, wd, D, k, ct_curve)
            ),
            argnums=(0, 1),
        )
    )

    val, (dx, dy) = grad_fn(xs, ys)
    dx.block_until_ready()
    dy.block_until_ready()
    val.block_until_ready()
    import time

    s = time.time()
    val, (dx, dy) = grad_fn(xs, ys)
    val.block_until_ready()
    dx.block_until_ready()
    dy.block_until_ready()
    print(time.time() - s)
