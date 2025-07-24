import tempfile
from functools import partial

import flax.linen as fnn
import jax
import jax.numpy as jnp
from flax import serialization
from jax import config as jcfg
from jax import custom_vjp, vjp
from jax.lax import while_loop

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


@partial(
    custom_vjp,
    nondiff_argnums=(0,),
    nondiff_argnames=["tol", "damp"],
)
def fixed_point(f, x_guess, a, tol=1e-6, damp=0.5):
    max_iter = max(20, len(jnp.atleast_1d(x_guess)))

    def cond_fun(carry):
        x_prev, x, it = carry
        tol_cond = jnp.max(jnp.abs(x_prev - x)) > tol
        iter_cond = it < max_iter
        return jnp.logical_and(tol_cond, iter_cond)

    def body_fun(carry):
        _, x, it = carry
        x_new = f(x, a)
        x_damped = damp * x_new + (1 - damp) * x
        return x, x_damped, it + 1

    _, x_star, _ = while_loop(cond_fun, body_fun, (x_guess, f(x_guess, a), 0))
    # jax.debug.print("\nFixed point found after {it} iterations", it=it)
    return x_star


def fixed_point_fwd(f, x_guess, a, tol, damp):
    x_star = fixed_point(f, x_guess, a, tol=tol, damp=damp)
    return x_star, (a, x_star)


def fixed_point_rev(f, tol, damp, res, x_star_bar):
    a, x_star = res
    # vjp wrt a at the fixed point
    _, vjp_a = vjp(lambda a: f(x_star, a), a)

    # run a second fixed-point solve in reverse
    a_bar_sum = vjp_a(
        fixed_point(
            lambda u, v: v + vjp(lambda x: f(x, a), x_star)[1](u)[0],
            x_star_bar,
            x_star_bar,
            tol=tol,
            damp=damp,
        )
    )[0]
    # fixed_point’s x_guess gets no gradient
    return jnp.zeros_like(x_star), a_bar_sum


fixed_point.defvjp(fixed_point_fwd, fixed_point_rev)


def noj_wake_step(ws_eff, a):
    """
    Single update step g(ws_eff, a) -> ws_eff_new.
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


@partial(jax.vmap, in_axes=(None, None, 0, 0, None, None, None))
def simulate_case_noj(xs, ys, ws, wd, D, k, ct_curve):
    """
    Solve for ws_eff := fixed_point( wake_step, init=full(ws), a )
    where only xs, ys are differentiable.
    """
    # unpack CT curve breakpoints
    ct_xp, ct_fp = ct_curve[:, 0], ct_curve[:, 1]

    # bundle all parameters; only xs, ys will get a gradient
    a = (xs, ys, ws, wd, D, k, ct_xp, ct_fp)

    # initial guess: no deficits
    x0 = jnp.full_like(xs, ws)

    # run to convergence via our custom fixed_point
    return fixed_point(noj_wake_step, x0, a, damp=1.0)


class WakeDeficitModelFlax(fnn.Module):
    scale_x = jnp.array(
        [21.21759238, 3.60546819, 0.31714823, 0.09218609, 18.70851079, 0.25810896]
    )
    mean_x = jnp.array(
        [
            3.34995157e1,
            3.63567130e-04,
            2.25024289e-02,
            1.43747711e-01,
            -1.45229452e-03,
            6.07149107e-01,
        ]
    )
    scale_y = jnp.array([0.02168894])
    mean_y = jnp.array([0.00614207])

    @fnn.compact
    def __call__(self, x):
        x = (x - self.mean_x) / self.scale_x
        x = fnn.tanh(fnn.Dense(70)(x))
        x = fnn.sigmoid(fnn.Dense(102)(x))
        x = fnn.sigmoid(fnn.Dense(102)(x))
        x = fnn.sigmoid(fnn.Dense(102)(x))
        x = fnn.Dense(1)(x)
        return (x * self.scale_y) + self.mean_y


class WakeAddedTIModelFlax(fnn.Module):
    scale_x = jnp.array(
        [21.21759238, 3.60546819, 0.31714823, 0.09218609, 18.70851079, 0.25810896]
    )
    mean_x = jnp.array(
        [
            3.34995157e1,
            3.63567130e-04,
            2.25024289e-02,
            1.43747711e-01,
            -1.45229452e-03,
            6.07149107e-01,
        ]
    )
    scale_y = jnp.array([0.00571155])
    mean_y = jnp.array([0.0014295])

    @fnn.compact
    def __call__(self, x):
        x = (x - self.mean_x) / self.scale_x
        x = fnn.sigmoid(fnn.Dense(118)(x))
        x = fnn.sigmoid(fnn.Dense(118)(x))
        x = fnn.sigmoid(fnn.Dense(118)(x))
        x = fnn.sigmoid(fnn.Dense(118)(x))
        x = fnn.Dense(1)(x)
        return (x * self.scale_y) + self.mean_y


# TODO: clean up!
deficit_model = WakeDeficitModelFlax()
variables = deficit_model.init(jax.random.PRNGKey(0), jnp.ones((1, 6)))
with open("./data/rans_deficit_surrogate.msgpack", "rb") as f:
    bytes_data = f.read()
deficit_weights = serialization.from_bytes(variables, bytes_data)

turbulence_model = WakeAddedTIModelFlax()
variables = deficit_model.init(jax.random.PRNGKey(0), jnp.ones((1, 6)))
with open("./data/rans_addedti_surrogate.msgpack", "rb") as f:
    bytes_data = f.read()
ti_weights = serialization.from_bytes(variables, bytes_data)


def rans_wake_step(ws_eff, a, use_effective=True):
    xs, ys, ws, wd, D, ct_xp, ct_fp, ambient_ti = a

    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    wd_rad = jnp.deg2rad((270.0 - wd + 180.0) % 360.0)
    cos_a = jnp.cos(wd_rad)
    sin_a = jnp.sin(wd_rad)
    x_d = -(dx * cos_a + dy * sin_a) / D
    y_d = (dx * sin_a - dy * cos_a) / D
    ct = jnp.interp(ws_eff, ct_xp, ct_fp)
    in_domain_mask = (
        (x_d < 70) & (x_d > -3) & (jnp.abs(y_d) < 6) & (jnp.eye(len(xs)) == 0)
    )

    def _predict(model, params, ti):
        md_input = jnp.stack(
            [
                x_d,  # normalized x distance
                y_d,  # normalized y distance
                jnp.zeros_like(x_d),  # (z - h_hub) / D; evaluating at hub height
                jnp.full_like(x_d, ti),  # turbulence intensity
                jnp.zeros_like(x_d),  # yaw
                jnp.broadcast_to(ct, x_d.shape),  # thrust coefficient
            ],
            axis=-1,
        ).reshape(-1, 6)
        output = model.apply(params, md_input).reshape(x_d.shape)
        return jnp.where(in_domain_mask, output, 0.0).sum(axis=1)

    effective_ti = ambient_ti + _predict(turbulence_model, ti_weights, ambient_ti)

    deficit = _predict(deficit_model, deficit_weights, effective_ti)

    if use_effective:
        deficit *= ws_eff
        return jnp.maximum(0.0, ws - deficit)  # (N,)

    return jnp.maximum(0.0, ws * (1.0 - deficit))


def simulate_case_rans(xs, ys, ws, wd, D, ct_curve):
    """
    Solve for ws_eff := fixed_point( wake_step, init=full(ws), a )
    where only xs, ys are differentiable.
    """
    # unpack CT curve breakpoints
    ct_xp, ct_fp = ct_curve[:, 0], ct_curve[:, 1]

    # bundle all parameters; only xs, ys will get a gradient
    ti = 0.1
    a = (xs, ys, ws, wd, D, ct_xp, ct_fp, ti)

    # initial guess: no deficits
    x0 = jnp.full_like(xs, ws)

    # run to convergence via our custom fixed_point
    return fixed_point(rans_wake_step, x0, a, damp=0.8, tol=1e-3)


def ws2power(ws_eff, pc):
    pc_xp, pc_fp = pc[:, 0], pc[:, 1]

    # ws_eff: (T, N)
    def per_case(ws_e):
        p = jnp.interp(ws_e, pc_xp, pc_fp)
        return p

    powers = jax.vmap(per_case)(ws_eff)
    return powers


def ws2aep(ws_eff, pc, prob=None, normalize_prob=False):
    wt_powers = ws2power(ws_eff, pc)  # kW
    wt_powers *= 1e3  # W

    h_year = 24 * 365
    to_giga = 1e-9

    if prob is None:
        # ws_eff.shape[0] is time dimension; i.e.
        # timeseries range should cover one year
        return (wt_powers * h_year * to_giga).sum() / ws_eff.shape[0]  # GWh

    norm = 1 if not normalize_prob else 0.0  # TODO: implement normalization
    return (wt_powers * prob / norm * h_year * to_giga).sum()
