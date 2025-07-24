from functools import partial

import jax
import jax.numpy as jnp
from jax import custom_vjp, vjp
from jax.lax import while_loop


from flax.struct import dataclass


from .types import Turbine


@dataclass
class SimulationState:
    xs: jnp.ndarray
    ys: jnp.ndarray
    ws: jnp.ndarray
    wd: jnp.ndarray
    turbine: Turbine

    @classmethod
    def create(cls, xs, ys, ws, wd, turbine, **kwargs):
        return cls(xs, ys, ws, wd, turbine, **kwargs)


@partial(
    custom_vjp,
    nondiff_argnums=(0,),
    nondiff_argnames=["tol", "damp"],
)
def fixed_point(f, x_guess, state, tol=1e-6, damp=0.5):
    max_iter = max(20, len(jnp.atleast_1d(x_guess)))

    def cond_fun(carry):
        x_prev, x, it = carry
        tol_cond = jnp.max(jnp.abs(x_prev - x)) > tol
        iter_cond = it < max_iter
        return jnp.logical_and(tol_cond, iter_cond)

    def body_fun(carry):
        _, x, it = carry
        x_new = f(x, state)
        x_damped = damp * x_new + (1 - damp) * x
        return x, x_damped, it + 1

    _, x_star, _ = while_loop(cond_fun, body_fun, (x_guess, f(x_guess, state), 0))
    # jax.debug.print("\nFixed point found after {it} iterations", it=it)
    return x_star


def fixed_point_fwd(f, x_guess, state, tol, damp):
    x_star = fixed_point(f, x_guess, state, tol=tol, damp=damp)
    return x_star, (state, x_star)


def fixed_point_rev(f, tol, damp, res, x_star_bar):
    state, x_star = res
    # vjp wrt a at the fixed point
    _, vjp_a = vjp(lambda s: f(x_star, s), state)

    # run a second fixed-point solve in reverse
    a_bar_sum = vjp_a(
        fixed_point(
            lambda u, v: v + vjp(lambda x: f(x, state), x_star)[1](u)[0],
            x_star_bar,
            x_star_bar,
            tol=tol,
            damp=damp,
        )
    )[0]
    # fixed_point’s x_guess gets no gradient
    return jnp.zeros_like(x_star), a_bar_sum


fixed_point.defvjp(fixed_point_fwd, fixed_point_rev)


class WakeSimulation:
    def __init__(self, model, mapping_strategy="vmap"):
        self.model = model
        self.mapping_strategy = mapping_strategy

    def __call__(self, xs, ys, ws, wd, turbine):
        if self.mapping_strategy == "vmap":
            return self._simulate_vmap(xs, ys, ws, wd, turbine)
        elif self.mapping_strategy == "map":
            return self._simulate_map(xs, ys, ws, wd, turbine)
        else:
            raise ValueError(f"Unknown mapping strategy: {self.mapping_strategy}")

    def _simulate_vmap(self, xs, ys, ws, wd, turbine):
        return jax.vmap(self._simulate_single_case, in_axes=(None, None, 0, 0, None))(
            xs, ys, ws, wd, turbine
        )

    def _simulate_map(self, xs, ys, ws, wd, turbine):
        def to_be_mapped(wr):
            return self._simulate_single_case(xs, ys, wr[0], wr[1], turbine)

        wind_resource = jnp.stack([ws, wd], axis=1)
        return jax.lax.map(to_be_mapped, wind_resource)

    def _simulate_single_case(self, xs, ys, ws, wd, turbine):
        state = SimulationState.create(xs, ys, ws, wd, turbine)
        x0 = jnp.full_like(state.xs, state.ws)
        return fixed_point(
            self.model, x0, state, damp=self.model.damp, tol=getattr(self.model, "tol", 1e-6)
        )


def calculate_power(effective_wind_speed, power_curve):
    """
    Calculates the power output of a wind turbine given the effective wind speed and power curve.
    """

    def power_per_case(wind_speed):
        return jnp.interp(
            wind_speed, power_curve.wind_speed, power_curve.values
        )

    return jax.vmap(power_per_case)(effective_wind_speed)


def calculate_aep(effective_wind_speed, power_curve, probabilities=None):
    """
    Calculates the Annual Energy Production (AEP) of a wind farm.
    """
    turbine_powers = calculate_power(effective_wind_speed, power_curve) * 1e3  # W

    hours_in_year = 24 * 365
    gwh_conversion_factor = 1e-9

    if probabilities is None:
        # Assuming timeseries covers one year
        return (turbine_powers * hours_in_year * gwh_conversion_factor).sum() / effective_wind_speed.shape[0]

    return (turbine_powers * probabilities / 1.0 * hours_in_year * gwh_conversion_factor).sum()
