from functools import partial

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from jax import custom_vjp, vjp
from jax.lax import while_loop


@dataclass
class Curve:
    """A dataclass to represent a curve, such as a power or thrust curve.

    Attributes:
        wind_speed: An array of wind speeds.
        values: An array of corresponding values (e.g., power or thrust).
    """

    wind_speed: jnp.ndarray
    values: jnp.ndarray


@dataclass
class Turbine:
    """A dataclass to represent a wind turbine.

    Attributes:
        rotor_diameter: The diameter of the turbine's rotor.
        hub_height: The height of the turbine's hub.
        power_curve: The turbine's power curve.
        ct_curve: The turbine's thrust coefficient curve.
    """

    rotor_diameter: float
    hub_height: float
    power_curve: Curve
    ct_curve: Curve


@dataclass
class SimulationState:
    """A dataclass to hold the state of a wind farm simulation.

    Attributes:
        xs: An array of x-coordinates for each turbine.
        ys: An array of y-coordinates for each turbine.
        ws: The free-stream wind speed.
        wd: The wind direction.
        turbine: The turbine object used in the simulation.
    """

    xs: jnp.ndarray
    ys: jnp.ndarray
    ws: jnp.ndarray
    wd: jnp.ndarray
    turbine: Turbine


@dataclass
class SimulationResult:
    """A dataclass to hold the results of a wind farm simulation.
    Attributes:
        effective_wind_speed: Effective wind speeds at each turbine for each wind condition.
        turbine: The turbine object used in the simulation.
    """

    effective_ws: jnp.ndarray
    turbine: Turbine

    def power(self):
        """Calculates the power of each turbine for each wind condition."""

        def power_per_case(wind_speed: jnp.ndarray):
            return jnp.interp(
                wind_speed,
                self.turbine.power_curve.wind_speed,
                self.turbine.power_curve.values,
            )

        return jax.vmap(power_per_case)(self.effective_ws)

    def aep(self, probabilities=None):
        """Calculates the Annual Energy Production (AEP) of a wind farm."""
        turbine_powers = self.power() * 1e3  # W

        hours_in_year = 24 * 365
        gwh_conversion_factor = 1e-9

        if probabilities is None:
            # Assuming timeseries covers one year
            return (
                turbine_powers * hours_in_year * gwh_conversion_factor
            ).sum() / self.effective_ws.shape[0]

        return (
            turbine_powers * probabilities / 1.0 * hours_in_year * gwh_conversion_factor
        ).sum()


class WakeSimulation:
    """The main class for running wind farm wake simulations.

    This class orchestrates the simulation by taking a wake model and handling
    the iterative process of solving for the effective wind speeds at each
    turbine. It supports different mapping strategies for running simulations
    over multiple wind conditions.
    """

    def __init__(self, model, fpi_damp=0.5, fpi_tol=1e-6, mapping_strategy="vmap"):
        """Initializes the WakeSimulation.

        Args:
            model: The wake model to use for the simulation.
            fpi_damp: The damping factor for the fixed-point iteration.
            fpi_tol: The tolerance for the fixed-point iteration.
            mapping_strategy: The strategy to use for mapping over multiple
                wind conditions. Can be 'vmap', 'map', or '_manual'.
        """
        self.model = model
        self.mapping_strategy = mapping_strategy
        self.fpi_damp = fpi_damp
        self.fpi_tol = fpi_tol

        self.__sim_call_table = {
            "vmap": self._simulate_vmap,
            "map": self._simulate_map,
            "_manual": self._simulate_manual,  # debug/profile purposes only
        }

    def __call__(self, xs, ys, ws, wd, turbine):
        """Runs the wake simulation.
        Args:
            xs: An array of x-coordinates for each turbine.
            ys: An array of y-coordinates for each turbine.
            ws: An array of free-stream wind speeds.
            wd: An array of wind directions.
            turbine: The turbine object to use in the simulation.
        Returns:
            A `SimulationResult` object containing relevant output information.
        """
        if self.mapping_strategy not in self.__sim_call_table.keys():
            raise ValueError(
                f"Invalid mapping strategy: {self.mapping_strategy}. "
                f"Valid options are: {self.__sim_call_table.keys()}"
            )

        sim_func = self.__sim_call_table[self.mapping_strategy]
        eff_ws = sim_func(xs, ys, ws, wd, turbine)
        return SimulationResult(effective_ws=eff_ws, turbine=turbine)

    def _simulate_vmap(self, xs, ys, ws, wd, turbine):
        """Simulates multiple wind conditions using jax.vmap."""
        vmaped_simulate_all_cases = jax.vmap(
            self._simulate_single_case, in_axes=(None, None, 0, 0, None)
        )
        return vmaped_simulate_all_cases(xs, ys, ws, wd, turbine)

    def _simulate_map(self, xs, ys, ws, wd, turbine):
        """Simulates multiple wind conditions using jax.lax.map."""
        return jax.lax.map(
            lambda case: self._simulate_single_case(xs, ys, case[0], case[1], turbine),
            (ws, wd),
        )

    def _simulate_manual(self, xs, ys, ws, wd, turbine):
        """Simulates multiple wind conditions using a manual loop (for debugging)."""
        return jnp.array(
            [
                self._simulate_single_case(xs, ys, _ws, _wd, turbine)
                for _ws, _wd in zip(ws, wd)
            ]
        )

    def _simulate_single_case(self, xs, ys, ws, wd, turbine):
        """Simulates a single wind condition."""
        state = SimulationState(xs, ys, ws, wd, turbine)
        x0 = jnp.full_like(state.xs, state.ws)
        return fixed_point(self.model, x0, state, damp=self.fpi_damp, tol=self.fpi_tol)


@partial(
    custom_vjp,
    nondiff_argnums=(0,),
    nondiff_argnames=["tol", "damp"],
)
def fixed_point(f, x_guess, state, tol=1e-6, damp=0.5):
    """Finds the fixed point of a function using iterative updates.

    This function is used to solve for the stable effective wind speeds in the
    wind farm. It has a custom vector-Jacobian product (VJP) defined to
    enable automatic differentiation through the fixed-point iteration.

    Args:
        f: The function to iterate.
        x_guess: The initial guess for the fixed point.
        state: The state of the simulation.
        tol: The tolerance for convergence.
        damp: The damping factor for the updates.

    Returns:
        The fixed point of the function.
    """
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

    _, x_star, it = while_loop(cond_fun, body_fun, (x_guess, f(x_guess, state), 0))
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
