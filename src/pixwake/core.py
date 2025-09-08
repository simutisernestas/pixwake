from functools import partial
from typing import Any, Callable

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

    def power(self, ws: jnp.ndarray) -> jnp.ndarray:
        """Calculates the power output of the turbine for a given wind speed.

        Args:
            ws: The wind speed(s) at which to calculate power.

        Returns:
            The power output(s) of the turbine.
        """

        def _power_single_case(_ws: jnp.ndarray) -> jnp.ndarray:
            return jnp.interp(_ws, self.power_curve.wind_speed, self.power_curve.values)

        return jax.vmap(_power_single_case)(ws)

    def ct(self, ws: jnp.ndarray) -> jnp.ndarray:
        """Calculates the thrust coefficient of the turbine for a given wind speed.

        Args:
            ws: The wind speed(s) at which to calculate thrust coefficient.

        Returns:
            The thrust coefficient(s) of the turbine.
        """

        def _ct_single_case(_ws: jnp.ndarray) -> jnp.ndarray:
            return jnp.interp(_ws, self.ct_curve.wind_speed, self.ct_curve.values)

        return jax.vmap(_ct_single_case)(ws)


@dataclass
class SimulationContext:
    """A dataclass to hold the context of a wind farm simulation.

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
        effective_ws: Effective wind speeds at each turbine for each wind condition.
        turbine: The turbine object used in the simulation.
    """

    effective_ws: jnp.ndarray
    ctx: SimulationContext

    def power(self) -> jnp.ndarray:
        """Calculates the power of each turbine for each wind condition."""
        return self.ctx.turbine.power(self.effective_ws)

    def aep(self, probabilities: jnp.ndarray | None = None) -> jnp.ndarray:
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
            turbine_powers * probabilities * hours_in_year * gwh_conversion_factor
        ).sum()


class WakeSimulation:
    """The main class for running wind farm wake simulations.

    This class orchestrates the simulation by taking a wake model and handling
    the iterative process of solving for the effective wind speeds at each
    turbine. It supports different mapping strategies for running simulations
    over multiple wind conditions.
    """

    def __init__(
        self,
        model: Any,
        turbine: Turbine | None = None,  # TODO: !!!
        fpi_damp: float = 0.5,
        fpi_tol: float = 1e-6,
        mapping_strategy: str = "vmap",
    ) -> None:
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
        self.turbine = turbine

        self.__sim_call_table: dict[str, Callable] = {
            "vmap": self._simulate_vmap,
            "map": self._simulate_map,
            "_manual": self._simulate_manual,  # debug/profile purposes only
        }

    def __call__(
        self,
        xs: jnp.ndarray,
        ys: jnp.ndarray,
        ws: jnp.ndarray,
        wd: jnp.ndarray,
        turbine: Turbine,
    ) -> SimulationResult:
        """Runs the wake simulation.
        Args:
            xs: An array of x-coordinates for each turbine.
            ys: An array of y-coordinates for each turbine.
            ws: An array of free-stream wind speeds.
            wd: An array of wind directions.
            turbine: The turbine object to use in the simulation.
            x: An array of x-coordinates for flow map evaluation points (optional).
            y: An array of y-coordinates for flow map evaluation points (optional).
        Returns:
            A `SimulationResult` object containing relevant output information.
        """
        if self.mapping_strategy not in self.__sim_call_table.keys():
            raise ValueError(
                f"Invalid mapping strategy: {self.mapping_strategy}. "
                f"Valid options are: {self.__sim_call_table.keys()}"
            )

        xs = jnp.asarray(xs)
        ys = jnp.asarray(ys)
        ws = jnp.asarray(ws)
        wd = jnp.asarray(wd)

        ctx = SimulationContext(xs, ys, ws, wd, turbine)
        sim_func = self.__sim_call_table[self.mapping_strategy]
        return SimulationResult(effective_ws=sim_func(ctx), ctx=ctx)

    def flow_map(
        self,
        wt_x: jnp.ndarray,
        wt_y: jnp.ndarray,
        fm_x: jnp.ndarray | None = None,  # TODO: handle None
        fm_y: jnp.ndarray | None = None,  # TODO: handle None
        ws: float | jnp.ndarray = 10.0,
        wd: float | jnp.ndarray = 270.0,
    ) -> jnp.ndarray:
        # TODO: replace with a call to self !
        ws = jnp.atleast_1d(ws)
        wd = jnp.atleast_1d(wd)

        ctx = SimulationContext(wt_x, wt_y, ws, wd, self.turbine)
        sim_func = self.__sim_call_table[self.mapping_strategy]
        effective_ws = sim_func(ctx)

        return jax.vmap(
            lambda _ws, _wd, _ws_eff: self.model.compute_deficit(
                _ws_eff,
                SimulationContext(ctx.xs, ctx.ys, _ws, _wd, ctx.turbine),
                xs_r=fm_x,
                ys_r=fm_y,
            )
        )(ws, wd, effective_ws)

    def _simulate_vmap(
        self,
        ctx: SimulationContext,
    ) -> jnp.ndarray:
        """Simulates multiple wind conditions using jax.vmap."""

        def _single_case(ws: jnp.ndarray, wd: jnp.ndarray) -> jnp.ndarray:
            return self._simulate_single_case(
                SimulationContext(ctx.xs, ctx.ys, ws, wd, ctx.turbine)
            )

        return jax.vmap(_single_case)(ctx.ws, ctx.wd)

    def _simulate_map(
        self,
        ctx: SimulationContext,
    ) -> jnp.ndarray:
        """Simulates multiple wind conditions using jax.lax.map."""

        def _single_case(case: tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
            ws, wd = case
            return self._simulate_single_case(
                SimulationContext(ctx.xs, ctx.ys, ws, wd, ctx.turbine)
            )

        return jax.lax.map(_single_case, (ctx.ws, ctx.wd))

    def _simulate_manual(
        self,
        ctx: SimulationContext,
    ) -> jnp.ndarray:
        """Simulates multiple wind conditions using a manual loop (for debugging)."""
        return jnp.array(
            [
                self._simulate_single_case(
                    SimulationContext(ctx.xs, ctx.ys, _ws, _wd, ctx.turbine)
                )
                for _ws, _wd in zip(ctx.ws, ctx.wd)
            ]
        )

    def _simulate_single_case(
        self,
        ctx: SimulationContext,
    ) -> jnp.ndarray:
        """Simulates a single wind condition."""
        x0 = jnp.full_like(ctx.xs, ctx.ws, dtype=jnp.float64)
        return fixed_point(self.model, x0, ctx, damp=self.fpi_damp, tol=self.fpi_tol)


@partial(
    custom_vjp,
    nondiff_argnums=(0,),
    nondiff_argnames=["tol", "damp"],
)
def fixed_point(
    f: Callable,
    x_guess: jnp.ndarray,
    ctx: SimulationContext,
    tol: float = 1e-6,
    damp: float = 0.5,
) -> jnp.ndarray:
    """Finds the fixed point of a function using iterative updates.

    This function is used to solve for the stable effective wind speeds in the
    wind farm. It has a custom vector-Jacobian product (VJP) defined to
    enable automatic differentiation through the fixed-point iteration.

    Args:
        f: The function to iterate.
        x_guess: The initial guess for the fixed point.
        ctx: The context of the simulation.
        tol: The tolerance for convergence.
        damp: The damping factor for the updates.

    Returns:
        The fixed point of the function.
    """
    max_iter = max(20, len(jnp.atleast_1d(x_guess)))

    def cond_fun(carry: tuple) -> jnp.ndarray:
        x_prev, x, it = carry
        tol_cond = jnp.max(jnp.abs(x_prev - x)) > tol
        iter_cond = it < max_iter
        return jnp.logical_and(tol_cond, iter_cond)

    def body_fun(carry: tuple) -> tuple:
        _, x, it = carry
        x_new = f(x, ctx)
        x_damped = damp * x_new + (1 - damp) * x
        return x, x_damped, it + 1

    # TODO: for debugging should implement this version to not trace any objects
    # carry = (x_guess, f(x_guess, ctx), 0)
    # while cond_fun(carry):
    #     carry = body_fun(carry)
    # _, x_star, it = carry

    _, x_star, it = while_loop(cond_fun, body_fun, (x_guess, f(x_guess, ctx), 0))
    # TODO: remove !
    # jax.debug.print("\nFixed point found after {it} iterations", it=it)
    return x_star


def fixed_point_fwd(
    f: Callable,
    x_guess: jnp.ndarray,
    ctx: Any,
    tol: float,
    damp: float,
) -> tuple[jnp.ndarray, tuple]:
    x_star = fixed_point(f, x_guess, ctx, tol=tol, damp=damp)
    return x_star, (ctx, x_star)


def fixed_point_rev(
    f: Callable, tol: float, damp: float, res: tuple, x_star_bar: jnp.ndarray
) -> tuple[jnp.ndarray, Any]:
    ctx, x_star = res
    # vjp wrt a at the fixed point
    _, vjp_a = vjp(lambda s: f(x_star, s), ctx)

    # run a second fixed-point solve in reverse
    a_bar_sum = vjp_a(
        fixed_point(
            lambda u, v: v + vjp(lambda x: f(x, ctx), x_star)[1](u)[0],
            x_star_bar,
            x_star_bar,
            tol=tol,
            damp=damp,
        )
    )[0]
    # fixed_point’s x_guess gets no gradient
    return jnp.zeros_like(x_star), a_bar_sum


fixed_point.defvjp(fixed_point_fwd, fixed_point_rev)
