from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from jax import custom_vjp, vjp
from jax.lax import while_loop

from pixwake.jax_utils import default_float_type


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
        ti: The turbulence intensity.
        turbine: The turbine object used in the simulation.
    """

    xs: jnp.ndarray
    ys: jnp.ndarray
    ws: jnp.ndarray
    wd: jnp.ndarray
    turbine: Turbine
    ti: jnp.ndarray | None = None


@dataclass
class SimulationResult:
    """A dataclass to hold the results of a wind farm simulation.
    Attributes:
        effective_ws: Effective wind speeds at each turbine for each wind condition.
        ctx: The context of the simulation.
        effective_ti: Effective turbulence intensity at each turbine for each wind condition.
    """

    effective_ws: jnp.ndarray
    ctx: SimulationContext
    effective_ti: jnp.ndarray | None = None

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
        turbine: Turbine,
        fpi_damp: float = 0.5,
        fpi_tol: float = 1e-6,
        mapping_strategy: str = "vmap",
    ) -> None:
        """Initializes the WakeSimulation.
        Args:
            model: The wake model to use for the simulation.
            turbine: An optional default turbine to use for simulations.
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
            "map": self._simulate_map,  # more memory efficient than vmap
            "_manual": self._simulate_manual,  # debug/profile purposes only
        }

    def __call__(
        self,
        xs: jnp.ndarray,
        ys: jnp.ndarray,
        ws: jnp.ndarray,
        wd: jnp.ndarray,
        ti: jnp.ndarray | float | None = None,
    ) -> SimulationResult:
        """Runs the wake simulation.
        Args:
            xs: An array of x-coordinates for each turbine.
            ys: An array of y-coordinates for each turbine.
            ws: An array of free-stream wind speeds.
            wd: An array of wind directions.
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

        xs = self._atleast_1d_jax(xs)
        ys = self._atleast_1d_jax(ys)
        ws = self._atleast_1d_jax(ws)
        wd = self._atleast_1d_jax(wd)
        if ti is not None:
            ti = self._atleast_1d_jax(ti)
            if ti.size == 1:
                ti = jnp.full_like(ws, ti)
            if ti.shape != ws.shape:
                raise ValueError(
                    f"Turbulence intensity shape {ti.shape} "
                    f"does not match wind speed shape {ws.shape}."
                )

        ctx = SimulationContext(xs, ys, ws, wd, self.turbine, ti)
        sim_func = self.__sim_call_table[self.mapping_strategy]

        sim_result = sim_func(ctx)
        if isinstance(sim_result, tuple) and len(sim_result) == 2:
            ws_eff, ti_eff = sim_result
        else:
            ws_eff, ti_eff = sim_result, None

        return SimulationResult(effective_ws=ws_eff, effective_ti=ti_eff, ctx=ctx)

    def flow_map(
        self,
        wt_x: jnp.ndarray,
        wt_y: jnp.ndarray,
        fm_x: jnp.ndarray | None = None,
        fm_y: jnp.ndarray | None = None,
        ws: jnp.ndarray | float = 10.0,
        wd: jnp.ndarray | float = 270.0,
        ti: jnp.ndarray | float | None = None,
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        """Generates a flow map for the wind farm."""
        ws = self._atleast_1d_jax(ws)
        wd = self._atleast_1d_jax(wd)
        if ti is not None:
            ti = self._atleast_1d_jax(ti)
            if ti.size == 1:
                ti = jnp.full_like(ws, ti)
            if ti.shape != ws.shape:
                raise ValueError(
                    f"Turbulence intensity shape {ti.shape} "
                    f"does not match wind speed shape {ws.shape}."
                )

        if fm_x is None or fm_y is None:
            grid_res = 100
            x_min, x_max = jnp.min(wt_x) - 200, jnp.max(wt_x) + 200
            y_min, y_max = jnp.min(wt_y) - 200, jnp.max(wt_y) + 200
            grid_x, grid_y = jnp.meshgrid(
                jnp.linspace(x_min, x_max, grid_res),
                jnp.linspace(y_min, y_max, grid_res),
            )
            fm_x, fm_y = grid_x.ravel(), grid_y.ravel()

        result = self(wt_x, wt_y, ws, wd, ti)

        return jax.vmap(
            lambda _ws, _wd, _ws_eff, _ti: self.model.compute_deficit(
                _ws_eff,
                SimulationContext(
                    result.ctx.xs, result.ctx.ys, _ws, _wd, self.turbine, _ti
                ),
                xs_r=fm_x,
                ys_r=fm_y,
                ti_eff=_ti,
            )
        )(ws, wd, result.effective_ws, result.effective_ti), (fm_x, fm_y)

    def _simulate_vmap(
        self,
        ctx: SimulationContext,
    ) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
        """Simulates multiple wind conditions using jax.vmap."""

        def _single_case(
            ws: jnp.ndarray, wd: jnp.ndarray, ti: jnp.ndarray | None
        ) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
            return self._simulate_single_case(
                SimulationContext(ctx.xs, ctx.ys, ws, wd, ctx.turbine, ti)
            )

        return jax.vmap(_single_case)(ctx.ws, ctx.wd, ctx.ti)

    def _simulate_map(
        self,
        ctx: SimulationContext,
    ) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
        """Simulates multiple wind conditions using jax.lax.map."""

        def _single_case(
            case: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None],
        ) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
            ws, wd, ti = case
            return self._simulate_single_case(
                SimulationContext(ctx.xs, ctx.ys, ws, wd, ctx.turbine, ti)
            )

        return jax.lax.map(_single_case, (ctx.ws, ctx.wd, ctx.ti))

    def _simulate_manual(
        self,
        ctx: SimulationContext,
    ) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
        """Simulates multiple wind conditions using a manual loop (for debugging)."""
        # Pre-allocate output arrays
        n_cases = ctx.ws.size
        n_turbines = ctx.xs.size
        ws_out = jnp.zeros((n_cases, n_turbines))
        ti_out = (
            jnp.zeros((n_cases, n_turbines))
            if getattr(self.model, "use_effective_ti", False)
            else None
        )

        for i in range(n_cases):
            _ws = ctx.ws[i]
            _wd = ctx.wd[i]
            _ti = None if ctx.ti is None else ctx.ti[i]

            result = self._simulate_single_case(
                SimulationContext(ctx.xs, ctx.ys, _ws, _wd, ctx.turbine, _ti)
            )

            if isinstance(result, tuple):
                ws_eff, ti_eff = result
                ws_out = ws_out.at[i].set(ws_eff)
                if ti_out is not None:
                    ti_out = ti_out.at[i].set(ti_eff)
            else:
                ws_out = ws_out.at[i].set(result)

        return (ws_out, ti_out) if ti_out is not None else ws_out

    def _simulate_single_case(
        self,
        ctx: SimulationContext,
    ) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
        """Simulates a single wind condition."""
        ws_effective = jnp.full_like(ctx.xs, ctx.ws, dtype=default_float_type())
        ti_effective = None if ctx.ti is None else jnp.full_like(ws_effective, ctx.ti)
        x_guess = (ws_effective, ti_effective)

        if self.mapping_strategy == "_manual":
            return fixed_point_debug(
                self.model, x_guess, ctx, damp=self.fpi_damp, tol=self.fpi_tol
            )

        return fixed_point(
            self.model, x_guess, ctx, damp=self.fpi_damp, tol=self.fpi_tol
        )

    def _atleast_1d_jax(self, x: jnp.ndarray | float | list) -> jnp.ndarray:
        return jnp.atleast_1d(jnp.asarray(x))


@partial(
    custom_vjp,
    nondiff_argnums=(0,),
    nondiff_argnames=["tol", "damp"],
)
def fixed_point(
    f: Callable,
    x_guess: jnp.ndarray | tuple,
    ctx: Any,
    tol: float = 1e-6,
    damp: float = 0.5,
) -> jnp.ndarray | tuple:
    """This function solves for a fixed point, i.e., a value `x` such that `f(x) = x`.
    In the context of wake modeling, this is used to determine the stable effective
    wind speeds within a wind farm, where the wind speed at each turbine is
    influenced by the wakes of others.

    The function is JAX-transformable and has a custom vector-Jacobian product (VJP)
    to allow for automatic differentiation. This is essential for gradient-based
    optimization of wind farm layouts or control strategies.

    The mathematical foundation for the custom VJP is the implicit function theorem.
    If we have a fixed-point equation `x_star = f(x_star, a)`, where `a` represents
    the parameters of `f`, we can differentiate both sides to find the derivative
    of `x_star` with respect to `a`.

    Args:
        f: The function to iterate, which should accept the current estimate of the
        fixed point and the simulation context `ctx`.
        x_guess: The initial guess for the fixed point. This can be a single array
                or a pytree of arrays.
        ctx: The context of the simulation, containing parameters and other data
            needed by the function `f`.
        tol: The tolerance for convergence. The iteration stops when the maximum
            absolute difference between successive estimates is below this value.
        damp: The damping factor for the updates, used to stabilize the iteration.
            A value of 0.0 means no damping, while a value closer to 1.0
            introduces more damping.
    Returns:
        The fixed point of the function, with the same structure as `x_guess`.
    """
    max_iter = max(20, len(jnp.atleast_1d(jax.tree.leaves(x_guess)[0])))

    def cond_fun(carry: tuple) -> jnp.ndarray:
        x_prev, x, it = carry
        ws_prev = x_prev[0] if isinstance(x_prev, tuple) else x_prev
        ws = x[0] if isinstance(x, tuple) else x
        tol_cond = jnp.max(jnp.abs(ws_prev - ws)) > tol
        iter_cond = it < max_iter
        return jnp.logical_and(tol_cond, iter_cond)

    def body_fun(carry: tuple) -> tuple:
        _, x, it = carry
        x_new = f(x, ctx)
        x_damped = jax.tree.map(lambda n, o: damp * n + (1 - damp) * o, x_new, x)
        return x, x_damped, it + 1

    _, x_star, it = while_loop(cond_fun, body_fun, (x_guess, f(x_guess, ctx), 0))
    return x_star


def fixed_point_fwd(
    f: Callable,
    x_guess: jnp.ndarray | tuple,
    ctx: Any,
    tol: float,
    damp: float,
) -> tuple[jnp.ndarray | tuple, tuple]:
    """Forward pass for the custom VJP of the fixed_point function."""
    x_star = fixed_point(f, x_guess, ctx, tol=tol, damp=damp)
    return x_star, (ctx, x_star)


def fixed_point_rev(
    f: Callable, tol: float, damp: float, res: tuple, x_star_bar: jnp.ndarray | tuple
) -> tuple[jnp.ndarray | tuple, Any]:
    """Reverse pass for the custom VJP of the fixed_point function."""
    ctx, x_star = res
    _, vjp_a = vjp(lambda s: f(x_star, s), ctx)

    def _inner_f(
        v: jnp.ndarray | tuple,
        _: Any,
    ) -> jnp.ndarray | tuple:
        """Helper function for the reverse-mode differentiation of the fixed point.

        This function represents the application of the Jacobian of `f` with respect
        to `x` to a vector `v`. It is used within another fixed-point iteration
        to solve for the gradients.
        """
        return jax.tree.map(
            jnp.add,
            vjp(lambda x: f(x, ctx), x_star)[1](v)[0],
            x_star_bar,
        )

    # Solve another fixed-point problem to compute the gradient.
    # This is derived from the implicit function theorem.
    a_bar_sum = vjp_a(fixed_point(_inner_f, x_star_bar, None, tol=tol, damp=damp))[0]

    # The gradient with respect to `x_guess` is zero, as the fixed point
    # does not depend on the initial guess.
    return jax.tree.map(jnp.zeros_like, x_star), a_bar_sum


fixed_point.defvjp(fixed_point_fwd, fixed_point_rev)


def fixed_point_debug(
    f: Callable,
    x_guess: jnp.ndarray | tuple,
    ctx: Any,
    tol: float = 1e-6,
    damp: float = 0.5,
) -> jnp.ndarray | tuple:
    """Finds the fixed point of a function using iterative updates.
    This function is for debugging purposes only and is not JAX-transformable.
    It uses a standard Python while loop instead of `jax.lax.while_loop`.
    """
    max_iter = max(20, len(jnp.atleast_1d(jax.tree.leaves(x_guess)[0])))

    it = 0
    x_prev = x_guess
    x = f(x_guess, ctx)

    ws_prev = x_prev[0] if isinstance(x_prev, tuple) else x_prev
    ws = x[0] if isinstance(x, tuple) else x

    while jnp.max(jnp.abs(ws - ws_prev)) > tol and it < max_iter:
        x_prev = x
        x_new = f(x, ctx)
        x = jax.tree.map(lambda n, o: damp * n + (1 - damp) * o, x_new, x)

        ws_prev = x_prev[0] if isinstance(x_prev, tuple) else x_prev
        ws = x[0] if isinstance(x, tuple) else x
        it += 1

    return x
