from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from pixwake.deficit.base import WakeDeficit
    from pixwake.turbulence.base import WakeTurbulence

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import custom_vjp, vjp
from jax.lax import while_loop
from jax.tree_util import register_pytree_node_class

from pixwake.jax_utils import default_float_type


@dataclass
class Curve:
    """Represents a performance curve, such as a power or thrust curve.

    This dataclass stores the wind speeds and corresponding values (e.g., power
    in kW or thrust coefficient) that define a turbine's performance
    characteristics.

    Attributes:
        wind_speed: A JAX numpy array of wind speeds, typically in m/s.
        values: A JAX numpy array of the corresponding performance values.
    """

    wind_speed: jnp.ndarray
    values: jnp.ndarray


@dataclass
class Turbine:
    """Represents a wind turbine's physical and performance characteristics.

    This dataclass holds all the necessary information about a wind turbine,
    including its dimensions and performance curves.

    Attributes:
        rotor_diameter: The diameter of the turbine's rotor in meters.
        hub_height: The height of the turbine's hub above the ground in meters.
        power_curve: A `Curve` object representing the turbine's power curve.
        ct_curve: A `Curve` object representing the turbine's thrust coefficient
            curve.
    """

    rotor_diameter: float
    hub_height: float
    power_curve: Curve
    ct_curve: Curve

    def power(self, ws: jnp.ndarray) -> jnp.ndarray:
        """Calculates the turbine's power output for given wind speeds.

        This method interpolates the power curve to find the power output for
        each wind speed in the input array.

        Args:
            ws: A JAX numpy array of wind speeds.

        Returns:
            A JAX numpy array of the corresponding power outputs.
        """

        def _power_single_case(_ws: jnp.ndarray) -> jnp.ndarray:
            return jnp.interp(_ws, self.power_curve.wind_speed, self.power_curve.values)

        return jax.vmap(_power_single_case)(ws)

    def ct(self, ws: jnp.ndarray) -> jnp.ndarray:
        """Calculates the thrust coefficient for given wind speeds.

        This method interpolates the thrust coefficient curve to find the `Ct`
        value for each wind speed in the input array.

        Args:
            ws: A JAX numpy array of wind speeds.

        Returns:
            A JAX numpy array of the corresponding thrust coefficients.
        """

        def _ct_single_case(_ws: jnp.ndarray) -> jnp.ndarray:
            return jnp.interp(_ws, self.ct_curve.wind_speed, self.ct_curve.values)

        return jax.vmap(_ct_single_case)(ws)


@dataclass
@register_pytree_node_class
class SimulationContext:
    """Holds the static context for a single wind farm simulation case.

    This dataclass is a container for all the static information required to
    run a single simulation scenario. It is designed to be compatible with JAX's
    pytree mechanism, allowing it to be used seamlessly with transformations
    like `jax.vmap` and `jax.lax.map`.

    The `tree_flatten` and `tree_unflatten` methods are implemented to specify
    how JAX should handle this class. Dynamic data (JAX arrays) are treated as

    "children," while static data (like the `Turbine` object) are treated as
    "auxiliary" data.

    Attributes:
        turbine: The `Turbine` object used in the simulation.
        dw: A JAX numpy array of downwind distances between all pairs of
            turbines.
        cw: A JAX numpy array of crosswind distances between all pairs of
            turbines.
        ws: The free-stream wind speed for the simulation case.
        ti: The ambient turbulence intensity for the simulation case.
    """

    # site variables
    turbine: Turbine
    dw: jnp.ndarray
    cw: jnp.ndarray
    ws: jnp.ndarray
    ti: jnp.ndarray | None = None

    # wake properties (set by deficit model)
    # wake_radius: jnp.ndarray | None = None # TODO: add wake radius here ?

    def tree_flatten(self) -> tuple[tuple, tuple]:
        """Flattens the `SimulationContext` for JAX's pytree mechanism."""
        children = (self.dw, self.cw, self.ws, self.ti)
        aux_data = (self.turbine,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: tuple, children: tuple) -> "SimulationContext":
        """Unflattens the `SimulationContext` for JAX's pytree mechanism."""
        return cls(*aux_data, *children)


@dataclass
class SimulationResult:
    """Holds the results of a wind farm simulation.

    This dataclass is a container for the output of a `WakeSimulation`,
    providing easy access to the effective wind speeds, power output, and other
    relevant data.

    Attributes:
        turbine: The `Turbine` object used in the simulation.
        wt_x: A JAX numpy array of the x-coordinates of the wind turbines.
        wt_y: A JAX numpy array of the y-coordinates of the wind turbines.
        wd: A JAX numpy array of the wind directions for each simulation case.
        ws: A JAX numpy array of the free-stream wind speeds for each case.
        effective_ws: A JAX numpy array of the effective wind speed at each
            turbine for each case.
        ti: An optional JAX numpy array of the ambient turbulence intensity for
            each case.
        effective_ti: An optional JAX numpy array of the effective turbulence
            intensity at each turbine for each case.
    """

    turbine: Turbine
    wt_x: jnp.ndarray
    wt_y: jnp.ndarray
    wd: jnp.ndarray
    ws: jnp.ndarray
    effective_ws: jnp.ndarray
    ti: jnp.ndarray | None = None
    effective_ti: jnp.ndarray | None = None

    def power(self) -> jnp.ndarray:
        """Calculates the power of each turbine for each wind condition."""
        return self.turbine.power(self.effective_ws)

    def aep(self, probabilities: jnp.ndarray | None = None) -> jnp.ndarray:
        """Calculates the Annual Energy Production (AEP) of the wind farm.

        This method computes the total AEP in GWh. If `probabilities` are not
        provided, it assumes a uniform distribution over the simulation cases.

        Args:
            probabilities: An optional JAX numpy array of the probabilities for
                each simulation case.

        Returns:
            The total AEP of the wind farm in GWh.
        """
        # TODO: power values can have arbitrary units;
        # these scaling factors should be parameters probably
        # or not consider units at all here. Now assuming power in kW...
        turbine_powers = self.power() * 1e3  # W

        hours_in_year = 24 * 365
        gwh_conversion_factor = 1e-9

        if probabilities is None:
            # Assuming timeseries covers one year
            return (
                turbine_powers * hours_in_year * gwh_conversion_factor
            ).sum() / self.effective_ws.shape[0]

        probabilities = probabilities.reshape(-1, 1)
        assert probabilities.shape[0] == turbine_powers.shape[0]

        return (
            turbine_powers * probabilities * hours_in_year * gwh_conversion_factor
        ).sum()


class WakeSimulation:
    """Orchestrates wind farm wake simulations.

    This is the main class for running simulations. It takes a deficit model,
    an optional turbulence model, and a turbine definition, and then runs the
    simulation for a given set of ambient conditions.

    The core of the simulation is a fixed-point iteration that solves for the
    stable effective wind speeds at each turbine, considering the wake
    interactions between them.

    This class supports multiple strategies for mapping over different wind
    conditions (`vmap`, `map`, `_manual`), allowing for flexibility in terms of
    performance and memory usage.
    """

    def __init__(
        self,
        turbine: Turbine,
        deficit: WakeDeficit,
        turbulence: WakeTurbulence | None = None,
        fpi_damp: float = 0.5,
        fpi_tol: float = 1e-6,
        mapping_strategy: str = "auto",
    ) -> None:
        """Initializes the `WakeSimulation`.

        Args:
            turbine: A `Turbine` object representing the wind turbines in the
                farm.
            deficit: A `WakeDeficit` model to calculate the velocity deficit.
            turbulence: An optional `WakeTurbulence` model to calculate the
                added turbulence.
            fpi_damp: The damping factor for the fixed-point iteration.
            fpi_tol: The convergence tolerance for the fixed-point iteration.
            mapping_strategy: The JAX mapping strategy to use for multiple wind
                conditions. Options are 'vmap', 'map', or '_manual'.
        """
        self.deficit = deficit
        self.turbine = turbine
        self.turbulence = turbulence
        self.mapping_strategy = mapping_strategy
        self.fpi_damp = fpi_damp
        self.fpi_tol = fpi_tol

        def __auto_mapping() -> Callable:
            return (
                self._simulate_map
                if jax.default_backend() == "cpu"
                else self._simulate_vmap
            )

        self.__sim_call_table: dict[str, Callable] = {
            "auto": __auto_mapping(),
            "vmap": self._simulate_vmap,
            "map": self._simulate_map,  # more memory efficient than vmap
            "_manual": self._simulate_manual,  # debug/profile purposes only
        }

    def __call__(
        self,
        wt_xs: jnp.ndarray,
        wt_ys: jnp.ndarray,
        ws_amb: jnp.ndarray,
        wd: jnp.ndarray,
        ti: jnp.ndarray | float | None = None,
    ) -> SimulationResult:
        """Runs the wake simulation for the given ambient conditions.

        Args:
            wt_xs: A JAX numpy array of the x-coordinates of the turbines.
            wt_ys: A JAX numpy array of the y-coordinates of the turbines.
            ws_amb: A JAX numpy array of the free-stream wind speeds.
            wd: A JAX numpy array of the wind directions.
            ti: An optional JAX numpy array of the ambient turbulence
                intensities.

        Returns:
            A `SimulationResult` object containing the results of the
            simulation.
        """
        if self.mapping_strategy not in self.__sim_call_table.keys():
            raise ValueError(
                f"Invalid mapping strategy: {self.mapping_strategy}. "
                f"Valid options are: {self.__sim_call_table.keys()}"
            )

        sc = self._preprocess_ambient_conditions(wt_xs, wt_ys, ws_amb, wd, ti)
        wt_xs, wt_ys, ws_amb, wd, ti = sc

        dw, cw = self._get_downwind_crosswind_distances(wd, wt_xs, wt_ys)
        ctx = SimulationContext(self.turbine, dw, cw, ws_amb, ti)
        sim_func = self.__sim_call_table[self.mapping_strategy]
        ws_eff, ti_eff = sim_func(ctx)
        return SimulationResult(
            self.turbine, wt_xs, wt_ys, wd, ws_amb, ws_eff, ti, ti_eff
        )

    def _atleast_1d_jax(self, x: jnp.ndarray | float | list) -> jnp.ndarray:
        """Ensures the input is at least a 1D JAX numpy array."""
        return jnp.atleast_1d(jnp.asarray(x))

    def _preprocess_ambient_conditions(
        self,
        wt_xs: jnp.ndarray | float | list,
        wt_ys: jnp.ndarray | float | list,
        ws_amb: jnp.ndarray | float | list,
        wd: jnp.ndarray | float | list,
        ti: jnp.ndarray | float | list | None,
    ) -> tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray | None,
    ]:
        """Preprocesses and validates the ambient conditions."""
        wt_xs = self._atleast_1d_jax(wt_xs)
        wt_ys = self._atleast_1d_jax(wt_ys)
        assert wt_xs.shape == wt_ys.shape
        assert len(wt_xs.shape) == 1

        ws_amb = self._atleast_1d_jax(ws_amb)
        wd = self._atleast_1d_jax(wd)
        assert ws_amb.shape == wd.shape
        assert len(ws_amb.shape) == 1

        if ti is not None:
            ti = self._atleast_1d_jax(ti)
            if ti.size == 1:
                ti = jnp.full_like(ws_amb, ti)
            assert ti.shape == ws_amb.shape

        return wt_xs, wt_ys, ws_amb, wd, ti

    def _get_downwind_crosswind_distances(
        self,
        wd: jnp.ndarray,
        xs_s: jnp.ndarray,
        ys_s: jnp.ndarray,
        xs_r: jnp.ndarray | None = None,
        ys_r: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Calculates downwind and crosswind distances between points.

        This method transforms the Cartesian coordinates of source and receiver
        points into a frame of reference aligned with the wind direction.

        Args:
            wd: The wind direction in degrees.
            xs_s: The x-coordinates of the source points.
            ys_s: The y-coordinates of the source points.
            xs_r: The x-coordinates of the receiver points. If `None`, the
                source points are used.
            ys_r: The y-coordinates of the receiver points. If `None`, the
                source points are used.

        Returns:
            A tuple containing the downwind and crosswind distances.
        """
        xs_r = xs_s if xs_r is None else xs_r
        ys_r = ys_s if ys_r is None else ys_r
        dx = xs_r[:, None] - xs_s[None, :]
        dy = ys_r[:, None] - ys_s[None, :]
        wd_rad = jnp.deg2rad((270.0 - wd + 180.0) % 360.0)
        cos_a = jnp.cos(wd_rad)
        sin_a = jnp.sin(wd_rad)
        # Result shape: (n_wd, n_turbines, n_turbines)
        x_d = -(dx * cos_a[:, None, None] + dy * sin_a[:, None, None])
        y_d = dx * sin_a[:, None, None] - dy * cos_a[:, None, None]
        return x_d, y_d

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
        """Generates a 2D flow map of the wind farm.

        This method calculates the wind speed at a grid of points in the wind
        farm, allowing for visualization of the wake effects.

        Args:
            wt_x: The x-coordinates of the wind turbines.
            wt_y: The y-coordinates of the wind turbines.
            fm_x: The x-coordinates of the flow map grid. If `None`, a default
                grid is generated.
            fm_y: The y-coordinates of the flow map grid. If `None`, a default
                grid is generated.
            ws: The free-stream wind speed.
            wd: The wind direction.
            ti: The ambient turbulence intensity.

        Returns:
            A tuple containing the flow map wind speeds and the grid
            coordinates.
        """

        sc = self._preprocess_ambient_conditions(wt_x, wt_y, ws, wd, ti)
        wt_x, wt_y, ws, wd, ti = sc

        if fm_x is None or fm_y is None:
            grid_res = 200
            x_min, x_max = jnp.min(wt_x) - 200, jnp.max(wt_x) + 200
            y_min, y_max = jnp.min(wt_y) - 200, jnp.max(wt_y) + 200
            grid_x, grid_y = jnp.meshgrid(
                jnp.linspace(x_min, x_max, grid_res),
                jnp.linspace(y_min, y_max, grid_res),
            )
            fm_x, fm_y = grid_x.ravel(), grid_y.ravel()

        result = self(wt_x, wt_y, ws, wd, ti)

        dw, cw = self._get_downwind_crosswind_distances(wd, wt_x, wt_y, fm_x, fm_y)

        return jax.vmap(
            lambda _ws_amb, _dw, _cw, _ws_eff, _ti_eff: self.deficit(
                _ws_eff,
                _ti_eff,
                SimulationContext(self.turbine, _dw, _cw, _ws_amb, _ti_eff),
            )
        )(ws, dw, cw, result.effective_ws, result.effective_ti)[0], (fm_x, fm_y)

    def _simulate_vmap(
        self, ctx: SimulationContext
    ) -> tuple[jnp.ndarray, jnp.ndarray | None]:
        """Simulates multiple wind conditions using `jax.vmap`."""

        def _single_case(
            dw: jnp.ndarray, cw: jnp.ndarray, ws: jnp.ndarray, ti: jnp.ndarray | None
        ) -> tuple[jnp.ndarray, jnp.ndarray | None]:
            return self._simulate_single_case(
                SimulationContext(ctx.turbine, dw, cw, ws, ti)
            )

        return jax.vmap(_single_case)(ctx.dw, ctx.cw, ctx.ws, ctx.ti)

    def _simulate_map(
        self, ctx: SimulationContext
    ) -> tuple[jnp.ndarray, jnp.ndarray | None]:
        """Simulates multiple wind conditions using `jax.lax.map`."""

        def _single_case(
            case: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray | None],
        ) -> tuple[jnp.ndarray, jnp.ndarray | None]:
            dw, cw, ws, ti = case
            single_flow_case_ctx = SimulationContext(ctx.turbine, dw, cw, ws, ti)
            return self._simulate_single_case(single_flow_case_ctx)

        return jax.lax.map(_single_case, (ctx.dw, ctx.cw, ctx.ws, ctx.ti))

    def _simulate_manual(
        self, ctx: SimulationContext
    ) -> tuple[jnp.ndarray, jnp.ndarray | None]:
        """Simulates multiple conditions with a Python loop (for debugging)."""
        n_cases = ctx.ws.size
        n_turbines = ctx.dw.shape[0]
        ws_out = jnp.zeros((n_cases, n_turbines))
        ti_out = None if ctx.ti is None else jnp.zeros((n_cases, n_turbines))

        for i in range(n_cases):
            _ws = ctx.ws[i]
            _dw = ctx.dw[i]
            _cw = ctx.cw[i]
            _ti = None if ctx.ti is None else ctx.ti[i]

            ws_eff, ti_eff = self._simulate_single_case(
                SimulationContext(ctx.turbine, _dw, _cw, _ws, _ti)
            )
            ws_out = ws_out.at[i].set(ws_eff)
            if ti_out is not None:
                assert ti_eff is not None
                ti_out = ti_out.at[i].set(ti_eff)

        return (ws_out, ti_out)

    def _simulate_single_case(
        self,
        ctx: SimulationContext,
    ) -> tuple[jnp.ndarray, jnp.ndarray | None]:
        """Simulates a single wind condition.

        This method initializes the effective wind speeds and turbulence
        intensities with the ambient conditions and then uses the `fixed_point`
        solver to find the stable state.
        """
        # initialize all turbines with ambient quantities
        fdtype = default_float_type()
        n_turbines = ctx.dw.shape[0]
        ws_amb = jnp.full(n_turbines, ctx.ws, dtype=fdtype)
        ti_amb = None if ctx.ti is None else jnp.full(n_turbines, ctx.ti, dtype=fdtype)
        x_guess = (ws_amb, ti_amb)

        fp_func = (  # fixed_point_debug is not traced and can be used with pydebugger
            fixed_point if self.mapping_strategy != "_manual" else fixed_point_debug
        )
        return fp_func(
            self._solve_farm, x_guess, ctx, damp=self.fpi_damp, tol=self.fpi_tol
        )

    def _solve_farm(
        self, effective: tuple, ctx: SimulationContext
    ) -> tuple[jnp.ndarray, jnp.ndarray | None]:
        """Performs one iteration of the fixed-point solver.

        This method takes the current estimates of effective wind speed and
        turbulence, calculates the new values based on the wake model, and
        returns the updated estimates.

        Args:
            effective: A tuple containing the current effective wind speed and
                turbulence intensity.
            ctx: The simulation context.

        Returns:
            A tuple with the updated effective wind speed and turbulence
            intensity.
        """
        ws_eff, ti_eff = effective
        ws_eff_new, wake_radius = self.deficit(ws_eff, ti_eff, ctx)

        ti_eff_new = ti_eff
        if self.turbulence:
            if ctx.ti is None:
                raise ValueError(
                    "Turbulence model provided but ambient TI is None in context."
                )
            ti_eff_new = self.turbulence(ws_eff_new, ti_eff, ctx, wake_radius)

        output: tuple[jnp.ndarray, jnp.ndarray | None] = (ws_eff_new, ti_eff_new)
        return output


@partial(
    custom_vjp,
    nondiff_argnums=(0,),
    nondiff_argnames=["tol", "damp"],
)
def fixed_point(
    f: Callable,
    x_guess: tuple[jnp.ndarray, jnp.ndarray | None],
    ctx: SimulationContext,
    tol: float = 1e-6,
    damp: float = 0.5,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
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
    ctx: SimulationContext,
    tol: float,
    damp: float,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray | None], tuple]:
    """Forward pass for the custom VJP of the fixed_point function."""
    x_star = fixed_point(f, x_guess, ctx, tol=tol, damp=damp)
    return x_star, (ctx, x_star)


def fixed_point_rev(
    f: Callable, tol: float, damp: float, res: tuple, x_star_bar: jnp.ndarray | tuple
) -> tuple[tuple[jnp.ndarray, jnp.ndarray | None], SimulationContext]:
    """Reverse pass for the custom VJP of the fixed_point function."""
    ctx, x_star = res
    _, vjp_a = vjp(lambda s: f(x_star, s), ctx)

    def _inner_f(v: jnp.ndarray | tuple, _: Any) -> jnp.ndarray | tuple:
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
    ctx: SimulationContext,
    tol: float = 1e-6,
    damp: float = 0.5,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
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
