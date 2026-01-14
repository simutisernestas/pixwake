from __future__ import annotations

import hashlib
import pickle
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

from pixwake.jax_utils import default_float_type, ssqrt


@dataclass(frozen=True)
@register_pytree_node_class
class Curve:
    """Represents a performance curve, such as a power or thrust curve.

    This dataclass stores the wind speeds and corresponding values (e.g., power
    in kW or thrust coefficient) that define a turbine's performance characteristics.

    Attributes:
        wind_speed: A JAX numpy array of wind speeds, typically in m/s.
        values: A JAX numpy array of the corresponding performance values.
    """

    ws: jnp.ndarray
    values: jnp.ndarray

    def tree_flatten(self) -> tuple[tuple, tuple]:
        children = (self.ws, self.values)
        aux_data = ()
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, _: tuple, children: tuple) -> Curve:
        return cls(*children)


@dataclass(frozen=True)
@register_pytree_node_class
class Turbine:
    """Represents a wind turbine's physical and performance characteristics.

    This dataclass holds all the necessary information about a wind turbine,
    including its dimensions and performance curves.

    Attributes:
        rotor_diameter: The diameter of the turbine's rotor in meters.
        hub_height: The height of the turbine's hub above the ground in meters.
        power_curve: A `Curve` object representing the turbine's power curve.
        ct_curve: A `Curve` object representing the turbine's thrust coefficient curve.
    """

    rotor_diameter: float | jnp.ndarray
    hub_height: float | jnp.ndarray
    power_curve: Curve
    ct_curve: Curve
    type_id: int | None = None
    _is_single_type: bool | None = None  # Cached flag for single vs multi turbine

    @property
    def __type_id(self) -> int:
        """Generates a unique type ID based on the turbine's serialized binary object.

        With k objects hashed & n number of bits -> P(collision) = 1 - np.exp(-k**2/(2*2**n))

        >>> import numpy as np
        >>> k = 20
        >>> n = 64
        >>> 1 - np.exp(-k**2/(2*2**n))
        np.float64(0.0)
        """
        binary_data = pickle.dumps(self)
        hash_obj = hashlib.sha256(binary_data)
        return int.from_bytes(hash_obj.digest()[:8])

    def __post_init__(self) -> None:
        if self.type_id is None:
            object.__setattr__(self, "type_id", self.__type_id)
        # Cache the single-type check to avoid repeated jnp.asarray calls
        if self._is_single_type is None:
            is_single = (
                not hasattr(self.rotor_diameter, "ndim")
                or jnp.asarray(self.rotor_diameter).ndim == 0
            )
            object.__setattr__(self, "_is_single_type", is_single)

    def power(self, ws: jnp.ndarray) -> jnp.ndarray:
        """Calculates the turbine's power output for given wind speeds.
        This method interpolates the power curve to find the power output for
        each wind speed in the input array.
        Args:
            ws: A JAX numpy array of wind speeds.
        Returns:
            A JAX numpy array of the corresponding power outputs.
        """
        if self._is_single_type:
            # Single turbine type - direct interpolation
            return jnp.interp(ws, self.power_curve.ws, self.power_curve.values)

        # Multiple turbine types - need vectorized interpolation
        def _interp(
            _ws: jnp.ndarray, _ws_curve: jnp.ndarray, _curve_values: jnp.ndarray
        ) -> jnp.ndarray:
            return jnp.interp(_ws, _ws_curve, _curve_values)

        if ws.ndim == 1:
            return jax.vmap(_interp, in_axes=(0, 0, 0))(
                ws, self.power_curve.ws, self.power_curve.values
            )

        return jax.vmap(_interp, in_axes=(1, 0, 0), out_axes=1)(
            ws, self.power_curve.ws, self.power_curve.values
        )

    def ct(self, ws: jnp.ndarray) -> jnp.ndarray:
        """Calculates the thrust coefficient for given wind speeds.
        This method interpolates the thrust coefficient curve to find the `Ct`
        value for each wind speed in the input array.
        Args:
            ws: A JAX numpy array of wind speeds.
        Returns:
            A JAX numpy array of the corresponding thrust coefficients.
        """
        if self._is_single_type:
            # Single turbine type - direct interpolation
            return jnp.interp(ws, self.ct_curve.ws, self.ct_curve.values)

        # Multiple turbine types - need vectorized interpolation
        def _interp(
            _ws: jnp.ndarray, _ws_curve: jnp.ndarray, _curve_values: jnp.ndarray
        ) -> jnp.ndarray:
            return jnp.interp(_ws, _ws_curve, _curve_values)

        if ws.ndim == 1:
            return jax.vmap(_interp, in_axes=(0, 0, 0))(
                ws, self.ct_curve.ws, self.ct_curve.values
            )

        return jax.vmap(_interp, in_axes=(1, 0, 0), out_axes=1)(
            ws, self.ct_curve.ws, self.ct_curve.values
        )

    def tree_flatten(self) -> tuple[tuple, tuple]:
        children = (
            self.rotor_diameter,
            self.hub_height,
            self.power_curve,
            self.ct_curve,
        )
        aux_data = (self.type_id, self._is_single_type)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: tuple, children: tuple) -> Turbine:
        type_id, is_single_type = aux_data
        rotor_diameter, hub_height, power_curve, ct_curve = children
        return cls(
            rotor_diameter=rotor_diameter,
            hub_height=hub_height,
            power_curve=power_curve,
            ct_curve=ct_curve,
            type_id=type_id,
            _is_single_type=is_single_type,
        )

    @classmethod
    def _from_types(
        cls,
        turbine_library: list[Turbine],
        turbine_types: list[int],
    ) -> Turbines:
        """Constructs a Turbine object representing multiple turbine types.
        Only used internally to handle multiple turbine types in simulation."""

        type_id_to_index = {t.type_id: i for i, t in enumerate(turbine_library)}
        assert len(type_id_to_index.keys()) == len(turbine_library), (
            "Identical turbines found in library! It's probably not intended.. "
            "Please check your turbine definitions.."
        )

        selected = [turbine_library[type_id_to_index[tt]] for tt in turbine_types]

        def _construct_stacked_curve(attr_name: str) -> Curve:
            return Curve(
                ws=jnp.stack([getattr(t, attr_name).ws for t in selected]),
                values=jnp.stack([getattr(t, attr_name).values for t in selected]),
            )

        return cls(
            rotor_diameter=jnp.array([t.rotor_diameter for t in selected]),
            hub_height=jnp.array([t.hub_height for t in selected]),
            power_curve=_construct_stacked_curve("power_curve"),
            ct_curve=_construct_stacked_curve("ct_curve"),
            type_id=-1,  # no type for aggregated turbines
            _is_single_type=False,  # explicitly mark as multi-turbine
        )


Turbines = Turbine


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
    "children," while static data are treated as "auxiliary" data.

    Attributes:
        turbine: The `Turbine` or `Turbines` object used in the simulation.
        dw: A JAX numpy array of downwind distances between all pairs of turbines.
        cw: A JAX numpy array of crosswind distances between all pairs of turbines.
        ws: The free-stream wind speed for the simulation case.
        ti: The ambient turbulence intensity for the simulation case (optional).
        wake_radius: The wake radius at each turbine, set by the deficit model at runtime.
    """

    # site variables
    turbine: Turbines
    dw: jnp.ndarray
    cw: jnp.ndarray
    ws: jnp.ndarray
    ti: jnp.ndarray | None = None
    # set by deficit model at runtime
    wake_radius: jnp.ndarray | None = None

    def tree_flatten(self) -> tuple[tuple, tuple]:
        """Flattens the `SimulationContext` for JAX's pytree mechanism."""
        children = (
            self.turbine,
            self.dw,
            self.cw,
            self.ws,
            self.ti,
            self.wake_radius,
        )
        aux_data = ()
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, _: tuple, children: tuple) -> SimulationContext:
        """Unflattens the `SimulationContext` for JAX's pytree mechanism."""
        return cls(*children)


@dataclass(frozen=True)
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

    turbine: Turbines
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
    an optional turbulence model, and turbine configuration, and then runs the
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
        turbines: Turbine | list[Turbine],
        deficit: WakeDeficit,
        turbulence: WakeTurbulence | None = None,
        blockage: WakeDeficit | None = None,
        fpi_damp: float = 1.0,
        fpi_tol: float = 1e-6,
        mapping_strategy: str = "auto",
    ) -> None:
        """Initializes the `WakeSimulation`.

        Args:
            turbines: Either a single `Turbine` object or a list of `Turbine` objects
                representing the turbine library. When calling the simulation with `wt_types`,
                turbines will be selected from this library.
            deficit: A `WakeDeficit` model to calculate the velocity deficit.
            turbulence: An optional `WakeTurbulence` model to calculate the
                added turbulence.
            blockage: An optional blockage model (e.g., SelfSimilarityBlockageDeficit)
                to calculate the induction/blockage effects. When used in combined mode
                (with a wake deficit model), the blockage model's exclude_downstream_speedup
                flag is automatically set to True to match PyWake's behavior.
            fpi_damp: The damping factor for the fixed-point iteration.
            fpi_tol: The convergence tolerance for the fixed-point iteration.
            mapping_strategy: The JAX mapping strategy to use for multiple wind
                conditions. Options are 'auto', 'vmap', 'map', or '_manual'.
        """
        self.deficit = deficit
        self.turbulence = turbulence
        self.blockage = blockage
        self.mapping_strategy = mapping_strategy
        self.fpi_damp = fpi_damp
        self.fpi_tol = fpi_tol
        self.turbines = turbines

        # In combined mode (wake + blockage), automatically set exclude_downstream_speedup
        # to match PyWake's behavior where downstream speedup is excluded in wake regions
        if self.blockage is not None and hasattr(
            self.blockage, "exclude_downstream_speedup"
        ):
            self.blockage.exclude_downstream_speedup = True

        def __auto_mapping() -> Callable:
            return (
                self._simulate_map
                if jax.default_backend() == "cpu"
                else self._simulate_vmap
            )

        self.__sim_call_table: dict[str, Callable] = {
            "auto": __auto_mapping(),
            "vmap": self._simulate_vmap,
            "map": self._simulate_map,
            "_manual": self._simulate_manual,
        }

    def __call__(
        self,
        wt_xs: jnp.ndarray,
        wt_ys: jnp.ndarray,
        ws_amb: jnp.ndarray | float | list,
        wd_amb: jnp.ndarray | float | list,
        ti_amb: jnp.ndarray | float | None = None,
        wt_types: list[int] | None = None,
    ) -> SimulationResult:
        """Runs the wake simulation for the given ambient conditions.

        Args:
            wt_xs: x-coordinates of the wind turbines.
            wt_ys: y-coordinates of the wind turbines.
            ws_amb: Free-stream wind speed(s). Can be a single value, list, or array
                for multiple wind conditions.
            wd_amb: Wind direction(s) in degrees. Can be a single value, list, or array
                for multiple wind conditions.
            ti_amb: Optional ambient turbulence intensity. Can be a single value, list,
                or array for multiple wind conditions.
            wt_types: Optional list of turbine type IDs corresponding to each position.
                When provided, turbines are selected from the turbine library provided
                at initialization. Must have the same length as wt_xs/wt_ys.

        Returns:
            A `SimulationResult` object containing the effective wind speeds, power
            output, and other simulation results.

        Example:
            ```python
            # Single turbine type
            turbine = Turbine(...)
            sim = WakeSimulation(turbine, deficit_model)
            result = sim(wt_xs=xs, wt_ys=ys, ws_amb=10.0, wd_amb=270.0)

            # Multiple turbine types from library
            turbine_lib = [turbine1, turbine2]
            sim = WakeSimulation(turbine_lib, deficit_model)
            result = sim(wt_xs=xs, wt_ys=ys, ws_amb=10.0, wd_amb=270.0, wt_types=[
                turbine1.type_id, turbine2.type_id, turbine1.type_id, turbine2.type_id
            ]) # Alternating turbine types
            ```
        """
        if self.mapping_strategy not in self.__sim_call_table.keys():
            raise ValueError(
                f"Invalid mapping strategy: {self.mapping_strategy}. "
                f"Valid options are: {self.__sim_call_table.keys()}"
            )

        sc = self._preprocess_ambient_conditions(wt_xs, wt_ys, ws_amb, wd_amb, ti_amb)
        wt_xs, wt_ys, ws_amb, wd_amb, ti_amb = sc

        turbines = self.turbines
        if isinstance(self.turbines, list) and wt_types is not None:
            assert len(wt_xs) == len(wt_types)
            assert isinstance(self.turbines, list)
            turbines = Turbines._from_types(
                turbine_library=self.turbines, turbine_types=wt_types
            )
        assert isinstance(turbines, (Turbine, Turbines))

        hh = turbines.hub_height
        dw, cw = self._get_downwind_crosswind_distances(wd_amb, wt_xs, wt_ys, hh)
        ctx = SimulationContext(turbines, dw, cw, ws_amb, ti_amb)
        sim_func = self.__sim_call_table[self.mapping_strategy]
        ws_eff, ti_eff = sim_func(ctx)
        return SimulationResult(
            turbines, wt_xs, wt_ys, wd_amb, ws_amb, ws_eff, ti_amb, ti_eff
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
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
        """Preprocesses and validates the ambient conditions."""
        wt_xs = self._atleast_1d_jax(wt_xs)
        wt_ys = self._atleast_1d_jax(wt_ys)
        assert wt_xs.shape == wt_ys.shape
        assert len(wt_xs.shape) == 1

        ws_amb = self._atleast_1d_jax(ws_amb)
        wd = self._atleast_1d_jax(wd)
        assert ws_amb.shape == wd.shape
        assert len(ws_amb.shape) == 1

        ti_arr: jnp.ndarray | None = None
        if ti is not None:
            ti_arr = self._atleast_1d_jax(ti)
            if ti_arr.size == 1:
                ti_arr = jnp.full_like(ws_amb, ti_arr)
            assert ti_arr.shape == ws_amb.shape

        return wt_xs, wt_ys, ws_amb, wd, ti_arr

    def _get_downwind_crosswind_distances(
        self,
        wd: jnp.ndarray,
        xs_s: jnp.ndarray,
        ys_s: jnp.ndarray,
        zs_s: jnp.ndarray | float,
        xs_r: jnp.ndarray | None = None,
        ys_r: jnp.ndarray | None = None,
        zs_r: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Calculates downwind and crosswind distances between points.

        This method transforms the Cartesian coordinates of source and receiver
        points into a frame of reference aligned with the wind direction.

        Args:
            wd: The wind direction in degrees.
            xs_s: The x-coordinates of the source points.
            ys_s: The y-coordinates of the source points.
            zs_s: The z-coordinates (hub heights) of the source points.
            xs_r: The x-coordinates of the receiver points. If `None`, the
                source points are used.
            ys_r: The y-coordinates of the receiver points. If `None`, the
                source points are used.
            zs_r: The z-coordinates (hub heights) of the receiver points. If `None`,
                the source points are used.

        Returns:
            A tuple containing the downwind and crosswind distances.
        """
        # TODO: dirty fix
        zs_s = jnp.atleast_1d(zs_s)
        zs_r = jnp.atleast_1d(zs_r) if zs_r is not None else zs_s

        xs_r = xs_s if xs_r is None else xs_r
        ys_r = ys_s if ys_r is None else ys_r
        zs_r = zs_s if zs_r is None else zs_r
        dx = xs_r[:, None] - xs_s[None, :]
        dy = ys_r[:, None] - ys_s[None, :]
        dz = zs_r[:, None] - zs_s[None, :]
        # Optimized angle calculation: cos(450° - wd) = sin(wd), sin(450° - wd) = cos(wd)
        # This avoids the modulo operation and extra arithmetic
        wd_rad = jnp.deg2rad(wd)
        cos_a = jnp.sin(wd_rad)  # equivalent to cos((450 - wd) % 360)
        sin_a = jnp.cos(wd_rad)  # equivalent to sin((450 - wd) % 360)
        # Clean up near-zero values from floating-point errors in trig functions
        # JAX's cos/sin can return values like 1e-8 instead of exactly 0 for angles
        # like 90, 180, 270 degrees. This ensures symmetric grids produce symmetric dw/cw matrices.
        trig_tol = 1e-7
        cos_a = jnp.where(jnp.abs(cos_a) < trig_tol, 0.0, cos_a)
        sin_a = jnp.where(jnp.abs(sin_a) < trig_tol, 0.0, sin_a)
        # Result shape: (n_wd, n_turbines, n_turbines)
        down_wind_d = -(dx * cos_a[:, None, None] + dy * sin_a[:, None, None])
        horizontal_cross_wind_d = dx * sin_a[:, None, None] - dy * cos_a[:, None, None]
        cross_wind_d = ssqrt(horizontal_cross_wind_d**2 + dz[None, :, :] ** 2)
        return down_wind_d, cross_wind_d

    def flow_map(
        self,
        wt_x: jnp.ndarray,
        wt_y: jnp.ndarray,
        fm_x: jnp.ndarray | None = None,
        fm_y: jnp.ndarray | None = None,
        fm_z: jnp.ndarray | None = None,
        ws: jnp.ndarray | float = 10.0,
        wd: jnp.ndarray | float = 270.0,
        ti: jnp.ndarray | float | None = None,
        wt_types: list[int] | None = None,
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        """Generates a 2D flow map of the wind farm.

        This method calculates the wind speed at a grid of points in the wind
        farm, allowing for visualization of the wake effects.

        Args:
            wt_x: The x-coordinates of the wind turbines.
            wt_y: The y-coordinates of the wind turbines.
            fm_x: The x-coordinates of the flow map grid. If `None`, a default
                grid is generated based on turbine positions.
            fm_y: The y-coordinates of the flow map grid. If `None`, a default
                grid is generated based on turbine positions.
            ws: The free-stream wind speed.
            wd: The wind direction in degrees.
            ti: The ambient turbulence intensity (optional).
            wt_types: Optional list of turbine type IDs corresponding to each position.
                When provided, turbines are selected from the turbine library provided
                at initialization. Must have the same length as wt_x/wt_y.

        Returns:
            A tuple containing the flow map wind speeds (flattened array) and
            the grid coordinates (fm_x, fm_y).
        """
        sc = self._preprocess_ambient_conditions(wt_x, wt_y, ws, wd, ti)
        wt_x, wt_y, ws, wd, ti = sc

        turbines = self.turbines
        if isinstance(self.turbines, list) and wt_types is not None:
            assert len(wt_x) == len(wt_types)
            assert isinstance(self.turbines, list)
            turbines = Turbines._from_types(
                turbine_library=self.turbines, turbine_types=wt_types
            )
        assert isinstance(turbines, (Turbine, Turbines))

        if fm_x is None or fm_y is None:
            grid_res = 200  # TOOD: on larger farms this is very course...
            x_min, x_max = jnp.min(wt_x) - 200, jnp.max(wt_x) + 200
            y_min, y_max = jnp.min(wt_y) - 200, jnp.max(wt_y) + 200
            grid_x, grid_y = jnp.meshgrid(
                jnp.linspace(x_min, x_max, grid_res),
                jnp.linspace(y_min, y_max, grid_res),
            )
            fm_x, fm_y = grid_x.ravel(), grid_y.ravel()

        result = self(wt_x, wt_y, ws, wd, ti, wt_types=wt_types)

        # TODO: should support passing of fm_z as well; this could be
        # all type heights together and then could average over height
        # for convenient plotting...
        fm_z = jnp.full_like(fm_x, jnp.mean(turbines.hub_height))
        dw, cw = self._get_downwind_crosswind_distances(
            wd, wt_x, wt_y, turbines.hub_height, fm_x, fm_y, fm_z
        )

        def _apply_models_to_flowmap(
            _ws_amb: jnp.ndarray,
            _dw: jnp.ndarray,
            _cw: jnp.ndarray,
            _ws_eff: jnp.ndarray,
            _ti_eff: jnp.ndarray | None,
        ) -> jnp.ndarray:
            """Apply blockage and deficit models to flow map points."""
            ctx = SimulationContext(turbines, _dw, _cw, _ws_amb, _ti_eff)
            ws_out = _ws_eff

            # Apply blockage first (if provided)
            if self.blockage is not None:
                ws_out, ctx = self.blockage(ws_out, _ti_eff, ctx)

            # Apply wake deficit
            ws_out, _ = self.deficit(ws_out, _ti_eff, ctx)
            return ws_out

        return jax.vmap(_apply_models_to_flowmap)(
            ws, dw, cw, result.effective_ws, result.effective_ti
        ), (fm_x, fm_y)

    def _simulate_vmap(
        self, ctx: SimulationContext
    ) -> tuple[jnp.ndarray, jnp.ndarray | None]:
        """Simulates multiple wind conditions using `jax.vmap`.

        Uses pytree-aware vmap to avoid creating SimulationContext objects
        inside the vmapped function, reducing Python object overhead.
        """
        # Specify vmap axes as a matching pytree structure
        # turbine is broadcast (same for all cases), dw/cw/ws/ti are vmapped over axis 0
        # Note: JAX in_axes uses int/None for axis spec, not actual types
        in_axes_ctx = SimulationContext(
            turbine=None,  # type: ignore[arg-type]  # broadcast
            dw=0,  # type: ignore[arg-type]
            cw=0,  # type: ignore[arg-type]
            ws=0,  # type: ignore[arg-type]
            ti=0 if ctx.ti is not None else None,  # type: ignore[arg-type]
            wake_radius=None,  # not set yet
        )
        return jax.vmap(self._simulate_single_case, in_axes=(in_axes_ctx,))(ctx)

    def _simulate_map(
        self, ctx: SimulationContext
    ) -> tuple[jnp.ndarray, jnp.ndarray | None]:
        """Simulates multiple wind conditions using `jax.lax.map`.

        Uses pytree-aware map to avoid creating SimulationContext objects
        inside the mapped function, reducing Python object overhead.
        """
        # For lax.map, we need to stack the varying parts into a single pytree
        # that gets sliced along axis 0. Turbine is constant, so we close over it.
        turbine = ctx.turbine

        def _single_case(
            case: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray | None],
        ) -> tuple[jnp.ndarray, jnp.ndarray | None]:
            dw, cw, ws, ti = case
            # Create context with closed-over turbine to avoid passing it through map
            return self._simulate_single_case(
                SimulationContext(turbine, dw, cw, ws, ti)
            )

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

        The order of model application matches PyWake:
        1. Wake deficit model - computes ws_after_wake = ambient - wake_deficit
        2. Blockage model (if provided) - computes blockage_effect, then
           ws_final = ws_after_wake - blockage_effect
        3. Turbulence model (if provided) - models added turbulence

        The blockage and wake effects are combined additively:
        ws_final = ambient - wake_deficit - blockage_deficit

        In combined mode, the blockage model's exclude_downstream_speedup flag
        is set to True (in __init__), so only upstream blockage effects are
        applied, matching PyWake's behavior.

        Args:
            effective: A tuple containing the current effective wind speed and
                turbulence intensity.
            ctx: The simulation context.

        Returns:
            A tuple with the updated effective wind speed and turbulence
            intensity.
        """
        ws_eff, ti_eff = effective

        # Apply wake deficit model first
        ws_eff_new, ctx = self.deficit(ws_eff, ti_eff, ctx)

        # Apply blockage model (if provided)
        # Blockage effect is computed from ambient wind speed (not wake-affected)
        # This matches PyWake's behavior where blockage uses WS_ilk (freestream)
        if self.blockage is not None:
            # Store wake radius from wake model for exclude_wake check
            wake_radius_from_wake_model = ctx.wake_radius

            # Compute blockage effect using ambient wind speed for CT calculation
            # Broadcast ambient ws to turbine array shape
            ws_ambient = jnp.broadcast_to(ctx.ws, ws_eff.shape)
            ws_with_blockage, _ = self.blockage(
                ws_ambient,
                ti_eff,
                ctx,
                wake_radius_for_exclude=wake_radius_from_wake_model,
            )
            # Blockage deficit = ambient - blockage_modified_ws
            # The blockage model handles exclude_wake logic internally
            blockage_deficit = ctx.ws - ws_with_blockage
            # Apply blockage deficit to wake result (additive combination)
            ws_eff_new = ws_eff_new - blockage_deficit
            ws_eff_new = jnp.maximum(0.0, ws_eff_new)

        ti_eff_new = ti_eff
        if self.turbulence:
            if ctx.ti is None:
                raise ValueError(
                    "Turbulence model provided but ambient TI is None in context."
                )
            ti_eff_new = self.turbulence(ws_eff_new, ti_eff, ctx)

        output: tuple[jnp.ndarray, jnp.ndarray | None] = (ws_eff_new, ti_eff_new)
        return output

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def _aep_gradients_chunked_static(
        sim: WakeSimulation,
        wt_xs: jnp.ndarray,
        wt_ys: jnp.ndarray,
        ws_chunk: jnp.ndarray,
        wd_chunk: jnp.ndarray,
        ti_chunk: jnp.ndarray | float | None = None,
        prob_chunk: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        """Helper for chunked gradient calculation to allow for a single JIT trace."""

        def grad_chunk(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            result = sim(x, y, ws_chunk, wd_chunk, ti_chunk)
            return result.aep(probabilities=prob_chunk)

        # We can define and jit internal functions, as long as the parent is jitted
        # and the inputs that would change compilation hash are static.
        # As chunk size is now static, we only trace this once per chunk size.
        return jax.value_and_grad(grad_chunk, argnums=(0, 1))(wt_xs, wt_ys)

    def aep_gradients_chunked(
        self,
        wt_xs: jnp.ndarray,
        wt_ys: jnp.ndarray,
        ws_amb: jnp.ndarray,
        wd_amb: jnp.ndarray,
        ti_amb: jnp.ndarray | float | None = None,
        chunk_size: int = 100,
        probabilities: jnp.ndarray | None = None,
    ) -> tuple[float, tuple[jnp.ndarray, jnp.ndarray]]:
        """Computes AEP gradients with chunking to reduce memory usage.

        This method processes wind conditions in chunks to avoid memory issues
        with large time series. Each chunk is padded to the specified chunk_size
        for consistent JIT compilation.

        Args:
            ...: __call__ arguments passed through.
            chunk_size: Number of timestamps to process per chunk.
            probabilities: Optional probability weights for each timestamp.

        Returns:
            Tuple of (total_aep, (grad_x, grad_y)) where gradients are with
            respect to turbine positions.
        """
        assert chunk_size > 0, "Chunk size must be positive."

        n_timestamps = len(ws_amb)
        n_chunks = (n_timestamps + chunk_size - 1) // chunk_size

        total_aep = 0.0
        grad_x_accum = jnp.zeros_like(wt_xs, dtype=default_float_type())
        grad_y_accum = jnp.zeros_like(wt_ys, dtype=default_float_type())

        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, n_timestamps)
            actual_chunk_size = end_idx - start_idx

            def pad_chunk(
                arr: jnp.ndarray | float | None,
            ) -> jnp.ndarray | float | None:
                if arr is None:
                    return None
                if isinstance(arr, float):
                    return arr
                pad_width = ((0, chunk_size - actual_chunk_size),) + ((0, 0),) * (
                    arr.ndim - 1
                )
                return jnp.pad(arr, pad_width, "constant")

            ws_chunk = pad_chunk(ws_amb[start_idx:end_idx])
            wd_chunk = pad_chunk(wd_amb[start_idx:end_idx])
            ti_chunk = pad_chunk(
                ti_amb[start_idx:end_idx] if isinstance(ti_amb, jnp.ndarray) else ti_amb
            )
            prob_chunk = pad_chunk(
                None if probabilities is None else probabilities[start_idx:end_idx]
            )

            # Mask padded values in probabilities
            if prob_chunk is not None:
                mask = jnp.arange(chunk_size) < actual_chunk_size
                prob_chunk = jnp.where(mask, prob_chunk, 0.0)
            else:
                # Create uniform probabilities for actual data, zero for padding
                mask = jnp.arange(chunk_size) < actual_chunk_size
                prob_chunk = jnp.where(mask, 1.0 / actual_chunk_size, 0.0)

            chunk_aep, (grad_x, grad_y) = self._aep_gradients_chunked_static(
                self, wt_xs, wt_ys, ws_chunk, wd_chunk, ti_chunk, prob_chunk
            )

            # Accumulate
            if probabilities is None:
                # Scale by the ratio of actual chunk size to total timestamps
                weight = actual_chunk_size / n_timestamps
                total_aep += chunk_aep * weight
                grad_x_accum += grad_x * weight
                grad_y_accum += grad_y * weight
            else:
                # Probabilities already weighted in aep calculation
                total_aep += chunk_aep
                grad_x_accum += grad_x
                grad_y_accum += grad_y

        return total_aep, (grad_x_accum, grad_y_accum)


@partial(
    custom_vjp,
    nondiff_argnums=(0, 3, 4),  # older jax syntax for rocm
    # nondiff_argnames=["tol", "damp"],
)
def fixed_point(
    f: Callable,
    x_guess: tuple[jnp.ndarray, jnp.ndarray | None],
    ctx: SimulationContext,
    tol: float = 1e-6,
    damp: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Solves for a fixed point, i.e., a value `x` such that `f(x, ctx) = x`.

    In the context of wake modeling, this is used to determine the stable effective
    wind speeds within a wind farm, where the wind speed at each turbine is
    influenced by the wakes of others.

    The function is JAX-transformable and has a custom vector-Jacobian product (VJP)
    to allow for automatic differentiation. This is essential for gradient-based
    optimization of wind farm layouts or control strategies.

    The mathematical foundation for the custom VJP is the implicit function theorem.
    If we have a fixed-point equation `x_star = f(x_star, ctx)`, where `ctx` represents
    the simulation context, we can differentiate both sides to find the derivative
    of `x_star` with respect to parameters in `ctx`.

    Args:
        f: The function to iterate, which should accept the current estimate of the
            fixed point and the simulation context `ctx`, returning a new estimate.
        x_guess: The initial guess for the fixed point as a tuple of (wind_speed,
            turbulence_intensity) where turbulence_intensity can be None.
        ctx: The simulation context containing parameters and other data
            needed by the function `f`.
        tol: The tolerance for convergence. The iteration stops when the maximum
            absolute difference between successive wind speed estimates is below this value.
        damp: The damping factor for the updates (0.0 to 1.0), used to stabilize the
            iteration. A value of 0.0 means no damping (full update), while 1.0 means
            maximum damping (no update).

    Returns:
        The fixed point as a tuple of (wind_speed, turbulence_intensity) with the
        same structure as `x_guess`.
    """
    # Get array size from first leaf of x_guess
    # x_guess can be a tuple (ws_array, ti_array_or_none) or a scalar for general use
    first_leaf = x_guess[0] if isinstance(x_guess, tuple) else x_guess
    first_leaf_arr = jnp.atleast_1d(first_leaf)
    max_iter = max(20, first_leaf_arr.size)

    def cond_fun(carry: tuple) -> jnp.ndarray:
        x_prev, x, it = carry
        # Extract wind speed component for convergence check
        ws_prev = x_prev[0] if isinstance(x_prev, tuple) else x_prev
        ws = x[0] if isinstance(x, tuple) else x
        tol_cond = jnp.max(jnp.abs(ws_prev - ws)) > tol
        iter_cond = it < max_iter
        return jnp.logical_and(tol_cond, iter_cond)

    # Optimize for common case where damp=1.0 (no damping)
    if damp == 1.0:

        def body_fun(carry: tuple) -> tuple:
            _, x, it = carry
            x_new = f(x, ctx)
            return x, x_new, it + 1
    else:

        def body_fun(carry: tuple) -> tuple:
            _, x, it = carry
            x_new = f(x, ctx)
            x_damped = jax.tree.map(lambda n, o: damp * n + (1 - damp) * o, x_new, x)
            return x, x_damped, it + 1

    _, x_star, _ = while_loop(cond_fun, body_fun, (x_guess, f(x_guess, ctx), 0))
    return x_star


def fixed_point_fwd(
    f: Callable,
    x_guess: jnp.ndarray | tuple,
    ctx: SimulationContext,
    tol: float,
    damp: float,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray | None], tuple]:
    """Forward pass for the custom VJP of the fixed_point function.

    Returns:
        Tuple of (fixed_point_solution, residuals_for_backward_pass).
    """
    x_star = fixed_point(f, x_guess, ctx, tol=tol, damp=damp)
    return x_star, (ctx, x_star)


def fixed_point_rev(
    f: Callable, tol: float, damp: float, res: tuple, x_star_bar: jnp.ndarray | tuple
) -> tuple[tuple[jnp.ndarray, jnp.ndarray | None], SimulationContext]:
    """Reverse pass for the custom VJP of the fixed_point function.

    Uses the implicit function theorem to compute gradients efficiently by
    solving another fixed-point problem.

    Returns:
        Tuple of (gradient_wrt_x_guess, gradient_wrt_ctx).
    """
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
    damp: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Finds the fixed point of a function using iterative updates.

    This function is for debugging purposes only and is not JAX-transformable.
    It uses a standard Python while loop instead of `jax.lax.while_loop`, making
    it easier to debug with standard Python debuggers.
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
