import jax.numpy as jnp
import numpy as np
import pytest
from jax import config as jcfg
from py_wake.deficit_models import NOJDeficit as PyWakeNOJDeficit
from py_wake.deficit_models.gaussian import (
    BastankhahGaussianDeficit as PyWakeBastankhahGaussianDeficit,
)
from py_wake.deficit_models.gaussian import (
    NiayifarGaussianDeficit as PyWakeNiayifarGaussianDeficit,
)
from py_wake.examples.data.hornsrev1 import V80, Hornsrev1Site, ct_curve, power_curve
from py_wake.superposition_models import SquaredSum
from py_wake.turbulence_models import CrespoHernandez as PyWakeCrespoHernandez
from py_wake.wind_farm_models.engineering_models import All2AllIterative
from py_wake.wind_turbines import WindTurbine, WindTurbines
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular

from pixwake.core import Curve, Turbine, WakeSimulation
from pixwake.deficit import (
    BastankhahGaussianDeficit,
    NiayifarGaussianDeficit,
    NOJDeficit,
)
from pixwake.turbulence import CrespoHernandez

jcfg.update("jax_enable_x64", True)


def turbines(n_turbines):
    """Returns a V80 turbine object."""

    wind_turbine0 = V80()
    ws = power_curve[:, 0]
    power_curve1 = power_curve.copy()
    power_curve1[:, 1] = power_curve1[:, 1] * 2
    ct_curve1 = ct_curve.copy()
    ct_curve1[:, 1] = ct_curve1[:, 1]
    rd1 = wind_turbine0.diameter() * 2.0
    hh1 = wind_turbine0.hub_height() * 2.0
    wind_turbine1 = WindTurbine(
        name="V200",
        diameter=rd1,
        hub_height=hh1,
        powerCtFunction=PowerCtTabular(ws, power_curve1[:, 1], "w", ct_curve1[:, 1]),
    )

    px_turbine0 = Turbine(
        rotor_diameter=wind_turbine0.diameter(),
        hub_height=wind_turbine0.hub_height(),
        power_curve=Curve(
            ws=jnp.array(ws),
            values=jnp.array(wind_turbine0.power(ws)),
        ),
        ct_curve=Curve(
            ws=jnp.array(ws),
            values=jnp.array(wind_turbine0.ct(ws)),
        ),
    )
    px_turbine1 = Turbine(
        rotor_diameter=wind_turbine1.diameter(),
        hub_height=wind_turbine1.hub_height(),
        power_curve=Curve(
            ws=jnp.array(ws),
            values=jnp.array(wind_turbine1.power(ws)),
        ),
        ct_curve=Curve(
            ws=jnp.array(ws),
            values=jnp.array(wind_turbine1.ct(ws)),
        ),
    )

    wts_px = [px_turbine0 if i % 2 == 0 else px_turbine1 for i in range(n_turbines)]
    wts_pw = [wind_turbine0 if i % 2 == 0 else wind_turbine1 for i in range(n_turbines)]
    return wts_pw, wts_px


@pytest.mark.parametrize(
    ("pw_deficit_model", "pw_kwargs", "px_deficit_model", "px_kwargs"),
    [
        (
            PyWakeNOJDeficit,
            {},
            NOJDeficit,
            {},
        ),
        (
            PyWakeBastankhahGaussianDeficit,
            {},
            BastankhahGaussianDeficit,
            {"use_radius_mask": False},
        ),
        (
            PyWakeNiayifarGaussianDeficit,
            {"use_effective_ws": True, "use_effective_ti": True},
            NiayifarGaussianDeficit,
            {
                "use_effective_ws": True,
                "use_effective_ti": True,
                "use_radius_mask": False,
            },
        ),
        (
            PyWakeNiayifarGaussianDeficit,
            {"use_effective_ws": False, "use_effective_ti": False},
            NiayifarGaussianDeficit,
            {
                "use_effective_ws": False,
                "use_effective_ti": False,
                "use_radius_mask": False,
            },
        ),
    ],
)
def test_multiple_turbine_type_simulation(
    pw_deficit_model, pw_kwargs, px_deficit_model, px_kwargs
):
    n_turbines = 18
    pw_turbines, px_turbines = turbines(n_turbines)
    rotor_avg_model = None
    deficit_model = px_deficit_model(rotor_avg_model=rotor_avg_model, **px_kwargs)

    def _create_turbine_layout(width, length, spacing_x=3e2, spacing_y=3e2):
        x, y = jnp.meshgrid(
            jnp.linspace(0, width * spacing_x, width),
            jnp.linspace(0, length * spacing_y, length),
        )
        return x.flatten(), y.flatten()

    RD = np.mean([wt.diameter() for wt in pw_turbines])
    xs, ys = _create_turbine_layout(6, 3, spacing_y=5 * RD)
    assert len(xs) == n_turbines

    np.random.seed(0)
    ws = np.random.uniform(5.0, 15.0, 100)
    wd = np.random.uniform(0.0, 360.0, 100)
    ti = jnp.array([0.05])

    sim = WakeSimulation(
        px_turbines[:2], deficit_model, fpi_damp=1.0, turbulence=CrespoHernandez()
    )

    # Call without positions (uses layout positions)
    sim_res = sim(
        xs,
        ys,
        jnp.array(ws),
        jnp.array(wd),
        ti,
        wt_types=[
            px_turbines[0].type_id if i % 2 == 0 else px_turbines[1].type_id
            for i in range(n_turbines)
        ],
    )
    ws_eff_pixwake = sim_res.effective_ws

    site_pw = Hornsrev1Site()
    rotor_avg_model_pw = None
    wfm = All2AllIterative(
        site_pw,
        WindTurbines.from_WindTurbine_lst(pw_turbines),
        wake_deficitModel=pw_deficit_model(
            rotorAvgModel=rotor_avg_model_pw, **pw_kwargs
        ),
        superpositionModel=SquaredSum(),
        turbulenceModel=PyWakeCrespoHernandez(rotorAvgModel=None),
    )
    sim_res_pw = wfm(
        x=xs.tolist(),
        y=ys.tolist(),
        wd=wd.tolist(),
        ws=ws.tolist(),
        TI=ti.tolist(),
        WS_eff=0,
        time=True,
        type=[i % 2 for i in range(n_turbines)],
    )
    ws_eff_pywake = sim_res_pw.WS_eff_ilk

    np.testing.assert_allclose(
        ws_eff_pixwake, ws_eff_pywake.T.squeeze(0), atol=1e-6, rtol=1e-3
    )


def test_layout_optimization_pattern():
    """Test the recommended pattern for layout optimization."""
    from pixwake.deficit import BastankhahGaussianDeficit

    # Setup turbine types
    v80 = Turbine(
        rotor_diameter=80.0,
        hub_height=70.0,
        power_curve=Curve(ws=jnp.array([0, 25]), values=jnp.array([0, 2000])),
        ct_curve=Curve(ws=jnp.array([0, 25]), values=jnp.array([0.8, 0.8])),
    )

    v200 = Turbine(
        rotor_diameter=200.0,
        hub_height=150.0,
        power_curve=Curve(ws=jnp.array([0, 25]), values=jnp.array([0, 8000])),
        ct_curve=Curve(ws=jnp.array([0, 25]), values=jnp.array([0.8, 0.8])),
    )

    # Initial layout
    xs_init = jnp.array([0.0, 500.0, 1000.0, 0.0, 500.0, 1000.0])
    ys_init = jnp.array([0.0, 0.0, 0.0, 500.0, 500.0, 500.0])
    types = [v80.type_id if i % 2 == 0 else v200.type_id for i in range(6)]

    # Create simulation
    deficit = BastankhahGaussianDeficit(use_radius_mask=False)
    turbines = [v80, v200]
    turbine_types = types
    sim = WakeSimulation(turbines, deficit)

    # Wind conditions
    ws = jnp.array([8.0, 10.0, 12.0])
    wd = jnp.array([270.0, 270.0, 270.0])

    # Simulate with layout positions
    result1 = sim(xs_init, ys_init, ws, wd, wt_types=turbine_types)
    aep1 = result1.aep()

    # Simulate with perturbed positions (optimization step)
    xs_new = xs_init + jnp.array([10.0, -5.0, 0.0, 5.0, -10.0, 0.0])
    ys_new = ys_init + jnp.array([5.0, 0.0, -5.0, -10.0, 5.0, 0.0])
    result2 = sim(xs_new, ys_new, ws, wd, wt_types=turbine_types)
    aep2 = result2.aep()

    # Both should run without error
    assert aep1 > 0
    assert aep2 > 0
    assert len(result1.effective_ws) == 3  # 3 wind conditions
    assert result1.effective_ws.shape[1] == 6  # 6 turbines
