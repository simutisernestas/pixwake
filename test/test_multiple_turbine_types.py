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
from py_wake.wind_turbines import WindTurbine
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
            wind_speed=jnp.array(ws),
            values=jnp.array(wind_turbine0.power(ws)),
        ),
        ct_curve=Curve(
            wind_speed=jnp.array(ws),
            values=jnp.array(wind_turbine0.ct(ws)),
        ),
    )
    px_turbine1 = Turbine(
        rotor_diameter=wind_turbine1.diameter(),
        hub_height=wind_turbine1.hub_height(),
        power_curve=Curve(
            wind_speed=jnp.array(ws),
            values=jnp.array(wind_turbine1.power(ws)),
        ),
        ct_curve=Curve(
            wind_speed=jnp.array(ws),
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
    pw_turbines, px_turbines = turbines(10)
    rotor_avg_model = None
    deficit_model = px_deficit_model(rotor_avg_model=rotor_avg_model, **px_kwargs)

    def _create_turbine_layout(width, length, spacing_x=3e2, spacing_y=3e2):
        x, y = jnp.meshgrid(
            jnp.linspace(0, width * spacing_x, width),
            jnp.linspace(0, length * spacing_y, length),
        )
        return x.flatten().tolist(), y.flatten().tolist()

    RD = np.mean([wt.diameter() for wt in pw_turbines])
    xs, ys = _create_turbine_layout(6, 3, spacing_y=5 * RD)
    ws = np.random.uniform(5.0, 15.0, 100).tolist()
    wd = np.random.uniform(0.0, 360.0, 100).tolist()
    ti = [0.05]

    sim = WakeSimulation(
        px_turbines, deficit_model, fpi_damp=0.5, turbulence=CrespoHernandez()
    )
    sim_res = sim(
        wt_xs=jnp.array(xs),
        wt_ys=jnp.array(ys),
        wd=jnp.array(wd),
        ws_amb=jnp.array(ws),
        ti=jnp.array(ti),
    )
    ws_eff_pixwake = sim_res.effective_ws

    site_pw = Hornsrev1Site()
    rotor_avg_model_pw = None
    wfm = All2AllIterative(
        site_pw,
        pw_turbines,
        wake_deficitModel=pw_deficit_model(
            rotorAvgModel=rotor_avg_model_pw, **pw_kwargs
        ),
        superpositionModel=SquaredSum(),
        turbulenceModel=PyWakeCrespoHernandez(rotorAvgModel=None),
    )

    sim_res_pw = wfm(x=xs, y=ys, wd=wd, ws=ws, TI=ti, WS_eff=0, time=True)
    ws_eff_pywake = sim_res_pw.WS_eff_ilk

    np.testing.assert_allclose(
        ws_eff_pixwake, ws_eff_pywake.T.squeeze(0), atol=1e-6, rtol=1e-3
    )
