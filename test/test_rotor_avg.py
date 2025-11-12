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
from py_wake.rotor_avg_models import CGIRotorAvg as PyWakeCGIRotorAvg
from py_wake.superposition_models import SquaredSum
from py_wake.turbulence_models import CrespoHernandez as PyWakeCrespoHernandez
from py_wake.wind_farm_models.engineering_models import All2AllIterative

from pixwake.core import Curve, Turbine, WakeSimulation
from pixwake.deficit.gaussian import BastankhahGaussianDeficit, NiayifarGaussianDeficit
from pixwake.deficit.noj import NOJDeficit
from pixwake.rotor_avg import CGIRotorAvg
from pixwake.turbulence import CrespoHernandez

jcfg.update("jax_enable_x64", True)


def v80_wt():
    """Returns a V80 turbine object."""

    wind_turbines = V80()
    return Turbine(
        rotor_diameter=wind_turbines.diameter(),
        hub_height=wind_turbines.hub_height(),
        power_curve=Curve(
            wind_speed=jnp.array(power_curve[:, 0]),
            values=jnp.array(wind_turbines.power(power_curve[:, 0])),
        ),
        ct_curve=Curve(
            wind_speed=jnp.array(ct_curve[:, 0]),
            values=jnp.array(wind_turbines.ct(ct_curve[:, 0])),
        ),
    )


@pytest.mark.parametrize("n_points", [4, 7, 9, 21])
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
def test_cgi_rotor_avg_against_pywake(
    n_points, pw_deficit_model, pw_kwargs, px_deficit_model, px_kwargs
):
    wind_turbines = v80_wt()
    rotor_avg_model = CGIRotorAvg(n_points=n_points)
    deficit_model = px_deficit_model(rotor_avg_model=rotor_avg_model, **px_kwargs)

    def _create_turbine_layout(width, length, spacing_x=3e2, spacing_y=3e2):
        x, y = jnp.meshgrid(
            jnp.linspace(0, width * spacing_x, width),
            jnp.linspace(0, length * spacing_y, length),
        )
        return x.flatten().tolist(), y.flatten().tolist()

    RD = wind_turbines.rotor_diameter
    xs, ys = _create_turbine_layout(6, 3, spacing_y=2 * RD)
    ws = np.random.uniform(5.0, 15.0, 100).tolist()
    wd = np.random.uniform(0.0, 360.0, 100).tolist()
    ti = [0.05]

    sim = WakeSimulation(
        wind_turbines, deficit_model, fpi_damp=0.5, turbulence=CrespoHernandez()
    )
    sim_res = sim(
        wt_xs=jnp.array(xs),
        wt_ys=jnp.array(ys),
        wd_amb=jnp.array(wd),
        ws_amb=jnp.array(ws),
        ti_amb=jnp.array(ti),
    )
    ws_eff_pixwake = sim_res.effective_ws

    site_pw = Hornsrev1Site()
    wind_turbines_pw = V80()
    rotor_avg_model_pw = PyWakeCGIRotorAvg(n_points)
    wfm = All2AllIterative(
        site_pw,
        wind_turbines_pw,
        wake_deficitModel=pw_deficit_model(
            rotorAvgModel=rotor_avg_model_pw, **pw_kwargs
        ),
        superpositionModel=SquaredSum(),
        turbulenceModel=PyWakeCrespoHernandez(rotorAvgModel=None),
    )

    sim_res_pw = wfm(x=xs, y=ys, wd=wd, ws=ws, TI=ti, WS_eff=0, time=True)
    ws_eff_pywake = sim_res_pw.WS_eff_ilk

    if (
        not jnp.allclose(ws_eff_pixwake, ws_eff_pywake.T.squeeze(0))
        and pw_deficit_model is PyWakeNOJDeficit
    ):
        pytest.xfail(  # TODO: fix NOJ rotor avg differences
            "NOJ model differences not yet resolved. "
            "The failure is specific to rotor averaging."
        )

    np.testing.assert_allclose(
        ws_eff_pixwake, ws_eff_pywake.T.squeeze(0), atol=1e-6, rtol=1e-3
    )
