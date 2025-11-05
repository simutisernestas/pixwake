import jax.numpy as jnp
import numpy as np
from py_wake.deficit_models.gaussian import (
    BastankhahGaussianDeficit as PyWakeBastankhahGaussianDeficit,
)
from numpy import newaxis as na
from py_wake.examples.data.hornsrev1 import V80, Hornsrev1Site, power_curve, ct_curve
from py_wake.rotor_avg_models import CGIRotorAvg as PyWakeCGIRotorAvg
from py_wake.superposition_models import SquaredSum
from py_wake.wind_farm_models.engineering_models import All2AllIterative

from pixwake.core import Curve, Turbine, WakeSimulation, SimulationContext
from pixwake.deficit.gaussian import BastankhahGaussianDeficit
from pixwake.rotor_avg import CGIRotorAvg


def v80_jax():
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


def test_cgi_rotor_avg_against_pywake():
    """Test the CGI rotor averaging model against the PyWake implementation."""
    wind_turbines = v80_jax()
    rotor_avg_model = CGIRotorAvg(n_points=21)
    deficit_model = BastankhahGaussianDeficit(
        rotor_avg_model=rotor_avg_model,
        use_radius_mask=False,
        k=0.0324555,
        use_effective_ws=True,
    )

    sim = WakeSimulation(wind_turbines, deficit_model, fpi_damp=1.0)
    sim_res = sim(
        wt_xs=jnp.array([0, 200]),
        wt_ys=jnp.array([0, 0]),
        wd=jnp.array([270.0]),
        ws_amb=jnp.array([10.0]),
    )
    ws_eff_pixwake = sim_res.effective_ws[0, 1]

    site_pw = Hornsrev1Site()
    wind_turbines_pw = V80()
    rotor_avg_model_pw = PyWakeCGIRotorAvg(21)
    wfm = All2AllIterative(
        site_pw,
        wind_turbines_pw,
        wake_deficitModel=PyWakeBastankhahGaussianDeficit(
            rotorAvgModel=rotor_avg_model_pw, k=0.0324555, use_effective_ws=True
        ),
        superpositionModel=SquaredSum(),
    )
    sim_res_pw = wfm(
        x=[0, 200],
        y=[0, 0],
        wd=270,
        ws=10,
    )
    ws_eff_pywake = sim_res_pw.WS_eff_ilk[1, 0, 0]

    assert jnp.allclose(ws_eff_pixwake, ws_eff_pywake, atol=1e-5)
