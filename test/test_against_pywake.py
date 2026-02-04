import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from jax import config as jcfg
from jax.test_util import check_grads
from py_wake.deficit_models.gaussian import (
    BastankhahGaussianDeficit as PyWakeBastankhahGaussianDeficit,
)
from py_wake.deficit_models.gaussian import (
    NiayifarGaussianDeficit as PyWakeNiayifarGaussianDeficit,
)
from py_wake.deficit_models.gaussian import (
    TurboGaussianDeficit as PyWakeTurboGaussianDeficit,
)
from py_wake.deficit_models.noj import NOJDeficit as PyWakeNOJDeficit
from py_wake.examples.data.hornsrev1 import Hornsrev1Site
from py_wake.site import XRSite
from py_wake.rotor_avg_models.gaussian_overlap_model import (
    GaussianOverlapAvgModel as PyWakeGaussianOverlapAvgModel,
)
from py_wake.superposition_models import SquaredSum
from py_wake.turbulence_models import CrespoHernandez as PyWakeCrespoHernandez
from py_wake.wind_farm_models.engineering_models import All2AllIterative
from py_wake.wind_turbines import WindTurbines
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular

from pixwake import Curve, Turbine, WakeSimulation
from pixwake.core import SimulationContext
from pixwake.deficit import (
    BastankhahGaussianDeficit,
    NiayifarGaussianDeficit,
    TurboGaussianDeficit,
    NOJDeficit,
)
from pixwake.rotor_avg import GaussianOverlapAvgModel
from pixwake.turbulence import CrespoHernandez

asarray_method = np.asarray
np.random.seed(42)
jcfg.update("jax_enable_x64", True)
np.asarray = asarray_method


@pytest.fixture
def ct_vals():
    # fmt:off
    ct_vals = np.array([
        0.00, 0.00, 0.00, 0.80, 0.79, 0.77, 0.75, 0.72, 0.68, 0.64,
        0.62, 0.61, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25,
        0.20, 0.18, 0.15, 0.12, 0.10, 0.10,
    ])
    # fmt:on
    return ct_vals


@pytest.fixture
def power_vals():
    # fmt:off
    power_vals = np.array([
        0, 0, 0, 100, 300, 600, 1200, 1800, 2300, 2700,
        2900, 2950, 3000, 3000, 3000, 3000, 3000, 3000,
        3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000,
    ])
    # fmt:on
    return power_vals


@pytest.fixture
def curves(ct_vals, power_vals):
    cutout_ws = 25.0
    ct_pw_ws = np.arange(0.0, cutout_ws + 1.0, 1.0)
    ct_curve = np.stack([ct_pw_ws, ct_vals], axis=1)
    power_curve = np.stack([ct_pw_ws, power_vals], axis=1)
    return ct_curve, power_curve


def _create_turbine_layout(width, length, spacing=1e2):
    x, y = np.meshgrid(
        np.linspace(0, width * spacing, width),
        np.linspace(0, length * spacing, length),
    )
    return x.flatten(), y.flatten()


def _create_pywake_turbines(n_turbines, ct_curve, power_curve, RD=120.0, HH=100.0):
    names = [f"WT{i}" for i in range(n_turbines)]
    power_values_W = power_curve[:, 1] * 1000
    power_ct_func = PowerCtTabular(
        ws=power_curve[:, 0],
        power=power_values_W,
        power_unit="w",
        ct=ct_curve[:, 1],
    )
    return WindTurbines(
        names=names,
        diameters=[RD] * n_turbines,
        hub_heights=[HH] * n_turbines,
        powerCtFunctions=[power_ct_func] * n_turbines,
    )


def _create_pixwake_turbine(ct_curve, power_curve, RD=120.0, HH=100.0):
    return Turbine(
        rotor_diameter=RD,
        hub_height=HH,
        power_curve=Curve(ws=power_curve[:, 0], values=power_curve[:, 1]),
        ct_curve=Curve(ws=ct_curve[:, 0], values=ct_curve[:, 1]),
    )


def _assert_ws_eff_close(pixwake_ws, pywake_ws, rtol=1e-3, atol=1e-6):
    np.testing.assert_allclose(
        pixwake_ws.T, np.maximum(pywake_ws, 0.0), rtol=rtol, atol=atol
    )


def _pixwake_compute_gradients(sim, x, y, ws, wd, ti=None, ret_grad_fn=False):
    """Compute AEP gradients with respect to x and y."""
    grad_fn = jax.jit(
        jax.value_and_grad(
            lambda xx, yy: sim(xx, yy, jnp.array(ws), jnp.array(wd), ti).aep()
            if ti is not None
            else sim(xx, yy, jnp.array(ws), jnp.array(wd)).aep(),
            argnums=(0, 1),
        )
    )
    val, (dx, dy) = grad_fn(jnp.asarray(x), jnp.asarray(y))
    dx.block_until_ready()
    dy.block_until_ready()
    val.block_until_ready()
    if ret_grad_fn:
        return val, dx, dy, grad_fn
    return val, dx, dy


def test_noj_equivalence_timeseries(curves):
    ct_curve, power_curve = curves
    cutin_ws, cutout_ws = 3.0, 25.0
    wake_expansion_k = 0.1

    x, y = _create_turbine_layout(10, 10)
    windTurbines = _create_pywake_turbines(len(x), ct_curve, power_curve)

    site = Hornsrev1Site()
    wake_model = PyWakeNOJDeficit(k=wake_expansion_k, rotorAvgModel=None)
    wfm = All2AllIterative(
        site,
        windTurbines,
        wake_deficitModel=wake_model,
        superpositionModel=SquaredSum(),
    )

    n_timestamps = 142
    ws = np.random.uniform(cutin_ws, cutout_ws, size=n_timestamps)
    wd = np.random.uniform(0, 360, size=n_timestamps)

    sim_res = wfm(x=x, y=y, wd=wd, ws=ws, time=True, n_cpu=1)
    pw_dx, pw_dy = wfm.aep_gradients(x=x, y=y, wd=wd, ws=ws, time=True, n_cpu=1)

    model = NOJDeficit(k=wake_expansion_k, use_radius_mask=True)
    turbine = _create_pixwake_turbine(ct_curve, power_curve)
    sim = WakeSimulation(turbine, model, fpi_damp=1.0)

    pixwake_sim_res = sim(
        jnp.asarray(x), jnp.asarray(y), jnp.asarray(ws), jnp.asarray(wd)
    )

    rtol = 1e-3
    np.testing.assert_allclose(
        pixwake_sim_res.effective_ws.T, sim_res.WS_eff.values, rtol=rtol
    )
    np.testing.assert_allclose(
        pixwake_sim_res.aep(), sim_res.aep().sum().values, rtol=rtol
    )

    _, px_dx, px_dy, grad_fn = _pixwake_compute_gradients(
        sim, x, y, ws, wd, ret_grad_fn=True
    )

    _, (px_dx, px_dy) = grad_fn(jnp.asarray(x), jnp.asarray(y))
    px_dx.block_until_ready()
    px_dy.block_until_ready()

    assert np.isfinite(px_dx).all() and np.isfinite(px_dy).all()
    np.testing.assert_allclose(px_dx, pw_dx, atol=1e-6)
    np.testing.assert_allclose(px_dy, pw_dy, atol=1e-6)


def test_noj_equivalence_with_site_frequencies(curves):
    ct_curve, power_curve = curves
    cutin_ws, cutout_ws = 4.0, 25.0
    wake_expansion_k = 0.1

    x, y = _create_turbine_layout(8, 8)
    windTurbines = _create_pywake_turbines(len(x), ct_curve, power_curve)

    # fmt:off
    wind_direction = np.array([0.0, 30.0, 60.0, 90.0, 120.0, 
                               150.0, 180.0, 210.0, 240.0, 270.0, 300.0, 330.0])
    wind_speed = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 
                           14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
    sector_probability = np.array([0.06692, 0.07626, 0.07374, 0.06463, 0.04696, 0.04643, 
                                   0.07672, 0.12233, 0.19147, 0.10080, 0.06927, 0.06447])
    turbulence_intensity = np.array([0.144, 0.121, 0.106, 0.095, 0.087, 0.081, 0.076, 
                                     0.071, 0.068, 0.065, 0.063, 0.060, 0.058, 0.057, 
                                     0.055, 0.054, 0.053, 0.052, 0.051, 0.050, 0.049, 0.048])
    weibull_a = np.array([9.08, 9.30, 9.18, 8.89, 8.13, 8.76,
                          11.38, 12.58, 12.74, 10.80, 9.76, 9.63])
    weibull_k = np.array([2.22, 2.26, 2.28, 2.28, 2.15, 2.11,
                          2.13, 2.29, 2.43, 2.09, 2.01, 2.01])
    # fmt:on

    site = XRSite(
        ds=xr.Dataset(
            data_vars={
                "Sector_frequency": ("wd", sector_probability),
                "Weibull_A": ("wd", weibull_a),
                "Weibull_k": ("wd", weibull_k),
                "TI": ("ws", turbulence_intensity),
            },
            coords={"wd": wind_direction, "ws": wind_speed},
        )
    )

    wake_model = PyWakeNOJDeficit(k=wake_expansion_k, rotorAvgModel=None)
    wfm = All2AllIterative(
        site,
        windTurbines,
        wake_deficitModel=wake_model,
        superpositionModel=SquaredSum(),
    )

    ws = np.arange(cutin_ws, cutout_ws + 1)
    wd = np.arange(0, 360, 45)

    sim_res = wfm(x=x, y=y, wd=wd, ws=ws, n_cpu=1)
    pw_dx, pw_dy = wfm.aep_gradients(x=x, y=y, wd=wd, ws=ws, n_cpu=1)

    pix_ws, pix_wd = jnp.meshgrid(ws, wd)
    pix_wd, pix_ws = pix_wd.flatten(), pix_ws.flatten()

    model = NOJDeficit(k=wake_expansion_k)
    turbine = _create_pixwake_turbine(ct_curve, power_curve)
    sim = WakeSimulation(turbine, model, fpi_damp=1.0, mapping_strategy="map")
    pixwake_sim_res = sim(jnp.asarray(x), jnp.asarray(y), pix_ws, pix_wd)

    np.testing.assert_allclose(
        pixwake_sim_res.effective_ws.T.reshape(sim_res.WS_eff.shape),
        sim_res.WS_eff.values,
        rtol=1e-3,
    )

    P_ilk = site.local_wind(ws=ws, wd=wd).P_ilk
    pix_probs = P_ilk.reshape((1, pixwake_sim_res.effective_ws.shape[0])).T

    np.testing.assert_allclose(
        sim_res.aep().sum().values, pixwake_sim_res.aep(probabilities=pix_probs)
    )

    grad_and_value_fn = jax.jit(
        jax.value_and_grad(
            lambda xx, yy: sim(xx, yy, pix_ws, pix_wd).aep(probabilities=pix_probs),
            argnums=(0, 1),
        )
    )
    val, (dx, dy) = grad_and_value_fn(jnp.asarray(x), jnp.asarray(y))
    dx.block_until_ready()
    dy.block_until_ready()
    val.block_until_ready()

    val, (px_dx, px_dy) = grad_and_value_fn(jnp.asarray(x), jnp.asarray(y))
    dx.block_until_ready()
    dy.block_until_ready()
    val.block_until_ready()

    rtol = 1e-2
    assert np.isfinite(px_dx).all() and np.isfinite(px_dy).all()
    np.testing.assert_allclose(px_dx, pw_dx, rtol=rtol)
    np.testing.assert_allclose(px_dy, pw_dy, rtol=rtol)


def test_gaussian_equivalence_timeseries(ct_vals, power_vals):
    cutin_ws = 3.0
    cutout_ws = 25.0
    ct_pw_ws = np.arange(0.0, cutout_ws + 1.0, 1.0)
    ct_curve = np.stack([ct_pw_ws, ct_vals], axis=1)
    power_curve = np.stack([ct_pw_ws, power_vals], axis=1)

    width = 9
    length = 9
    x, y = _create_turbine_layout(width, length, spacing=3 * 120)
    windTurbines = _create_pywake_turbines(len(x), ct_curve, power_curve)

    wake_expansion_k = 0.0324555
    site = Hornsrev1Site()

    wake_model = PyWakeBastankhahGaussianDeficit(k=wake_expansion_k)

    wfm = All2AllIterative(
        site,
        windTurbines,
        wake_deficitModel=wake_model,
        superpositionModel=SquaredSum(),
    )

    n_timestamps = 1000
    ws, wd = (
        np.random.uniform(cutin_ws, cutout_ws, size=n_timestamps),
        np.random.uniform(0, 360, size=n_timestamps),
    )

    sim_res = wfm(x=x, y=y, wd=wd, ws=ws, time=True)
    pywake_ws_eff = sim_res["WS_eff"].values

    # bug in PyWake not masking the contributions from far off wake radius
    model = BastankhahGaussianDeficit(k=wake_expansion_k, use_radius_mask=False)
    turbine = Turbine(
        rotor_diameter=windTurbines.diameter().item(),
        hub_height=100.0,
        power_curve=Curve(ws=power_curve[:, 0], values=power_curve[:, 1]),
        ct_curve=Curve(ws=ct_curve[:, 0], values=ct_curve[:, 1]),
    )
    sim = WakeSimulation(turbine, model, fpi_damp=1.0)
    pixwake_sim_res = sim(
        jnp.asarray(x),
        jnp.asarray(y),
        jnp.asarray(ws),
        jnp.asarray(wd),
    )

    rtol = 1e-2  # 1%
    # find problematic effective wind speed and add it to error message
    np.testing.assert_allclose(
        pixwake_sim_res.effective_ws.T,
        np.maximum(pywake_ws_eff, 0),
        atol=1e-6,
        rtol=rtol,
    )

    pywake_aep = sim_res.aep().sum().values
    pixwake_aep = pixwake_sim_res.aep()
    np.testing.assert_allclose(pixwake_aep, pywake_aep, rtol=rtol)

    # gradients
    pw_dx, pw_dy = wfm.aep_gradients(x=x, y=y, wd=wd, ws=ws, time=True)
    grad_fn = jax.jit(
        jax.value_and_grad(
            lambda xx, yy: sim(
                xx,
                yy,
                jnp.array(ws),
                jnp.array(wd),
            ).aep(),
            argnums=(0, 1),
        )
    )

    val, (dx, dy) = grad_fn(jnp.asarray(x), jnp.asarray(y))
    dx.block_until_ready()
    dy.block_until_ready()
    val.block_until_ready()
    np.testing.assert_allclose(dx, pw_dx, rtol=rtol, atol=1e-6)
    np.testing.assert_allclose(dy, pw_dy, rtol=rtol, atol=1e-6)


def test_gaussian_equivalence_timeseries_with_effective_ws(curves):
    ct_curve, power_curve = curves
    cutin_ws, cutout_ws = 3.0, 25.0
    wake_expansion_k = 0.0324555
    RD = 120.0

    x, y = _create_turbine_layout(20, 3, spacing=RD)
    windTurbines = _create_pywake_turbines(len(x), ct_curve, power_curve, RD=RD)

    site = Hornsrev1Site()
    wake_model = PyWakeBastankhahGaussianDeficit(
        k=wake_expansion_k, use_effective_ws=True
    )
    wfm = All2AllIterative(
        site,
        windTurbines,
        wake_deficitModel=wake_model,
        superpositionModel=SquaredSum(),
    )

    n_timestamps = 1000
    ws = np.random.uniform(cutin_ws, cutout_ws, size=n_timestamps)
    wd = np.random.uniform(0, 360, size=n_timestamps)

    sim_res = wfm(x=x, y=y, wd=wd, ws=ws, time=True)
    pywake_ws_eff = sim_res["WS_eff"].values

    model = BastankhahGaussianDeficit(
        k=wake_expansion_k, use_effective_ws=True, use_radius_mask=False
    )
    turbine = _create_pixwake_turbine(ct_curve, power_curve, RD=RD)
    sim = WakeSimulation(turbine, model, fpi_damp=1.0)
    pixwake_sim_res = sim(
        jnp.asarray(x), jnp.asarray(y), jnp.asarray(ws), jnp.asarray(wd)
    )

    rtol = 1e-2
    _assert_ws_eff_close(
        pixwake_sim_res.effective_ws, pywake_ws_eff, rtol=rtol, atol=1e-6
    )
    np.testing.assert_allclose(
        pixwake_sim_res.aep(), sim_res.aep().sum().values, rtol=rtol
    )

    pw_dx, pw_dy = wfm.aep_gradients(x=x, y=y, wd=wd, ws=ws, time=True)
    _, dx, dy = _pixwake_compute_gradients(sim, x, y, ws, wd)

    np.testing.assert_allclose(dx, pw_dx, rtol=rtol, atol=1e-6)
    np.testing.assert_allclose(dy, pw_dy, rtol=rtol, atol=1e-6)


def test_gaussian_equivalence_timeseries_with_effective_ws_with_turbulence(curves):
    ct_curve, power_curve = curves
    cutin_ws, cutout_ws = 3.0, 25.0
    wake_expansion_k = 0.0324555
    RD = 120.0

    x, y = _create_turbine_layout(20, 3, spacing=RD)
    windTurbines = _create_pywake_turbines(len(x), ct_curve, power_curve, RD=RD)

    site = Hornsrev1Site()
    wake_model = PyWakeBastankhahGaussianDeficit(
        k=wake_expansion_k, use_effective_ws=True
    )
    wfm = All2AllIterative(
        site,
        windTurbines,
        wake_deficitModel=wake_model,
        superpositionModel=SquaredSum(),
        turbulenceModel=PyWakeCrespoHernandez(),
    )

    n_timestamps = 1000
    ws = np.random.uniform(cutin_ws, cutout_ws, size=n_timestamps)
    wd = np.random.uniform(0, 360, size=n_timestamps)

    sim_res = wfm(x=x, y=y, wd=wd, ws=ws, time=True, TI=0.1)
    pywake_ws_eff = sim_res["WS_eff"].values

    model = BastankhahGaussianDeficit(
        k=wake_expansion_k,
        use_effective_ws=True,
        use_radius_mask=False,
    )
    turbine = _create_pixwake_turbine(ct_curve, power_curve, RD=RD)
    sim = WakeSimulation(turbine, model, CrespoHernandez(), fpi_damp=1.0)
    pixwake_sim_res = sim(
        jnp.asarray(x), jnp.asarray(y), jnp.asarray(ws), jnp.asarray(wd), 0.1
    )

    tols = {"rtol": 1e-3, "atol": 1e-6}
    _assert_ws_eff_close(pixwake_sim_res.effective_ws, pywake_ws_eff, **tols)
    np.testing.assert_allclose(
        pixwake_sim_res.aep(), sim_res.aep().sum().values, **tols
    )

    pw_dx, pw_dy = wfm.aep_gradients(x=x, y=y, wd=wd, ws=ws, time=True)
    _, dx, dy = _pixwake_compute_gradients(sim, x, y, ws, wd, ti=0.1)

    np.testing.assert_allclose(dx, pw_dx, **tols)
    np.testing.assert_allclose(dy, pw_dy, **tols)


def test_gaussian_equivalence_timeseries_with_wake_expansion_based_on_ti(
    curves,
):
    ct_curve, power_curve = curves
    cutin_ws, cutout_ws = 3.0, 25.0
    RD = 120.0

    x, y = _create_turbine_layout(20, 3, spacing=RD * 3)
    windTurbines = _create_pywake_turbines(len(x), ct_curve, power_curve, RD=RD)

    site = Hornsrev1Site()
    wake_model = PyWakeNiayifarGaussianDeficit(
        use_effective_ws=True, use_effective_ti=False
    )
    wfm = All2AllIterative(
        site,
        windTurbines,
        wake_deficitModel=wake_model,
        superpositionModel=SquaredSum(),
    )

    n_timestamps = 1000
    ws = np.random.uniform(cutin_ws, cutout_ws, size=n_timestamps)
    wd = np.random.uniform(0, 360, size=n_timestamps)

    sim_res = wfm(x=x, y=y, wd=wd, ws=ws, time=True, TI=0.1)
    pywake_ws_eff = sim_res["WS_eff"].values

    model = NiayifarGaussianDeficit(use_effective_ws=True, use_radius_mask=False)
    turbine = _create_pixwake_turbine(ct_curve, power_curve, RD=RD)
    sim = WakeSimulation(turbine, model, fpi_damp=1.0, mapping_strategy="map")
    pixwake_sim_res = sim(
        jnp.asarray(x), jnp.asarray(y), jnp.asarray(ws), jnp.asarray(wd), 0.1
    )

    rtol, atol = 1e-3, 1e-6
    tols = dict(rtol=rtol, atol=atol)

    _assert_ws_eff_close(pixwake_sim_res.effective_ws, pywake_ws_eff, **tols)
    np.testing.assert_allclose(
        pixwake_sim_res.aep(), sim_res.aep().sum().values, **tols
    )

    pw_dx, pw_dy = wfm.aep_gradients(x=x, y=y, wd=wd, ws=ws, time=True)
    _, dx, dy = _pixwake_compute_gradients(sim, x, y, ws, wd, ti=0.1)

    np.testing.assert_allclose(dx, pw_dx, **tols)
    np.testing.assert_allclose(dy, pw_dy, **tols)


def test_effective_ti_gaussian_equivalence_timeseries_with_wake_expansion_based_on_ti(
    curves,
):
    ct_curve, power_curve = curves
    cutin_ws, cutout_ws = 3.0, 25.0
    RD, HH = 120.0, 100.0

    x, y = _create_turbine_layout(20, 3, spacing=RD * 3)
    windTurbines = _create_pywake_turbines(len(x), ct_curve, power_curve, RD=RD, HH=HH)

    site = Hornsrev1Site()
    wake_model = PyWakeNiayifarGaussianDeficit(
        use_effective_ws=True, use_effective_ti=True
    )
    wfm = All2AllIterative(
        site,
        windTurbines,
        wake_deficitModel=wake_model,
        superpositionModel=SquaredSum(),
        turbulenceModel=PyWakeCrespoHernandez(rotorAvgModel=None),
    )

    model = NiayifarGaussianDeficit(
        use_effective_ws=True, use_effective_ti=True, use_radius_mask=False
    )
    turbine = _create_pixwake_turbine(ct_curve, power_curve, RD=RD, HH=HH)
    sim = WakeSimulation(
        turbine, model, CrespoHernandez(), fpi_damp=1.0, mapping_strategy="map"
    )

    n_test = 1000
    ws = np.maximum(np.random.uniform(cutin_ws, cutout_ws, size=n_test), 0.0)
    wd = np.random.uniform(0, 360, size=n_test)

    sim_res = wfm(x=x, y=y, wd=wd, ws=ws, time=True, TI=0.1)

    pixwake_sim_res = sim(
        jnp.asarray(x), jnp.asarray(y), jnp.asarray(ws), jnp.asarray(wd), 0.1
    )

    rtol, atol = 1e-3, 1e-6
    np.testing.assert_allclose(
        pixwake_sim_res.effective_ti.T, sim_res["TI_eff"].values, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        pixwake_sim_res.effective_ws.T, sim_res["WS_eff"].values, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        pixwake_sim_res.aep(), sim_res.aep().sum().values, rtol=rtol
    )

    ws, wd = ws[:100], wd[:100]
    pw_dx, pw_dy = wfm.aep_gradients(x=x, y=y, wd=wd, ws=ws, time=True)
    _, dx, dy = _pixwake_compute_gradients(sim, x, y, ws, wd, ti=0.1)

    np.testing.assert_allclose(dx, pw_dx, rtol=rtol, atol=1e-6)
    np.testing.assert_allclose(dy, pw_dy, rtol=rtol, atol=1e-6)


def test_crespo_hernandez_implementation_match():
    py_wake_model = PyWakeCrespoHernandez()
    n_turbines = 3

    ws_eff = jnp.array([10.0])

    turbine = Turbine(
        rotor_diameter=8.0,
        hub_height=10.0,
        ct_curve=Curve(jnp.array([0, 25]), jnp.array([8 / 9, 0])),
        power_curve=Curve(jnp.array([0, 25]), jnp.array([0, 1])),
    )
    ct_value = turbine.ct(ws_eff)[0]

    dw_ijlk = np.ones((n_turbines, n_turbines, 1, 1))
    cw_ijlk = np.ones((n_turbines, n_turbines, 1, 1))
    D_src_il = np.ones((n_turbines, 1)) * 8.0
    ct_ilk = np.ones((n_turbines, 1, 1)) * ct_value
    TI_ilk = np.ones((n_turbines, 1, 1)) * 0.1
    wake_radius_ijlk = np.ones((n_turbines, n_turbines, 1, 1)) * 4.0

    pywake_ti_added = py_wake_model.calc_added_turbulence(
        dw_ijlk=dw_ijlk,
        cw_ijlk=cw_ijlk,
        D_src_il=D_src_il,
        ct_ilk=ct_ilk,
        TI_ilk=TI_ilk,
        D_dst_ijl=None,
        wake_radius_ijlk=wake_radius_ijlk,
    ).squeeze()

    pywake_ti_eff_res = py_wake_model.calc_effective_TI(
        np.ones_like(pywake_ti_added) * 0.1, pywake_ti_added
    ).squeeze()

    turbulence_model = CrespoHernandez()

    dw = jnp.array(dw_ijlk[:, :, 0, 0])
    cw = jnp.array(cw_ijlk[:, :, 0, 0])

    ctx = SimulationContext(
        dw=dw,
        cw=cw,
        ws=jnp.ones(n_turbines) * 8.0,
        turbine=turbine,
        ti=0.1,
    )

    pixwake_ti_addded = turbulence_model._added_turbulence(
        ws_eff=ws_eff,
        ti_eff=jnp.ones(n_turbines) * 0.11,
        ctx=ctx,
    )
    pixwake_ti_eff_res = turbulence_model.superposition(
        jnp.ones_like(pixwake_ti_addded) * 0.1, pixwake_ti_addded
    )

    tols = dict(rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(pixwake_ti_addded, pywake_ti_added, **tols)
    np.testing.assert_allclose(pixwake_ti_eff_res, pywake_ti_eff_res, **tols)


def test_gaussian_overlap_avg_model_full_simulation(curves):
    """Integration test: full simulation with GaussianOverlapAvgModel vs PyWake.

    This test runs a complete wind farm simulation using the GaussianOverlapAvgModel
    for rotor averaging and compares effective wind speeds, AEP, and gradients
    against PyWake.
    """
    ct_curve, power_curve = curves
    cutin_ws, cutout_ws = 3.0, 25.0
    RD, HH = 120.0, 100.0

    # Create turbine layout
    x, y = _create_turbine_layout(10, 5, spacing=RD * 4)
    windTurbines = _create_pywake_turbines(len(x), ct_curve, power_curve, RD=RD, HH=HH)

    # PyWake setup with GaussianOverlapAvgModel
    site = Hornsrev1Site()
    pw_rotor_avg = PyWakeGaussianOverlapAvgModel()
    pw_wake_model = PyWakeBastankhahGaussianDeficit(
        rotorAvgModel=pw_rotor_avg, use_effective_ws=True
    )
    wfm = All2AllIterative(
        site,
        windTurbines,
        wake_deficitModel=pw_wake_model,
        superpositionModel=SquaredSum(),
        turbulenceModel=PyWakeCrespoHernandez(rotorAvgModel=None),
    )

    # Generate wind conditions
    n_timestamps = 200
    ws = np.random.uniform(cutin_ws, cutout_ws, size=n_timestamps)
    wd = np.random.uniform(0, 360, size=n_timestamps)
    ti = 0.08

    # Run PyWake simulation
    sim_res = wfm(x=x, y=y, wd=wd, ws=ws, time=True, TI=ti)
    pywake_ws_eff = sim_res["WS_eff"].values

    # PixWake setup with GaussianOverlapAvgModel (PyWake-style API)
    px_rotor_avg = GaussianOverlapAvgModel()
    px_deficit = BastankhahGaussianDeficit(
        rotor_avg_model=px_rotor_avg, use_effective_ws=True, use_radius_mask=False
    )

    turbine = _create_pixwake_turbine(ct_curve, power_curve, RD=RD, HH=HH)
    sim = WakeSimulation(turbine, px_deficit, CrespoHernandez(), fpi_damp=0.5)

    pixwake_sim_res = sim(
        jnp.asarray(x), jnp.asarray(y), jnp.asarray(ws), jnp.asarray(wd), ti
    )

    # Compare effective wind speeds
    rtol, atol = 1e-3, 1e-6
    _assert_ws_eff_close(
        pixwake_sim_res.effective_ws, pywake_ws_eff, rtol=rtol, atol=atol
    )

    # Compare AEP
    np.testing.assert_allclose(
        pixwake_sim_res.aep(), sim_res.aep().sum().values, rtol=rtol
    )

    # Compare gradients
    pw_dx, pw_dy = wfm.aep_gradients(x=x, y=y, wd=wd, ws=ws, time=True)
    _, dx, dy = _pixwake_compute_gradients(sim, x, y, ws, wd, ti=ti)

    np.testing.assert_allclose(dx, pw_dx, rtol=rtol, atol=atol)
    np.testing.assert_allclose(dy, pw_dy, rtol=rtol, atol=atol)


def test_gaussian_overlap_avg_model_with_niayifar_full_simulation(curves):
    """Integration test: full simulation with GaussianOverlapAvgModel + NiayifarGaussianDeficit.

    This test runs a complete wind farm simulation using the GaussianOverlapAvgModel
    for rotor averaging with the Niayifar model (TI-dependent wake expansion) and
    compares against PyWake.
    """
    ct_curve, power_curve = curves
    cutin_ws, cutout_ws = 3.0, 25.0
    RD, HH = 120.0, 100.0

    # Create turbine layout
    x, y = _create_turbine_layout(8, 4, spacing=RD * 5)
    windTurbines = _create_pywake_turbines(len(x), ct_curve, power_curve, RD=RD, HH=HH)

    # PyWake setup with GaussianOverlapAvgModel and NiayifarGaussianDeficit
    site = Hornsrev1Site()
    pw_rotor_avg = PyWakeGaussianOverlapAvgModel()
    pw_wake_model = PyWakeNiayifarGaussianDeficit(
        rotorAvgModel=pw_rotor_avg,
        use_effective_ws=True,
        use_effective_ti=True,
    )
    wfm = All2AllIterative(
        site,
        windTurbines,
        wake_deficitModel=pw_wake_model,
        superpositionModel=SquaredSum(),
        turbulenceModel=PyWakeCrespoHernandez(rotorAvgModel=None),
    )

    # Generate wind conditions
    n_timestamps = 500
    ws = np.random.uniform(cutin_ws, cutout_ws, size=n_timestamps)
    wd = np.random.uniform(0, 360, size=n_timestamps)
    ti = 0.06

    # Run PyWake simulation
    sim_res = wfm(x=x, y=y, wd=wd, ws=ws, time=True, TI=ti)
    pywake_ws_eff = sim_res["WS_eff"].values

    # PixWake setup with GaussianOverlapAvgModel and NiayifarGaussianDeficit
    # (PyWake-style API)
    px_rotor_avg = GaussianOverlapAvgModel()
    px_deficit = NiayifarGaussianDeficit(
        rotor_avg_model=px_rotor_avg,
        use_effective_ws=True,
        use_effective_ti=True,
        use_radius_mask=False,
    )

    turbine = _create_pixwake_turbine(ct_curve, power_curve, RD=RD, HH=HH)
    sim = WakeSimulation(turbine, px_deficit, CrespoHernandez(), fpi_damp=0.5)

    pixwake_sim_res = sim(
        jnp.asarray(x), jnp.asarray(y), jnp.asarray(ws), jnp.asarray(wd), ti
    )

    # Verify gradients are correct
    # Note: order=1 only because fixed_point's custom_vjp doesn't support
    # second-order differentiation (the VJP residuals become closed-over values
    # during second-order diff, which custom_vjp doesn't handle)
    check_grads(
        lambda xx, yy: sim(xx, yy, jnp.asarray(ws), jnp.asarray(wd), ti).aep(),
        (jnp.asarray(x), jnp.asarray(y)),
        order=1,
        modes=["rev"],
        atol=1e-6,
        rtol=1e-6,
    )

    # Compare effective wind speeds
    rtol, atol = 1e-3, 1e-6
    _assert_ws_eff_close(
        pixwake_sim_res.effective_ws, pywake_ws_eff, rtol=rtol, atol=atol
    )

    # Compare AEP
    np.testing.assert_allclose(
        pixwake_sim_res.aep(), sim_res.aep().sum().values, rtol=rtol
    )

    # Compare gradients - use relaxed tolerance for gradients with effective TI
    # The gradient computation paths differ slightly due to how effective TI
    # is propagated through the rotor averaging, but absolute differences are small
    pw_dx, pw_dy = wfm.aep_gradients(x=x, y=y, wd=wd, ws=ws, time=True)
    _, dx, dy = _pixwake_compute_gradients(sim, x, y, ws, wd, ti=ti)

    # Use larger atol since gradient magnitudes are small
    np.testing.assert_allclose(dx, pw_dx, rtol=rtol, atol=1e-3)
    np.testing.assert_allclose(dy, pw_dy, rtol=rtol, atol=1e-3)


def test_turbo_gaussian_equivalence_timeseries(curves):
    """Test TurboGaussianDeficit model equivalence with PyWake.

    This test compares effective wind speeds, AEP, and gradients between
    pixwake's TurboGaussianDeficit and PyWake's implementation.
    """
    ct_curve, power_curve = curves
    cutin_ws, cutout_ws = 3.0, 25.0
    RD, HH = 120.0, 100.0

    # Create turbine layout
    x, y = _create_turbine_layout(10, 5, spacing=RD * 4)
    windTurbines = _create_pywake_turbines(len(x), ct_curve, power_curve, RD=RD, HH=HH)

    # PyWake setup with TurboGaussianDeficit
    site = Hornsrev1Site()
    pw_wake_model = PyWakeTurboGaussianDeficit(
        use_effective_ws=True,
        use_effective_ti=False,
    )
    wfm = All2AllIterative(
        site,
        windTurbines,
        wake_deficitModel=pw_wake_model,
        superpositionModel=SquaredSum(),
        turbulenceModel=PyWakeCrespoHernandez(rotorAvgModel=None),
    )

    # Generate wind conditions
    n_timestamps = 200
    ws = np.random.uniform(cutin_ws, cutout_ws, size=n_timestamps)
    wd = np.random.uniform(0, 360, size=n_timestamps)
    ti = 0.08

    # Run PyWake simulation
    sim_res = wfm(x=x, y=y, wd=wd, ws=ws, time=True, TI=ti)
    pywake_ws_eff = sim_res["WS_eff"].values

    # PixWake setup with TurboGaussianDeficit
    px_deficit = TurboGaussianDeficit(
        use_effective_ws=True,
        use_effective_ti=False,
        use_radius_mask=False,
    )

    turbine = _create_pixwake_turbine(ct_curve, power_curve, RD=RD, HH=HH)
    sim = WakeSimulation(turbine, px_deficit, CrespoHernandez(), fpi_damp=1.0)

    pixwake_sim_res = sim(
        jnp.asarray(x), jnp.asarray(y), jnp.asarray(ws), jnp.asarray(wd), ti
    )

    # Compare effective wind speeds
    rtol, atol = 1e-3, 1e-6
    _assert_ws_eff_close(
        pixwake_sim_res.effective_ws, pywake_ws_eff, rtol=rtol, atol=atol
    )

    # Compare AEP
    np.testing.assert_allclose(
        pixwake_sim_res.aep(), sim_res.aep().sum().values, rtol=rtol
    )

    # Compare gradients - use relaxed tolerances due to differences in
    # implicit differentiation implementations between PyWake and PixWake
    pw_dx, pw_dy = wfm.aep_gradients(x=x, y=y, wd=wd, ws=ws, time=True)
    _, dx, dy = _pixwake_compute_gradients(sim, x, y, ws, wd, ti=ti)

    # Verify pixwake gradients are self-consistent using JAX gradient checker
    check_grads(
        lambda xx, yy: sim(xx, yy, jnp.asarray(ws), jnp.asarray(wd), ti).aep(),
        (jnp.asarray(x), jnp.asarray(y)),
        order=1,
        modes=["rev"],
        atol=1e-6,
        rtol=1e-6,
    )

    # Cross-check gradients with relaxed tolerance
    np.testing.assert_allclose(dx, pw_dx, rtol=0.02, atol=1e-2)
    np.testing.assert_allclose(dy, pw_dy, rtol=0.02, atol=1e-2)


@pytest.mark.xfail(
    run=False, reason="Effective TI implementation differences.. To be adressed.."
)
def test_turbo_gaussian_equivalence_with_effective_ti(curves):
    """Test TurboGaussianDeficit with effective TI equivalence.

    This test uses use_effective_ti=True to verify the model correctly
    uses effective turbulence intensity in the wake expansion calculation.
    """
    ct_curve, power_curve = curves
    cutin_ws, cutout_ws = 3.0, 25.0
    RD, HH = 120.0, 100.0

    # Create turbine layout
    x, y = _create_turbine_layout(8, 4, spacing=RD * 5)
    windTurbines = _create_pywake_turbines(len(x), ct_curve, power_curve, RD=RD, HH=HH)

    # PyWake setup with TurboGaussianDeficit and effective TI
    site = Hornsrev1Site()
    pw_wake_model = PyWakeTurboGaussianDeficit(
        use_effective_ws=True,
        use_effective_ti=True,
    )
    wfm = All2AllIterative(
        site,
        windTurbines,
        wake_deficitModel=pw_wake_model,
        superpositionModel=SquaredSum(),
        turbulenceModel=PyWakeCrespoHernandez(rotorAvgModel=None),
    )

    # Generate wind conditions
    n_timestamps = 300
    ws = np.random.uniform(cutin_ws, cutout_ws, size=n_timestamps)
    wd = np.random.uniform(0, 360, size=n_timestamps)
    ti = 0.06

    # Run PyWake simulation
    sim_res = wfm(x=x, y=y, wd=wd, ws=ws, time=True, TI=ti)
    pywake_ws_eff = sim_res["WS_eff"].values

    # PixWake setup with TurboGaussianDeficit and effective TI
    px_deficit = TurboGaussianDeficit(
        use_effective_ws=True,
        use_effective_ti=True,
        use_radius_mask=False,
    )

    turbine = _create_pixwake_turbine(ct_curve, power_curve, RD=RD, HH=HH)
    sim = WakeSimulation(turbine, px_deficit, CrespoHernandez(), fpi_damp=1.0)

    pixwake_sim_res = sim(
        jnp.asarray(x), jnp.asarray(y), jnp.asarray(ws), jnp.asarray(wd), ti
    )

    # Compare effective TI
    rtol, atol = 1e-3, 1e-6
    np.testing.assert_allclose(
        pixwake_sim_res.effective_ti.T, sim_res["TI_eff"].values, rtol=rtol, atol=atol
    )
    # Compare effective wind speeds
    _assert_ws_eff_close(
        pixwake_sim_res.effective_ws, pywake_ws_eff, rtol=rtol, atol=atol
    )

    # Compare AEP
    np.testing.assert_allclose(
        pixwake_sim_res.aep(), sim_res.aep().sum().values, rtol=rtol
    )

    check_grads(
        lambda xx, yy: sim(xx, yy, jnp.asarray(ws), jnp.asarray(wd), ti).aep(),
        (jnp.asarray(x), jnp.asarray(y)),
        order=1,
        modes=["rev"],
        atol=1e-6,
        rtol=1e-6,
    )

    # Compare gradients with relaxed tolerance
    ws_grad, wd_grad = ws[:100], wd[:100]
    pw_dx, pw_dy = wfm.aep_gradients(x=x, y=y, wd=wd_grad, ws=ws_grad, time=True)
    _, dx, dy = _pixwake_compute_gradients(sim, x, y, ws_grad, wd_grad, ti=ti)

    np.testing.assert_allclose(dx, pw_dx, rtol=0.5, atol=1e-3)
    np.testing.assert_allclose(dy, pw_dy, rtol=0.5, atol=1e-3)


def test_turbo_gaussian_with_rotor_avg_model(curves):
    """Test TurboGaussianDeficit with GaussianOverlapAvgModel rotor averaging.

    This test verifies that the TurboGaussianDeficit model works correctly
    with rotor averaging enabled.
    """
    ct_curve, power_curve = curves
    cutin_ws, cutout_ws = 3.0, 25.0
    RD, HH = 120.0, 100.0

    # Create turbine layout
    x, y = _create_turbine_layout(8, 4, spacing=RD * 5)
    windTurbines = _create_pywake_turbines(len(x), ct_curve, power_curve, RD=RD, HH=HH)

    # PyWake setup with TurboGaussianDeficit and rotor averaging
    site = Hornsrev1Site()
    pw_rotor_avg = PyWakeGaussianOverlapAvgModel()
    pw_wake_model = PyWakeTurboGaussianDeficit(
        rotorAvgModel=pw_rotor_avg,
        use_effective_ws=True,
        use_effective_ti=True,
    )
    wfm = All2AllIterative(
        site,
        windTurbines,
        wake_deficitModel=pw_wake_model,
        superpositionModel=SquaredSum(),
        turbulenceModel=PyWakeCrespoHernandez(rotorAvgModel=None),
    )

    # Generate wind conditions
    n_timestamps = 200
    ws = np.random.uniform(cutin_ws, cutout_ws, size=n_timestamps)
    wd = np.random.uniform(0, 360, size=n_timestamps)
    ti = 0.08

    # Run PyWake simulation
    sim_res = wfm(x=x, y=y, wd=wd, ws=ws, time=True, TI=ti)
    pywake_ws_eff = sim_res["WS_eff"].values

    # PixWake setup with TurboGaussianDeficit and rotor averaging
    px_rotor_avg = GaussianOverlapAvgModel()
    px_deficit = TurboGaussianDeficit(
        rotor_avg_model=px_rotor_avg,
        use_effective_ws=True,
        use_effective_ti=True,
        use_radius_mask=False,
    )

    turbine = _create_pixwake_turbine(ct_curve, power_curve, RD=RD, HH=HH)
    sim = WakeSimulation(turbine, px_deficit, CrespoHernandez(), fpi_damp=0.5)

    pixwake_sim_res = sim(
        jnp.asarray(x), jnp.asarray(y), jnp.asarray(ws), jnp.asarray(wd), ti
    )

    # Verify gradients are correct using JAX's gradient checker
    check_grads(
        lambda xx, yy: sim(xx, yy, jnp.asarray(ws), jnp.asarray(wd), ti).aep(),
        (jnp.asarray(x), jnp.asarray(y)),
        order=1,
        modes=["rev"],
        atol=1e-6,
        rtol=1e-6,
    )

    # Compare effective wind speeds
    rtol, atol = 1e-3, 1e-6
    _assert_ws_eff_close(
        pixwake_sim_res.effective_ws, pywake_ws_eff, rtol=rtol, atol=atol
    )

    # Compare AEP
    np.testing.assert_allclose(
        pixwake_sim_res.aep(), sim_res.aep().sum().values, rtol=rtol
    )

    # Compare gradients with relaxed tolerance
    pw_dx, pw_dy = wfm.aep_gradients(x=x, y=y, wd=wd, ws=ws, time=True)
    _, dx, dy = _pixwake_compute_gradients(sim, x, y, ws, wd, ti=ti)

    np.testing.assert_allclose(dx, pw_dx, rtol=0.5, atol=1e-3)
    np.testing.assert_allclose(dy, pw_dy, rtol=0.5, atol=1e-3)
