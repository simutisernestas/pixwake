import time

import jax
import jax.numpy as jnp
import numpy as np

asarray_method = np.asarray
import pytest
import xarray as xr
from jax import config as jcfg
from py_wake.deficit_models.gaussian import (
    BastankhahGaussianDeficit as PyWakeBastankhahGaussianDeficit,
)
from py_wake.deficit_models.gaussian import (
    NiayifarGaussianDeficit as PyWakeNiayifarGaussianDeficit,
)
from py_wake.deficit_models.noj import NOJDeficit as PyWakeNOJDeficit
from py_wake.examples.data.hornsrev1 import Hornsrev1Site
from py_wake.site import XRSite
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
    NOJDeficit,
)
from pixwake.turbulence import CrespoHernandez

np.random.seed(42)
jcfg.update("jax_enable_x64", True)
np.asarray = asarray_method


@pytest.fixture
def ct_vals():
    # fmt: off
    ct_vals = np.array([
        0.00, 0.00, 0.00, 0.80, 0.79, 0.77, 0.75, 0.72, 0.68, 0.64,
        0.62, 0.61, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25,
        0.20, 0.18, 0.15, 0.12, 0.10, 0.10,
    ])
    # fmt: on
    return ct_vals


@pytest.fixture
def power_vals():
    # fmt: off
    power_vals = np.array([
        0, 0, 0, 100, 300, 600, 1200, 1800, 2300, 2700,
        2900, 2950, 3000, 3000, 3000, 3000, 3000, 3000,
        3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000,
    ])
    # fmt: on
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


def _create_pywake_turbines(x, y, ct_curve, power_curve, RD=120.0, HH=100.0):
    names = [f"WT{i}" for i in range(len(x))]
    power_values_W = power_curve[:, 1] * 1000
    power_ct_func = PowerCtTabular(
        ws=power_curve[:, 0],
        power=power_values_W,
        power_unit="w",
        ct=ct_curve[:, 1],
    )
    return WindTurbines(
        names=names,
        diameters=[RD] * len(x),
        hub_heights=[HH] * len(x),
        powerCtFunctions=[power_ct_func] * len(x),
    )


def _create_pixwake_turbine(ct_curve, power_curve, RD=120.0, HH=100.0):
    return Turbine(
        rotor_diameter=RD,
        hub_height=HH,
        power_curve=Curve(wind_speed=power_curve[:, 0], values=power_curve[:, 1]),
        ct_curve=Curve(wind_speed=ct_curve[:, 0], values=ct_curve[:, 1]),
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
    windTurbines = _create_pywake_turbines(x, y, ct_curve, power_curve)

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

    s = time.time()
    sim_res = wfm(x=x, y=y, wd=wd, ws=ws, time=True, n_cpu=1)
    pw_dx, pw_dy = wfm.aep_gradients(x=x, y=y, wd=wd, ws=ws, time=True, n_cpu=1)
    pywake_runtime = time.time() - s

    model = NOJDeficit(k=wake_expansion_k)
    turbine = _create_pixwake_turbine(ct_curve, power_curve)
    sim = WakeSimulation(model, turbine, fpi_damp=1.0)

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

    s = time.time()
    _, (px_dx, px_dy) = grad_fn(jnp.asarray(x), jnp.asarray(y))
    px_dx.block_until_ready()
    px_dy.block_until_ready()
    pixwake_runtime = time.time() - s

    assert np.isfinite(px_dx).all() and np.isfinite(px_dy).all()
    np.testing.assert_allclose(px_dx, pw_dx, atol=1e-5)
    np.testing.assert_allclose(px_dy, pw_dy, atol=1e-5)

    speedup = pywake_runtime / pixwake_runtime
    assert speedup > 2.0, speedup


def test_noj_equivalence_with_site_frequencies(curves):
    ct_curve, power_curve = curves
    cutin_ws, cutout_ws = 4.0, 25.0
    wake_expansion_k = 0.1

    x, y = _create_turbine_layout(8, 8)
    windTurbines = _create_pywake_turbines(x, y, ct_curve, power_curve)

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

    s = time.time()
    sim_res = wfm(x=x, y=y, wd=wd, ws=ws, n_cpu=1, WS_eff=0)
    pw_dx, pw_dy = wfm.aep_gradients(x=x, y=y, wd=wd, ws=ws, n_cpu=1, WS_eff=0)
    pywake_runtime = time.time() - s

    pix_ws, pix_wd = jnp.meshgrid(ws, wd)
    pix_wd, pix_ws = pix_wd.flatten(), pix_ws.flatten()

    model = NOJDeficit(k=wake_expansion_k)
    turbine = _create_pixwake_turbine(ct_curve, power_curve)
    sim = WakeSimulation(model, turbine, fpi_damp=1.0, mapping_strategy="map")
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

    s = time.time()
    val, (px_dx, px_dy) = grad_and_value_fn(jnp.asarray(x), jnp.asarray(y))
    dx.block_until_ready()
    dy.block_until_ready()
    val.block_until_ready()
    pixwake_runtime = time.time() - s

    rtol = 1e-2
    assert np.isfinite(px_dx).all() and np.isfinite(px_dy).all()
    np.testing.assert_allclose(px_dx, pw_dx, rtol=rtol)
    np.testing.assert_allclose(px_dy, pw_dy, rtol=rtol)

    speedup = pywake_runtime / pixwake_runtime
    assert speedup > 2.0, speedup


def test_gaussian_equivalence_timeseries(ct_vals, power_vals):
    cutin_ws = 3.0
    cutout_ws = 25.0
    ct_pw_ws = np.arange(0.0, cutout_ws + 1.0, 1.0)
    ct_curve = np.stack([ct_pw_ws, ct_vals], axis=1)
    power_curve = np.stack([ct_pw_ws, power_vals], axis=1)

    width = 9
    length = 9
    x, y = _create_turbine_layout(width, length, spacing=3 * 120)
    windTurbines = _create_pywake_turbines(x, y, ct_curve, power_curve)

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
        power_curve=Curve(wind_speed=power_curve[:, 0], values=power_curve[:, 1]),
        ct_curve=Curve(wind_speed=ct_curve[:, 0], values=ct_curve[:, 1]),
    )
    sim = WakeSimulation(model, turbine, fpi_damp=1.0)
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
        atol=1e-5,
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
    np.testing.assert_allclose(dx, pw_dx, rtol=rtol, atol=1e-5)
    np.testing.assert_allclose(dy, pw_dy, rtol=rtol, atol=1e-5)


def test_gaussian_equivalence_timeseries_with_effective_ws(curves):
    ct_curve, power_curve = curves
    cutin_ws, cutout_ws = 3.0, 25.0
    wake_expansion_k = 0.0324555
    RD = 120.0

    x, y = _create_turbine_layout(20, 3, spacing=RD)
    windTurbines = _create_pywake_turbines(x, y, ct_curve, power_curve, RD=RD)

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
    sim = WakeSimulation(model, turbine, fpi_damp=1.0)
    pixwake_sim_res = sim(
        jnp.asarray(x), jnp.asarray(y), jnp.asarray(ws), jnp.asarray(wd)
    )

    rtol = 1e-2
    _assert_ws_eff_close(
        pixwake_sim_res.effective_ws, pywake_ws_eff, rtol=rtol, atol=1e-5
    )
    np.testing.assert_allclose(
        pixwake_sim_res.aep(), sim_res.aep().sum().values, rtol=rtol
    )

    pw_dx, pw_dy = wfm.aep_gradients(x=x, y=y, wd=wd, ws=ws, time=True)
    _, dx, dy = _pixwake_compute_gradients(sim, x, y, ws, wd)

    np.testing.assert_allclose(dx, pw_dx, rtol=rtol)
    np.testing.assert_allclose(dy, pw_dy, rtol=rtol)


def test_gaussian_equivalence_timeseries_with_effective_ws_with_turbulence(curves):
    ct_curve, power_curve = curves
    cutin_ws, cutout_ws = 3.0, 25.0
    wake_expansion_k = 0.0324555
    RD = 120.0

    x, y = _create_turbine_layout(20, 3, spacing=RD)
    windTurbines = _create_pywake_turbines(x, y, ct_curve, power_curve, RD=RD)

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
        turbulence_model=CrespoHernandez(),
    )
    turbine = _create_pixwake_turbine(ct_curve, power_curve, RD=RD)
    sim = WakeSimulation(model, turbine, fpi_damp=1.0)
    pixwake_sim_res = sim(
        jnp.asarray(x), jnp.asarray(y), jnp.asarray(ws), jnp.asarray(wd), 0.1
    )

    rtol = 1e-2
    _assert_ws_eff_close(
        pixwake_sim_res.effective_ws, pywake_ws_eff, rtol=rtol, atol=1e-5
    )
    np.testing.assert_allclose(
        pixwake_sim_res.aep(), sim_res.aep().sum().values, rtol=rtol
    )

    pw_dx, pw_dy = wfm.aep_gradients(x=x, y=y, wd=wd, ws=ws, time=True)
    _, dx, dy = _pixwake_compute_gradients(sim, x, y, ws, wd)

    np.testing.assert_allclose(dx, pw_dx, rtol=rtol)
    np.testing.assert_allclose(dy, pw_dy, rtol=rtol)


def test_crespo_hernandez_implementation_match():
    py_wake_model = PyWakeCrespoHernandez()
    n_turbines = 3

    dw_ijlk = np.ones((n_turbines, n_turbines, 1, 1))
    cw_ijlk = np.ones((n_turbines, n_turbines, 1, 1))
    D_src_il = np.ones((n_turbines, 1)) * 8.0
    ct_ilk = np.ones((n_turbines, 1, 1)) * 8.0 / 9.0
    TI_ilk = np.ones((n_turbines, 1, 1)) * 0.1
    wake_radius_ijlk = np.ones((n_turbines, n_turbines, 1, 1)) * 4.0

    pywake_ti_res = py_wake_model.calc_added_turbulence(
        dw_ijlk=dw_ijlk,
        cw_ijlk=cw_ijlk,
        D_src_il=D_src_il,
        ct_ilk=ct_ilk,
        TI_ilk=TI_ilk,
        D_dst_ijl=None,
        wake_radius_ijlk=wake_radius_ijlk,
    ).squeeze()

    pywake_ti_eff_res = py_wake_model.calc_effective_TI(
        np.ones_like(pywake_ti_res) * 0.1, pywake_ti_res
    ).squeeze()

    turbulence_model = CrespoHernandez()
    turbine = Turbine(
        rotor_diameter=8.0,
        hub_height=10.0,
        ct_curve=Curve(jnp.array([0, 25]), jnp.array([8 / 9, 0])),
        power_curve=Curve(jnp.array([0, 25]), jnp.array([0, 1])),
    )
    ctx = SimulationContext(
        dw=jnp.array([0.0, 200.0]),
        cw=jnp.array([0.0, 0.0]),
        ws=jnp.array([8.0, 8.0]),
        wd=jnp.array([0.0, 0.0]),
        turbine=turbine,
        ti=0.1,
    )

    dw = jnp.array(dw_ijlk[:, :, 0, 0])
    cw = jnp.array(cw_ijlk[:, :, 0, 0])
    wake_radius = jnp.array(wake_radius_ijlk[:, :, 0, 0])
    ct = jnp.array(ct_ilk[:, 0, 0])

    pixwake_ti_res = turbulence_model.calc_added_turbulence(
        ctx,
        dw=dw,
        cw=cw,
        wake_radius=wake_radius,
        ct=ct,
    )
    pixwake_ti_eff_res = turbulence_model.superposition(
        jnp.ones_like(pixwake_ti_res) * 0.1, pixwake_ti_res
    )

    tols = dict(rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(pixwake_ti_res, pywake_ti_res, **tols)
    np.testing.assert_allclose(pixwake_ti_eff_res, pywake_ti_eff_res, **tols)


def test_gaussian_equivalence_timeseries_with_wake_expansion_based_on_ti(
    curves,
):
    ct_curve, power_curve = curves
    cutin_ws, cutout_ws = 3.0, 25.0
    RD = 120.0

    x, y = _create_turbine_layout(20, 3, spacing=RD * 3)
    windTurbines = _create_pywake_turbines(x, y, ct_curve, power_curve, RD=RD)

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
    sim = WakeSimulation(model, turbine, fpi_damp=1.0, mapping_strategy="map")
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
    windTurbines = _create_pywake_turbines(x, y, ct_curve, power_curve, RD=RD, HH=HH)

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
        use_effective_ws=True, use_effective_ti=True, turbulence_model=CrespoHernandez()
    )
    turbine = _create_pixwake_turbine(ct_curve, power_curve, RD=RD, HH=HH)

    n_test = 1000
    ws = np.maximum(np.random.uniform(cutin_ws, cutout_ws, size=n_test), 0.0)
    wd = np.random.uniform(0, 360, size=n_test)

    sim_res = wfm(x=x, y=y, wd=wd, ws=ws, time=True, TI=0.1, WS_eff=0)

    sim = WakeSimulation(model, turbine, fpi_damp=1.0, mapping_strategy="map")
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

    np.testing.assert_allclose(dx, pw_dx, rtol=rtol)
    np.testing.assert_allclose(dy, pw_dy, rtol=rtol)
