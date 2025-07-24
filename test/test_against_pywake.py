import time

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jax import config as jcfg
from py_wake.deficit_models.noj import NOJDeficit
from py_wake.examples.data.hornsrev1 import Hornsrev1Site
from py_wake.site import XRSite
from py_wake.superposition_models import SquaredSum
from py_wake.wind_farm_models.engineering_models import All2AllIterative
from py_wake.wind_turbines import WindTurbines
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular

from pixwake import NOJModel, WakeSimulation, calculate_aep
from pixwake.types import Curve, Turbine

jcfg.update("jax_enable_x64", True)  # need float64 to match pywake


def test_noj_aep_and_gradients_equivalence_timeseries():
    # fmt: off
    ct_vals = np.array(
        [
            0.00, 0.00, 0.00,  # 0,1,2 m/s
            0.80,  # 3 m/s (cut-in)
            0.79, 0.77, 0.75,  # 4,5,6 m/s
            0.72, 0.68, 0.64,  # 7,8,9 m/s
            0.62, 0.61, 0.60,  # 10,11,12 m/s (rated)
            0.55, 0.50, 0.45,  # 13,14,15 m/s
            0.40, 0.35, 0.30,  # 16,17,18 m/s
            0.25, 0.20, 0.18,  # 19,20,21 m/s
            0.15, 0.12, 0.10,  # 22,23,24 m/s
            0.10,  # 25 m/s (cut-out)
        ]
    )
    power_vals = np.array(
        [
            0, 0, 0,  # 0,1,2 m/s
            100, 300, 600,  # 3,4,5 m/s
            1200, 1800, 2300,  # 6,7,8 m/s
            2700, 2900, 2950,  # 9,10,11 m/s
            3000, 3000, 3000,  # 12,13,14 m/s
            3000, 3000, 3000,  # 15,16,17 m/s
            3000, 3000, 3000,  # 18,19,20 m/s
            3000, 3000, 3000,  # 21,22,23 m/s 
            3000, 3000,  # 24,25 m/s
        ]
    )
    # fmt: on

    cutin_ws = 3.0
    cutout_ws = 25.0
    ct_pw_ws = np.arange(0.0, cutout_ws + 1.0, 1.0)
    ct_curve = np.stack([ct_pw_ws, ct_vals], axis=1)
    power_curve = np.stack([ct_pw_ws, power_vals], axis=1)

    width = 10
    length = 10
    x, y = np.meshgrid(
        np.linspace(0, width * 1e2, width),
        np.linspace(0, length * 1e2, length),
    )
    x, y = x.flatten(), y.flatten()
    turbines = [
        {
            "id": i,
            "x": x[i],
            "y": y[i],
            "hub_height": 100.0,
            "rotor_diameter": 120.0,
            "ct_curve": ct_curve,
            "power_curve": power_curve,
        }
        for i in range(width * length)
    ]

    wake_expansion_k = 0.1

    site = Hornsrev1Site(
        # p_wd=[1.0],  # Probability of wind direction
        # ti=standalone_site["turbulence_intensity"], # Hornsrev1Site default ti=0.10
        # air_density=standalone_site["air_density"] # Hornsrev1Site default air_density=1.225
    )

    names = [f"WT{t['id']}" for t in turbines]
    wt_x = [t["x"] for t in turbines]
    wt_y = [t["y"] for t in turbines]
    hub_heights = [t["hub_height"] for t in turbines]
    rotor_diameters = [t["rotor_diameter"] for t in turbines]

    _sa_ct_curve = turbines[0]["ct_curve"]
    _sa_power_curve = turbines[0]["power_curve"]
    power_values_W = _sa_power_curve[:, 1] * 1000
    wt_type_0_power_ct = PowerCtTabular(
        ws=_sa_power_curve[:, 0],  # ws array
        power=power_values_W,  # power array in W
        power_unit="w",
        ct=_sa_ct_curve[:, 1],  # ct array
    )

    windTurbines = WindTurbines(
        names=names,
        diameters=rotor_diameters,
        hub_heights=hub_heights,
        powerCtFunctions=[wt_type_0_power_ct] * len(names),
    )
    wake_model = NOJDeficit(k=wake_expansion_k, rotorAvgModel=None)

    wfm = All2AllIterative(
        site,
        windTurbines,
        wake_deficitModel=wake_model,
        superpositionModel=SquaredSum(),
    )

    n_timestamps = 142
    ws, wd = (
        np.random.uniform(cutin_ws, cutout_ws, size=n_timestamps),
        np.random.uniform(0, 360, size=n_timestamps),
    )

    n_cpu = 1
    s = time.time()
    sim_res = wfm(x=wt_x, y=wt_y, wd=wd, ws=ws, time=True, n_cpu=n_cpu)
    _ = wfm.aep_gradients(x=wt_x, y=wt_y, wd=wd, ws=ws, time=True, n_cpu=n_cpu)
    pywake_runtime = time.time() - s
    pywake_ws_eff = sim_res.WS_eff.values
    pw_dx, pw_dy = wfm.aep_gradients(
        x=wt_x, y=wt_y, wd=wd, ws=ws, time=True, n_cpu=n_cpu
    )

    model = NOJModel(k=wake_expansion_k, damp=1.0)
    sim = WakeSimulation(model)
    turbine = Turbine(
        rotor_diameter=windTurbines.diameter(),
        hub_height=100.0,
        power_curve=Curve(
            wind_speed=power_curve[:, 0], values=power_curve[:, 1]
        ),
        ct_curve=Curve(wind_speed=ct_curve[:, 0], values=ct_curve[:, 1]),
    )
    pixwake_ws_eff = sim(
        jnp.asarray(wt_x),
        jnp.asarray(wt_y),
        jnp.asarray(ws),
        jnp.asarray(wd),
        turbine,
    )
    rtol = 1e-3
    np.testing.assert_allclose(pixwake_ws_eff.T, pywake_ws_eff, rtol=rtol)
    np.testing.assert_allclose(
        calculate_aep(pixwake_ws_eff, turbine.power_curve), sim_res.aep().sum().values, rtol=rtol
    )

    grad_fn = jax.jit(
        jax.value_and_grad(
            lambda xx, yy: calculate_aep(
                sim(
                    xx,
                    yy,
                    ws,
                    wd,
                    turbine,
                ),
                turbine.power_curve,
            ),
            argnums=(0, 1),
        )
    )
    val, (dx, dy) = grad_fn(jnp.asarray(wt_x), jnp.asarray(wt_y))
    dx.block_until_ready()
    dy.block_until_ready()
    val.block_until_ready()

    s = time.time()
    val, (px_dx, px_dy) = grad_fn(jnp.asarray(wt_x), jnp.asarray(wt_y))
    dx.block_until_ready()
    dy.block_until_ready()
    val.block_until_ready()
    pixwake_runtime = time.time() - s

    assert np.isfinite(px_dx).all(), np.isnan(px_dx).sum()
    assert np.isfinite(px_dy).all(), np.isnan(px_dx).sum()
    np.testing.assert_allclose(px_dx, pw_dx, rtol=1e-2)
    np.testing.assert_allclose(px_dy, pw_dy, rtol=1e-2)

    assert (pywake_runtime / pixwake_runtime) > 5.0  # at least 5x speedup


def test_noj_aep_and_gradients_equivalence_with_site_frequencies():
    # fmt: off
    ct_vals = np.array(
        [
            0.00, 0.00, 0.00,  # 0,1,2 m/s
            0.80,  # 3 m/s (cut-in)
            0.79, 0.77, 0.75,  # 4,5,6 m/s
            0.72, 0.68, 0.64,  # 7,8,9 m/s
            0.62, 0.61, 0.60,  # 10,11,12 m/s (rated)
            0.55, 0.50, 0.45,  # 13,14,15 m/s
            0.40, 0.35, 0.30,  # 16,17,18 m/s
            0.25, 0.20, 0.18,  # 19,20,21 m/s
            0.15, 0.12, 0.10,  # 22,23,24 m/s
            0.10,  # 25 m/s (cut-out)
        ]
    )
    power_vals = np.array(
        [
            0, 0, 0,  # 0,1,2 m/s
            100, 300, 600,  # 3,4,5 m/s
            1200, 1800, 2300,  # 6,7,8 m/s
            2700, 2900, 2950,  # 9,10,11 m/s
            3000, 3000, 3000,  # 12,13,14 m/s
            3000, 3000, 3000,  # 15,16,17 m/s
            3000, 3000, 3000,  # 18,19,20 m/s
            3000, 3000, 3000,  # 21,22,23 m/s 
            3000, 3000,  # 24,25 m/s
        ]
    )
    # fmt: on

    cutin_ws = 4.0
    cutout_ws = 25.0
    ct_pw_ws = np.arange(0.0, cutout_ws + 1.0, 1.0)
    ct_curve = np.stack([ct_pw_ws, ct_vals], axis=1)
    power_curve = np.stack([ct_pw_ws, power_vals], axis=1)

    width = 8
    length = 8
    x, y = np.meshgrid(
        np.linspace(0, width * 1e2, width),
        np.linspace(0, length * 1e2, length),
    )
    x, y = x.flatten(), y.flatten()
    turbines = [
        {
            "id": i,
            "x": x[i],
            "y": y[i],
            "hub_height": 100.0,
            "rotor_diameter": 120.0,
            "ct_curve": ct_curve,
            "power_curve": power_curve,
        }
        for i in range(width * length)
    ]

    wake_expansion_k = 0.1

    # fmt:off
    wind_direction = np.array([
        0.0, 30.0, 60.0, 90.0, 120.0, 150.0,
        180.0, 210.0, 240.0, 270.0, 300.0, 330.0
    ])
    wind_speed = np.array([
        4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25
    ])
    sector_probability = np.array([
        0.06692, 0.07626, 0.07374, 0.06463, 0.04696, 0.04643,
        0.07672, 0.12233, 0.19147, 0.10080, 0.06927, 0.06447
    ])
    turbulence_intensity = np.array([
        0.144, 0.121, 0.106, 0.095, 0.087, 0.081, 0.076, 0.071,
        0.068, 0.065, 0.063, 0.060, 0.058, 0.057, 0.055, 0.054,
        0.053, 0.052, 0.051, 0.050, 0.049, 0.048
    ])
    weibull_a = np.array([
        9.08, 9.30, 9.18, 8.89, 8.13, 8.76,
        11.38, 12.58, 12.74, 10.80, 9.76, 9.63
    ])
    weibull_k = np.array([
        2.22, 2.26, 2.28, 2.28, 2.15, 2.11,
        2.13, 2.29, 2.43, 2.09, 2.01, 2.01
    ])
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

    names = [f"WT{t['id']}" for t in turbines]
    wt_x = [t["x"] for t in turbines]
    wt_y = [t["y"] for t in turbines]
    hub_heights = [t["hub_height"] for t in turbines]
    rotor_diameters = [t["rotor_diameter"] for t in turbines]

    _sa_ct_curve = turbines[0]["ct_curve"]
    _sa_power_curve = turbines[0]["power_curve"]
    power_values_W = _sa_power_curve[:, 1] * 1000
    wt_type_0_power_ct = PowerCtTabular(
        ws=_sa_power_curve[:, 0],  # ws array
        power=power_values_W,  # power array in W
        power_unit="w",
        ct=_sa_ct_curve[:, 1],  # ct array
    )

    windTurbines = WindTurbines(
        names=names,
        diameters=rotor_diameters,
        hub_heights=hub_heights,
        powerCtFunctions=[wt_type_0_power_ct] * len(names),
    )
    wake_model = NOJDeficit(k=wake_expansion_k, rotorAvgModel=None)

    wfm = All2AllIterative(
        site,
        windTurbines,
        wake_deficitModel=wake_model,
        superpositionModel=SquaredSum(),
    )

    ws = np.arange(cutin_ws, cutout_ws + 1)
    wd = np.arange(0, 360)
    s = time.time()
    sim_res = wfm(x=wt_x, y=wt_y, wd=wd, ws=ws)
    pw_dx, pw_dy = wfm.aep_gradients(x=wt_x, y=wt_y, wd=wd, ws=ws)

    pix_ws, pix_wd = jnp.meshgrid(ws, wd)
    pix_wd, pix_ws = pix_wd.flatten(), pix_ws.flatten()

    model = NOJModel(k=wake_expansion_k, damp=1.0)
    sim = WakeSimulation(model, mapping_strategy="map")
    turbine = Turbine(
        rotor_diameter=windTurbines.diameter(),
        hub_height=100.0,
        power_curve=Curve(
            wind_speed=power_curve[:, 0], values=power_curve[:, 1]
        ),
        ct_curve=Curve(wind_speed=ct_curve[:, 0], values=ct_curve[:, 1]),
    )
    pixwake_ws_eff = sim(
        jnp.asarray(wt_x),
        jnp.asarray(wt_y),
        pix_ws,
        pix_wd,
        turbine,
    )  # transpose to match pywake shape

    np.testing.assert_allclose(
        pixwake_ws_eff.T.reshape(sim_res.WS_eff.shape), sim_res.WS_eff.values, rtol=1e-3
    )

    P_ilk = site.local_wind().P_ilk
    pix_probs = P_ilk.reshape((1, pixwake_ws_eff.shape[0])).T

    np.testing.assert_allclose(
        sim_res.aep().sum().values, calculate_aep(pixwake_ws_eff, turbine.power_curve, probabilities=pix_probs)
    )

    n_cpu = 1
    s = time.time()
    aep = wfm(x=wt_x, y=wt_y, wd=wd, ws=ws, n_cpu=n_cpu, WS_eff=0).aep().sum()
    _ = wfm.aep_gradients(x=wt_x, y=wt_y, wd=wd, ws=ws, n_cpu=n_cpu, WS_eff=0)
    pywake_runtime = time.time() - s

    grad_and_value_fn = jax.jit(
        jax.value_and_grad(
            lambda xx, yy: calculate_aep(
                sim(
                    xx,
                    yy,
                    pix_ws,
                    pix_wd,
                    turbine,
                ),
                turbine.power_curve,
                pix_probs,
            ),
            argnums=(0, 1),
        )
    )
    val, (dx, dy) = grad_and_value_fn(jnp.asarray(wt_x), jnp.asarray(wt_y))
    dx.block_until_ready()
    dy.block_until_ready()
    val.block_until_ready()

    s = time.time()
    val, (px_dx, px_dy) = grad_and_value_fn(jnp.asarray(wt_x), jnp.asarray(wt_y))
    dx.block_until_ready()
    dy.block_until_ready()
    val.block_until_ready()
    pixwake_runtime = time.time() - s

    rtol = 1e-2
    assert np.isfinite(px_dx).all(), np.isnan(px_dx).sum()
    assert np.isfinite(px_dy).all(), np.isnan(px_dx).sum()
    np.testing.assert_allclose(px_dx, pw_dx, rtol=rtol)
    np.testing.assert_allclose(px_dy, pw_dy, rtol=rtol)

    assert (pywake_runtime / pixwake_runtime) > 5.0, (
        pywake_runtime / pixwake_runtime
    )  # at least 5x speedup
