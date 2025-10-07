import jax.numpy as jnp
import numpy as np
from jax import config as jcfg
from py_wake.deficit_models.gaussian import (
    NiayifarGaussianDeficit as PyWakeNiayifarGaussianDeficit,
)
from py_wake.examples.data.hornsrev1 import Hornsrev1Site
from py_wake.superposition_models import SquaredSum
from py_wake.turbulence_models import CrespoHernandez as PyWakeCrespoHernandez
from py_wake.wind_farm_models.engineering_models import All2AllIterative
from py_wake.wind_turbines import WindTurbines
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular

from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import NiayifarGaussianDeficit
from pixwake.turbulence import CrespoHernandez

jcfg.update("jax_enable_x64", False)

ct_vals = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.8,
        0.79,
        0.77,
        0.75,
        0.72,
        0.68,
        0.64,
        0.62,
        0.61,
        0.6,
        0.55,
        0.5,
        0.45,
        0.4,
        0.35,
        0.3,
        0.25,
        0.2,
        0.18,
        0.15,
        0.12,
        0.1,
        0.1,
    ]
)
power_vals = np.array(
    [
        0,
        0,
        0,
        100,
        300,
        600,
        1200,
        1800,
        2300,
        2700,
        2900,
        2950,
        3000,
        3000,
        3000,
        3000,
        3000,
        3000,
        3000,
        3000,
        3000,
        3000,
        3000,
        3000,
        3000,
        3000,
    ]
)

cutin_ws = 3.0
cutout_ws = 25.0
ct_pw_ws = np.arange(0.0, cutout_ws + 1.0, 1.0)
ct_curve = np.stack([ct_pw_ws, ct_vals], axis=1)
power_curve = np.stack([ct_pw_ws, power_vals], axis=1)

# 2x3 grid (6 turbines)
RD = 120.0
width, length = 20, 3
x, y = np.meshgrid(
    np.linspace(0, width * RD, width),
    np.linspace(0, length * RD, length),
)
x, y = x.flatten(), y.flatten()

# PyWake setup
power_values_W = power_curve[:, 1] * 1000
wt_type_0_power_ct = PowerCtTabular(
    ws=power_curve[:, 0],
    power=power_values_W,
    power_unit="w",
    ct=ct_curve[:, 1],
)

names = [f"WT{i}" for i in range(len(x))]
windTurbines = WindTurbines(
    names=names,
    diameters=[RD] * len(x),
    hub_heights=[100.0] * len(x),
    powerCtFunctions=[wt_type_0_power_ct] * len(x),
)

site = Hornsrev1Site()

wake_model = PyWakeNiayifarGaussianDeficit(use_effective_ws=True, use_effective_ti=True)

wfm = All2AllIterative(
    site,
    windTurbines,
    wake_deficitModel=wake_model,
    superpositionModel=SquaredSum(),
    turbulenceModel=PyWakeCrespoHernandez(rotorAvgModel=None),
)

for _ in range(20):
    n_test = 1
    ws = np.random.uniform(cutin_ws, cutout_ws, size=n_test)
    wd = np.random.uniform(0, 360, size=n_test)

    # print("=" * 80)
    # print("PyWake")
    # print("=" * 80)
    sim_res = wfm(x=x, y=y, wd=wd, ws=ws, time=True, TI=0.1, WS_eff=0)
    pywake_ws_eff = sim_res["WS_eff"].values
    # print("Shape:", pywake_ws_eff.shape)
    # print("WS_eff:\n", pywake_ws_eff)

    # pixwake
    model = NiayifarGaussianDeficit(
        use_effective_ws=True,
        use_radius_mask=False,
        use_effective_ti=True,
        turbulence_model=CrespoHernandez(),
    )
    turbine = Turbine(
        rotor_diameter=RD,
        hub_height=100.0,
        power_curve=Curve(
            wind_speed=jnp.array(power_curve[:, 0]), values=jnp.array(power_curve[:, 1])
        ),
        ct_curve=Curve(
            wind_speed=jnp.array(ct_curve[:, 0]), values=jnp.array(ct_curve[:, 1])
        ),
    )
    sim = WakeSimulation(model, turbine, fpi_damp=0.5, mapping_strategy="_manual")

    # print("\n" + "=" * 80)
    # print("Pixwake")
    # print("=" * 80)
    pixwake_sim_res = sim(
        jnp.asarray(x), jnp.asarray(y), jnp.asarray(ws), jnp.asarray(wd), 0.1
    )
    ws_eff, ti_eff = pixwake_sim_res.effective_ws, pixwake_sim_res.effective_ti
    # print(ws_eff, ti_eff)

    # print("Shape:", pixwake_sim_res.effective_ws.T.shape)
    # print("WS_eff:\n", pixwake_sim_res.effective_ws.T)

    # print("\n" + "=" * 80)
    # print("Comparison")
    # print("=" * 80)
    diff = pywake_ws_eff - ws_eff.reshape(pywake_ws_eff.shape)
    # print("Difference:\n", diff)
    # print(f"Max abs diff: {np.abs(diff).max():.6f}")
    # print(f"Mean abs diff: {np.abs(diff).mean():.6f}")
    rel_diff = diff / np.maximum(pywake_ws_eff, 1e-6)

    max_rel_diff = np.abs(rel_diff).max() * 100
    if max_rel_diff > 1.0:
        # print(f"Max rel diff: {max_rel_diff:.6f} %")
        # print(ti_eff)
        # print(sim_res["TI_eff"].values.T)
        raise Exception(f"Max rel diff: {max_rel_diff:.6f} %")

    # print(f"Max rel diff: {np.abs(rel_diff).max() * 100:.6f} %")
    # print(f"Mismatched (>1% rel): {(np.abs(rel_diff) > 0.01).sum()} / {rel_diff.size}")

    # print(pixwake_sim_res.ctx.ti)
    # print(sim_res["TI_eff"].values.T)
