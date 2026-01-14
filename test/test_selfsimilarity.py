"""Tests for the self-similarity blockage deficit models.

These tests compare the pixwake implementation against the PyWake reference
implementation to ensure numerical equivalence.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import config as jcfg
from py_wake.deficit_models.selfsimilarity import (
    SelfSimilarityDeficit as PyWakeSelfSimilarityDeficit,
)
from py_wake.deficit_models.selfsimilarity import (
    SelfSimilarityDeficit2020 as PyWakeSelfSimilarityDeficit2020,
)

from pixwake import Curve, Turbine
from pixwake.core import SimulationContext
from pixwake.deficit import (
    SelfSimilarityBlockageDeficit,
    SelfSimilarityBlockageDeficit2020,
)

# Preserve np.asarray before jax config
asarray_method = np.asarray
np.random.seed(42)
jcfg.update("jax_enable_x64", True)
np.asarray = asarray_method


def _create_test_turbine(rd: float = 80.0, hh: float = 100.0) -> Turbine:
    """Create a simple test turbine with constant Ct."""
    ws_curve = jnp.array([0.0, 25.0])
    ct_vals = jnp.array([0.8, 0.8])
    power_vals = jnp.array([0.0, 3000.0])
    return Turbine(
        rotor_diameter=rd,
        hub_height=hh,
        power_curve=Curve(ws=ws_curve, values=power_vals),
        ct_curve=Curve(ws=ws_curve, values=ct_vals),
    )


class TestSelfSimilarityDeficitFormulas:
    """Test individual component functions of the self-similarity model."""

    def test_r12_original(self):
        """Test the r12 function for the original model."""
        model = SelfSimilarityBlockageDeficit()
        pywake_model = PyWakeSelfSimilarityDeficit()

        # Test at various normalized distances
        x_vals = jnp.array([-1.0, -2.0, -4.0, -6.0, -10.0])

        pixwake_r12 = model.r12(x_vals)
        pywake_r12 = pywake_model.r12(np.asarray(x_vals))

        np.testing.assert_allclose(pixwake_r12, pywake_r12, rtol=1e-10)

    def test_r12_2020(self):
        """Test the r12 function for the 2020 model (linear approximation)."""
        model = SelfSimilarityBlockageDeficit2020()
        pywake_model = PyWakeSelfSimilarityDeficit2020()

        x_vals = jnp.array([-1.0, -2.0, -4.0, -6.0, -10.0])

        pixwake_r12 = model.r12(x_vals)
        pywake_r12 = pywake_model.r12(np.asarray(x_vals))

        np.testing.assert_allclose(pixwake_r12, pywake_r12, rtol=1e-10)

    def test_ct2af(self):
        """Test the axial induction shape function."""
        model = SelfSimilarityBlockageDeficit()
        pywake_model = PyWakeSelfSimilarityDeficit()

        x_vals = jnp.array([-1.0, -2.0, -4.0, -6.0, -10.0])

        pixwake_ct2af = model.ct2af(x_vals)
        pywake_ct2af = pywake_model.ct2af(np.asarray(x_vals))

        np.testing.assert_allclose(pixwake_ct2af, pywake_ct2af, rtol=1e-10)

    def test_ct2a0_original(self):
        """Test the axial induction factor calculation."""
        model = SelfSimilarityBlockageDeficit()
        pywake_model = PyWakeSelfSimilarityDeficit()

        ct_vals = jnp.array([0.4, 0.6, 0.8])

        pixwake_ct2a0 = model.ct2a0(ct_vals)
        # PyWake uses a 4D array format, we just need 1D
        pywake_ct2a0 = pywake_model.ct2a0(
            np.array([[[[-1.0]]]]),  # x_ijlk (not used for original)
            np.asarray(ct_vals)[:, np.newaxis, np.newaxis],
        ).squeeze()

        np.testing.assert_allclose(pixwake_ct2a0, pywake_ct2a0, rtol=1e-6)

    def test_near_gamma_2020(self):
        """Test the near-field gamma function for 2020 model."""
        model = SelfSimilarityBlockageDeficit2020()
        pywake_model = PyWakeSelfSimilarityDeficit2020()

        ct_vals = jnp.array([0.4, 0.6, 0.8])

        pixwake_ng = model.near_gamma(ct_vals)
        # PyWake expects (i, l, k) format
        pywake_ng = pywake_model.near_gamma(
            np.asarray(ct_vals)[:, np.newaxis, np.newaxis]
        ).squeeze()

        np.testing.assert_allclose(pixwake_ng, pywake_ng, rtol=1e-10)

    def test_far_gamma_2020(self):
        """Test the far-field gamma function for 2020 model."""
        model = SelfSimilarityBlockageDeficit2020()
        pywake_model = PyWakeSelfSimilarityDeficit2020()

        ct_vals = jnp.array([0.4, 0.6, 0.8])

        pixwake_fg = model.far_gamma(ct_vals)
        pywake_fg = pywake_model.far_gamma(
            np.asarray(ct_vals)[:, np.newaxis, np.newaxis]
        ).squeeze()

        np.testing.assert_allclose(pixwake_fg, pywake_fg, rtol=1e-10)


class TestSelfSimilarityDeficitCenterline:
    """Test centerline (cw=0) deficit calculations."""

    @pytest.fixture
    def setup(self):
        """Set up common test parameters."""
        ws = 10.0
        D = 80.0
        R = D / 2.0
        ct = 0.8
        return {"ws": ws, "D": D, "R": R, "ct": ct}

    def test_original_model_centerline(self, setup):
        """Test original self-similarity model on centerline."""
        ws, D, R, ct = setup["ws"], setup["D"], setup["R"], setup["ct"]

        # Create pixwake model and turbine
        model = SelfSimilarityBlockageDeficit()
        turbine = _create_test_turbine(rd=D)

        # Test points upstream (negative dw in pixwake convention)
        # PyWake uses x (positive upstream), pixwake uses dw (negative upstream)
        upstream_distances = np.arange(1, 200)  # in meters
        x_pywake = -upstream_distances  # PyWake: negative x is upstream of rotor

        # Create PyWake model
        pywake_model = PyWakeSelfSimilarityDeficit()

        # PyWake format: (i, j, l, k) = (source, point, sector, ws_bin)
        WS_ilk = np.array([[[ws]]])
        D_src_il = np.array([[D]])
        ct_ilk = np.array([[[ct]]])
        dw_ijlk = x_pywake.reshape((1, len(x_pywake), 1, 1))
        cw_ijlk = np.zeros_like(dw_ijlk)

        pywake_deficit = pywake_model.calc_deficit(
            WS_ref_ilk=WS_ilk,
            D_src_il=D_src_il,
            dw_ijlk=dw_ijlk,
            cw_ijlk=cw_ijlk,
            ct_ilk=ct_ilk,
            wake_radius_ijlk=np.zeros_like(dw_ijlk),
        ).squeeze()

        # Create pixwake context
        # In pixwake: dw = downstream distance (negative = upstream)
        dw = jnp.array(-upstream_distances).reshape(-1, 1)  # (n_points, 1)
        cw = jnp.zeros_like(dw)
        ws_eff = jnp.array([ws])

        ctx = SimulationContext(
            turbine=turbine,
            dw=dw,
            cw=cw,
            ws=ws,
            ti=None,
        )

        pixwake_deficit = model._deficit(ws_eff, None, ctx).squeeze()

        # PyWake returns positive deficit for upstream blockage
        # The sign should match
        np.testing.assert_allclose(
            pixwake_deficit, pywake_deficit, rtol=1e-5, atol=1e-10
        )

    def test_2020_model_centerline(self, setup):
        """Test 2020 self-similarity model on centerline."""
        ws, D, R, ct = setup["ws"], setup["D"], setup["R"], setup["ct"]

        model = SelfSimilarityBlockageDeficit2020()
        turbine = _create_test_turbine(rd=D)

        upstream_distances = np.arange(1, 200)
        x_pywake = -upstream_distances

        pywake_model = PyWakeSelfSimilarityDeficit2020()

        WS_ilk = np.array([[[ws]]])
        D_src_il = np.array([[D]])
        ct_ilk = np.array([[[ct]]])
        dw_ijlk = x_pywake.reshape((1, len(x_pywake), 1, 1))
        cw_ijlk = np.zeros_like(dw_ijlk)

        pywake_deficit = pywake_model.calc_deficit(
            WS_ref_ilk=WS_ilk,
            D_src_il=D_src_il,
            dw_ijlk=dw_ijlk,
            cw_ijlk=cw_ijlk,
            ct_ilk=ct_ilk,
            wake_radius_ijlk=np.zeros_like(dw_ijlk),
        ).squeeze()

        dw = jnp.array(-upstream_distances).reshape(-1, 1)
        cw = jnp.zeros_like(dw)
        ws_eff = jnp.array([ws])

        ctx = SimulationContext(
            turbine=turbine,
            dw=dw,
            cw=cw,
            ws=ws,
            ti=None,
        )

        pixwake_deficit = model._deficit(ws_eff, None, ctx).squeeze()

        np.testing.assert_allclose(
            pixwake_deficit, pywake_deficit, rtol=1e-5, atol=1e-10
        )


class TestSelfSimilarityDeficitRadial:
    """Test radial profile of deficit at fixed upstream distances."""

    def test_radial_profile_original(self):
        """Test radial decay profile for original model."""
        ws, D, ct = 10.0, 80.0, 0.8
        R = D / 2.0

        model = SelfSimilarityBlockageDeficit()
        turbine = _create_test_turbine(rd=D)
        pywake_model = PyWakeSelfSimilarityDeficit()

        # Fixed upstream distance of 2R
        upstream_dist = 2 * R
        x_pywake = -upstream_dist

        # Lateral positions
        y_vals = np.arange(0, 200)

        WS_ilk = np.array([[[ws]]])
        D_src_il = np.array([[D]])
        ct_ilk = np.array([[[ct]]])
        dw_ijlk = np.full((1, len(y_vals), 1, 1), x_pywake)
        cw_ijlk = np.abs(y_vals).reshape((1, len(y_vals), 1, 1))

        pywake_deficit = pywake_model.calc_deficit(
            WS_ref_ilk=WS_ilk,
            D_src_il=D_src_il,
            dw_ijlk=dw_ijlk,
            cw_ijlk=cw_ijlk,
            ct_ilk=ct_ilk,
            wake_radius_ijlk=np.zeros_like(dw_ijlk),
        ).squeeze()

        # Pixwake
        dw = jnp.full((len(y_vals), 1), -upstream_dist)
        cw = jnp.array(np.abs(y_vals)).reshape(-1, 1)
        ws_eff = jnp.array([ws])

        ctx = SimulationContext(
            turbine=turbine,
            dw=dw,
            cw=cw,
            ws=ws,
            ti=None,
        )

        pixwake_deficit = model._deficit(ws_eff, None, ctx).squeeze()

        np.testing.assert_allclose(
            pixwake_deficit, pywake_deficit, rtol=1e-5, atol=1e-10
        )

    def test_radial_profile_2020(self):
        """Test radial decay profile for 2020 model."""
        ws, D, ct = 10.0, 80.0, 0.8
        R = D / 2.0

        model = SelfSimilarityBlockageDeficit2020()
        turbine = _create_test_turbine(rd=D)
        pywake_model = PyWakeSelfSimilarityDeficit2020()

        upstream_dist = 2 * R
        x_pywake = -upstream_dist
        y_vals = np.arange(0, 200)

        WS_ilk = np.array([[[ws]]])
        D_src_il = np.array([[D]])
        ct_ilk = np.array([[[ct]]])
        dw_ijlk = np.full((1, len(y_vals), 1, 1), x_pywake)
        cw_ijlk = np.abs(y_vals).reshape((1, len(y_vals), 1, 1))

        pywake_deficit = pywake_model.calc_deficit(
            WS_ref_ilk=WS_ilk,
            D_src_il=D_src_il,
            dw_ijlk=dw_ijlk,
            cw_ijlk=cw_ijlk,
            ct_ilk=ct_ilk,
            wake_radius_ijlk=np.zeros_like(dw_ijlk),
        ).squeeze()

        dw = jnp.full((len(y_vals), 1), -upstream_dist)
        cw = jnp.array(np.abs(y_vals)).reshape(-1, 1)
        ws_eff = jnp.array([ws])

        ctx = SimulationContext(
            turbine=turbine,
            dw=dw,
            cw=cw,
            ws=ws,
            ti=None,
        )

        pixwake_deficit = model._deficit(ws_eff, None, ctx).squeeze()

        np.testing.assert_allclose(
            pixwake_deficit, pywake_deficit, rtol=1e-5, atol=1e-10
        )


class TestSelfSimilarityDeficitVaryingCt:
    """Test models with varying thrust coefficients."""

    @pytest.mark.parametrize("ct", [0.4, 0.6, 0.8, 0.9])
    def test_original_model_varying_ct(self, ct):
        """Test original model with different Ct values."""
        ws, D = 10.0, 80.0

        # Create turbine with specified Ct
        ws_curve = jnp.array([0.0, 25.0])
        ct_vals = jnp.array([ct, ct])
        power_vals = jnp.array([0.0, 3000.0])
        turbine = Turbine(
            rotor_diameter=D,
            hub_height=100.0,
            power_curve=Curve(ws=ws_curve, values=power_vals),
            ct_curve=Curve(ws=ws_curve, values=ct_vals),
        )

        model = SelfSimilarityBlockageDeficit()
        pywake_model = PyWakeSelfSimilarityDeficit()

        upstream_distances = np.array([20, 40, 80, 160])
        x_pywake = -upstream_distances

        WS_ilk = np.array([[[ws]]])
        D_src_il = np.array([[D]])
        ct_ilk = np.array([[[ct]]])
        dw_ijlk = x_pywake.reshape((1, len(x_pywake), 1, 1))
        cw_ijlk = np.zeros_like(dw_ijlk)

        pywake_deficit = pywake_model.calc_deficit(
            WS_ref_ilk=WS_ilk,
            D_src_il=D_src_il,
            dw_ijlk=dw_ijlk,
            cw_ijlk=cw_ijlk,
            ct_ilk=ct_ilk,
            wake_radius_ijlk=np.zeros_like(dw_ijlk),
        ).squeeze()

        dw = jnp.array(-upstream_distances).reshape(-1, 1)
        cw = jnp.zeros_like(dw)
        ws_eff = jnp.array([ws])

        ctx = SimulationContext(
            turbine=turbine,
            dw=dw,
            cw=cw,
            ws=ws,
            ti=None,
        )

        pixwake_deficit = model._deficit(ws_eff, None, ctx).squeeze()

        np.testing.assert_allclose(
            pixwake_deficit, pywake_deficit, rtol=1e-5, atol=1e-10
        )

    @pytest.mark.parametrize("ct", [0.4, 0.6, 0.8, 0.9])
    def test_2020_model_varying_ct(self, ct):
        """Test 2020 model with different Ct values."""
        ws, D = 10.0, 80.0

        ws_curve = jnp.array([0.0, 25.0])
        ct_vals = jnp.array([ct, ct])
        power_vals = jnp.array([0.0, 3000.0])
        turbine = Turbine(
            rotor_diameter=D,
            hub_height=100.0,
            power_curve=Curve(ws=ws_curve, values=power_vals),
            ct_curve=Curve(ws=ws_curve, values=ct_vals),
        )

        model = SelfSimilarityBlockageDeficit2020()
        pywake_model = PyWakeSelfSimilarityDeficit2020()

        upstream_distances = np.array([20, 40, 80, 160])
        x_pywake = -upstream_distances

        WS_ilk = np.array([[[ws]]])
        D_src_il = np.array([[D]])
        ct_ilk = np.array([[[ct]]])
        dw_ijlk = x_pywake.reshape((1, len(x_pywake), 1, 1))
        cw_ijlk = np.zeros_like(dw_ijlk)

        pywake_deficit = pywake_model.calc_deficit(
            WS_ref_ilk=WS_ilk,
            D_src_il=D_src_il,
            dw_ijlk=dw_ijlk,
            cw_ijlk=cw_ijlk,
            ct_ilk=ct_ilk,
            wake_radius_ijlk=np.zeros_like(dw_ijlk),
        ).squeeze()

        dw = jnp.array(-upstream_distances).reshape(-1, 1)
        cw = jnp.zeros_like(dw)
        ws_eff = jnp.array([ws])

        ctx = SimulationContext(
            turbine=turbine,
            dw=dw,
            cw=cw,
            ws=ws,
            ti=None,
        )

        pixwake_deficit = model._deficit(ws_eff, None, ctx).squeeze()

        np.testing.assert_allclose(
            pixwake_deficit, pywake_deficit, rtol=1e-5, atol=1e-10
        )


class TestSelfSimilarityBlockageGradients:
    """Test that gradients work correctly for the blockage models."""

    def test_gradients_original(self):
        """Test that gradients can be computed for original model."""
        model = SelfSimilarityBlockageDeficit()
        turbine = _create_test_turbine(rd=80.0)

        def compute_deficit_sum(dw_vals):
            dw = dw_vals.reshape(-1, 1)
            cw = jnp.zeros_like(dw)
            ctx = SimulationContext(
                turbine=turbine,
                dw=dw,
                cw=cw,
                ws=10.0,
                ti=None,
            )
            ws_eff = jnp.array([10.0])
            deficit = model._deficit(ws_eff, None, ctx)
            return jnp.sum(deficit)

        dw_vals = jnp.array([-40.0, -80.0, -120.0])
        grad_fn = jax.grad(compute_deficit_sum)
        grads = grad_fn(dw_vals)

        assert jnp.all(jnp.isfinite(grads))
        # Deficit should decrease as we move further upstream (more negative dw)
        # So gradient should be positive (deficit increases as dw increases toward 0)
        assert jnp.all(grads > 0)

    def test_gradients_2020(self):
        """Test that gradients can be computed for 2020 model."""
        model = SelfSimilarityBlockageDeficit2020()
        turbine = _create_test_turbine(rd=80.0)

        def compute_deficit_sum(dw_vals):
            dw = dw_vals.reshape(-1, 1)
            cw = jnp.zeros_like(dw)
            ctx = SimulationContext(
                turbine=turbine,
                dw=dw,
                cw=cw,
                ws=10.0,
                ti=None,
            )
            ws_eff = jnp.array([10.0])
            deficit = model._deficit(ws_eff, None, ctx)
            return jnp.sum(deficit)

        dw_vals = jnp.array([-40.0, -80.0, -120.0])
        grad_fn = jax.grad(compute_deficit_sum)
        grads = grad_fn(dw_vals)

        assert jnp.all(jnp.isfinite(grads))


class TestSelfSimilarityBlockageIntegration:
    """Integration tests for the blockage models with full call chain."""

    def test_blockage_and_speedup_effects(self):
        """Verify blockage (upstream) and speedup (downstream) effects."""
        model = SelfSimilarityBlockageDeficit()
        turbine = _create_test_turbine(rd=80.0)

        # Create context with points both upstream and downstream
        dw = jnp.array([[-80.0], [80.0]])  # upstream, downstream
        cw = jnp.zeros_like(dw)
        ws_eff = jnp.array([10.0])

        ctx = SimulationContext(
            turbine=turbine,
            dw=dw,
            cw=cw,
            ws=10.0,
            ti=None,
        )

        new_ws, _ = model(ws_eff, None, ctx)

        # Upstream point (dw < 0) should have reduced wind speed (blockage effect)
        assert new_ws[0] < 10.0, (
            "Upstream point should experience blockage (reduced wind speed)"
        )
        # Downstream point (dw > 0) should have increased wind speed (speedup effect)
        assert new_ws[1] > 10.0, (
            "Downstream point should experience speedup (increased wind speed)"
        )

    def test_blockage_magnitude_reasonable(self):
        """Verify blockage magnitude is physically reasonable."""
        model = SelfSimilarityBlockageDeficit()
        turbine = _create_test_turbine(rd=80.0)

        # Point 1D upstream on centerline
        dw = jnp.array([[-80.0]])
        cw = jnp.zeros_like(dw)
        ws_eff = jnp.array([10.0])

        ctx = SimulationContext(
            turbine=turbine,
            dw=dw,
            cw=cw,
            ws=10.0,
            ti=None,
        )

        new_ws, _ = model(ws_eff, None, ctx)

        # Blockage at 1D upstream should be a few percent
        blockage_fraction = (10.0 - new_ws[0]) / 10.0
        assert 0.01 < blockage_fraction < 0.2  # 1% to 20%

    def test_model_jit_compatible(self):
        """Test that model can be JIT compiled."""
        model = SelfSimilarityBlockageDeficit()
        turbine = _create_test_turbine(rd=80.0)

        @jax.jit
        def run_model(dw_vals):
            dw = dw_vals.reshape(-1, 1)
            cw = jnp.zeros_like(dw)
            ctx = SimulationContext(
                turbine=turbine,
                dw=dw,
                cw=cw,
                ws=10.0,
                ti=None,
            )
            ws_eff = jnp.array([10.0])
            return model(ws_eff, None, ctx)[0]

        dw_vals = jnp.array([-40.0, -80.0, -120.0])
        result = run_model(dw_vals)

        assert result.shape == (3,)
        assert jnp.all(jnp.isfinite(result))
        assert jnp.all(result < 10.0)  # All should have some blockage


# =============================================================================
# Full Integration Tests - Compare complete simulations with PyWake
# =============================================================================

from py_wake.deficit_models.gaussian import (
    BastankhahGaussianDeficit as PyWakeBastankhahGaussianDeficit,
)
from py_wake.deficit_models.no_wake import NoWakeDeficit
from py_wake.deficit_models.noj import NOJDeficit as PyWakeNOJDeficit
from py_wake.examples.data.hornsrev1 import Hornsrev1Site
from py_wake.superposition_models import LinearSum, SquaredSum
from py_wake.wind_farm_models import All2AllIterative
from py_wake.wind_turbines import WindTurbines
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular

from pixwake import WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit, NOJDeficit


def _create_turbine_layout(width: int, length: int, spacing: float = 500.0):
    """Create a grid layout of turbines."""
    x, y = np.meshgrid(
        np.linspace(0, width * spacing, width),
        np.linspace(0, length * spacing, length),
    )
    return x.flatten(), y.flatten()


def _create_pywake_turbines(n_turbines: int, ct_curve, power_curve, RD=80.0, HH=100.0):
    """Create PyWake WindTurbines object."""
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


def _create_pixwake_turbine(ct_curve, power_curve, RD=80.0, HH=100.0):
    """Create pixwake Turbine object."""
    return Turbine(
        rotor_diameter=RD,
        hub_height=HH,
        power_curve=Curve(
            ws=jnp.array(power_curve[:, 0]), values=jnp.array(power_curve[:, 1])
        ),
        ct_curve=Curve(ws=jnp.array(ct_curve[:, 0]), values=jnp.array(ct_curve[:, 1])),
    )


@pytest.fixture
def turbine_curves():
    """Create ct and power curves for testing."""
    ws = np.arange(0.0, 26.0, 1.0)
    # Constant Ct for simplicity
    ct_vals = np.where(ws < 4, 0.0, np.where(ws > 25, 0.0, 0.8))
    power_vals = np.where(
        ws < 4, 0.0, np.where(ws > 25, 0.0, np.minimum(3000, 50 * ws**2))
    )
    ct_curve = np.stack([ws, ct_vals], axis=1)
    power_curve = np.stack([ws, power_vals], axis=1)
    return ct_curve, power_curve


class TestSelfSimilarityFullIntegration:
    """Full integration tests comparing pixwake and PyWake simulations."""

    @pytest.mark.parametrize(
        "pixwake_model_cls,pywake_model_cls",
        [
            (SelfSimilarityBlockageDeficit, PyWakeSelfSimilarityDeficit),
            (SelfSimilarityBlockageDeficit2020, PyWakeSelfSimilarityDeficit2020),
        ],
        ids=["original", "2020"],
    )
    def test_blockage_equivalence_row_of_turbines(
        self, turbine_curves, pixwake_model_cls, pywake_model_cls
    ):
        """Test blockage effect on a row of turbines aligned with wind direction.

        With wind from 270 degrees (west), turbines in a row experience blockage
        from downstream turbines.
        """
        ct_curve, power_curve = turbine_curves
        RD, HH = 80.0, 100.0

        # Row of 5 turbines spaced 5D apart, aligned with wind
        n_turbines = 5
        spacing = 5 * RD
        x = np.arange(n_turbines) * spacing
        y = np.zeros(n_turbines)

        windTurbines = _create_pywake_turbines(
            len(x), ct_curve, power_curve, RD=RD, HH=HH
        )

        site = Hornsrev1Site()

        # PyWake: Use NoWakeDeficit with blockage model
        pywake_blockage = pywake_model_cls()
        wfm = All2AllIterative(
            site,
            windTurbines,
            wake_deficitModel=NoWakeDeficit(),
            superpositionModel=LinearSum(),
            blockage_deficitModel=pywake_blockage,
        )

        # Test with single wind direction (270 = from west)
        ws = np.array([10.0])
        wd = np.array([270.0])

        sim_res = wfm(x=x, y=y, wd=wd, ws=ws)
        pywake_ws_eff = sim_res["WS_eff"].values.squeeze()

        # Pixwake
        pixwake_model = pixwake_model_cls()
        turbine = _create_pixwake_turbine(ct_curve, power_curve, RD=RD, HH=HH)
        sim = WakeSimulation(turbine, pixwake_model, fpi_damp=1.0)

        pixwake_sim_res = sim(
            jnp.asarray(x), jnp.asarray(y), jnp.asarray(ws), jnp.asarray(wd)
        )
        pixwake_ws_eff = pixwake_sim_res.effective_ws.squeeze()

        # Compare effective wind speeds
        rtol = 1e-3
        np.testing.assert_allclose(
            pixwake_ws_eff,
            pywake_ws_eff,
            rtol=rtol,
            atol=1e-6,
            err_msg=f"Wind speed mismatch for {pixwake_model_cls.__name__}",
        )

    @pytest.mark.parametrize(
        "pixwake_model_cls,pywake_model_cls",
        [
            (SelfSimilarityBlockageDeficit, PyWakeSelfSimilarityDeficit),
            (SelfSimilarityBlockageDeficit2020, PyWakeSelfSimilarityDeficit2020),
        ],
        ids=["original", "2020"],
    )
    def test_blockage_equivalence_grid_layout(
        self, turbine_curves, pixwake_model_cls, pywake_model_cls
    ):
        """Test blockage on a grid layout with multiple wind directions."""
        ct_curve, power_curve = turbine_curves
        RD, HH = 80.0, 100.0

        # 3x3 grid of turbines
        x, y = _create_turbine_layout(3, 3, spacing=5 * RD)

        windTurbines = _create_pywake_turbines(
            len(x), ct_curve, power_curve, RD=RD, HH=HH
        )

        site = Hornsrev1Site()

        pywake_blockage = pywake_model_cls()
        wfm = All2AllIterative(
            site,
            windTurbines,
            wake_deficitModel=NoWakeDeficit(),
            superpositionModel=LinearSum(),
            blockage_deficitModel=pywake_blockage,
        )

        # Test with multiple wind directions
        ws = np.array([8.0, 10.0, 12.0])
        wd = np.array([0.0, 90.0, 270.0])

        sim_res = wfm(x=x, y=y, wd=wd, ws=ws, time=True)
        pywake_ws_eff = sim_res["WS_eff"].values

        # Pixwake
        pixwake_model = pixwake_model_cls()
        turbine = _create_pixwake_turbine(ct_curve, power_curve, RD=RD, HH=HH)
        sim = WakeSimulation(turbine, pixwake_model, fpi_damp=1.0)

        pixwake_sim_res = sim(
            jnp.asarray(x), jnp.asarray(y), jnp.asarray(ws), jnp.asarray(wd)
        )

        rtol = 1e-3
        np.testing.assert_allclose(
            pixwake_sim_res.effective_ws.T,
            pywake_ws_eff,
            rtol=rtol,
            atol=1e-6,
            err_msg=f"Wind speed mismatch for {pixwake_model_cls.__name__}",
        )

    @pytest.mark.parametrize(
        "pixwake_model_cls,pywake_model_cls",
        [
            (SelfSimilarityBlockageDeficit, PyWakeSelfSimilarityDeficit),
            (SelfSimilarityBlockageDeficit2020, PyWakeSelfSimilarityDeficit2020),
        ],
        ids=["original", "2020"],
    )
    def test_blockage_equivalence_timeseries(
        self, turbine_curves, pixwake_model_cls, pywake_model_cls
    ):
        """Test blockage equivalence over a time series of wind conditions."""
        ct_curve, power_curve = turbine_curves
        RD, HH = 80.0, 100.0

        # 4x4 grid
        x, y = _create_turbine_layout(4, 4, spacing=5 * RD)

        windTurbines = _create_pywake_turbines(
            len(x), ct_curve, power_curve, RD=RD, HH=HH
        )

        site = Hornsrev1Site()

        pywake_blockage = pywake_model_cls()
        wfm = All2AllIterative(
            site,
            windTurbines,
            wake_deficitModel=NoWakeDeficit(),
            superpositionModel=LinearSum(),
            blockage_deficitModel=pywake_blockage,
        )

        # Random time series
        n_timestamps = 50
        np.random.seed(123)
        ws = np.random.uniform(6.0, 14.0, size=n_timestamps)
        wd = np.random.uniform(0, 360, size=n_timestamps)

        sim_res = wfm(x=x, y=y, wd=wd, ws=ws, time=True)
        pywake_ws_eff = sim_res["WS_eff"].values
        pywake_aep = sim_res.aep().sum().values

        # Pixwake
        pixwake_model = pixwake_model_cls()
        turbine = _create_pixwake_turbine(ct_curve, power_curve, RD=RD, HH=HH)
        sim = WakeSimulation(turbine, pixwake_model, fpi_damp=1.0)

        pixwake_sim_res = sim(
            jnp.asarray(x), jnp.asarray(y), jnp.asarray(ws), jnp.asarray(wd)
        )
        pixwake_aep = pixwake_sim_res.aep()

        rtol = 1e-3
        np.testing.assert_allclose(
            pixwake_sim_res.effective_ws.T,
            pywake_ws_eff,
            rtol=rtol,
            atol=1e-6,
            err_msg=f"Wind speed mismatch for {pixwake_model_cls.__name__}",
        )
        np.testing.assert_allclose(
            pixwake_aep,
            pywake_aep,
            rtol=rtol,
            err_msg=f"AEP mismatch for {pixwake_model_cls.__name__}",
        )

    @pytest.mark.parametrize(
        "pixwake_model_cls,pywake_model_cls",
        [
            (SelfSimilarityBlockageDeficit, PyWakeSelfSimilarityDeficit),
            (SelfSimilarityBlockageDeficit2020, PyWakeSelfSimilarityDeficit2020),
        ],
        ids=["original", "2020"],
    )
    def test_blockage_gradients(
        self, turbine_curves, pixwake_model_cls, pywake_model_cls
    ):
        """Test that AEP gradients match between pixwake and PyWake."""
        ct_curve, power_curve = turbine_curves
        RD, HH = 80.0, 100.0

        # Small layout for faster gradient computation
        x, y = _create_turbine_layout(3, 3, spacing=5 * RD)

        windTurbines = _create_pywake_turbines(
            len(x), ct_curve, power_curve, RD=RD, HH=HH
        )

        site = Hornsrev1Site()

        pywake_blockage = pywake_model_cls()
        wfm = All2AllIterative(
            site,
            windTurbines,
            wake_deficitModel=NoWakeDeficit(),
            superpositionModel=LinearSum(),
            blockage_deficitModel=pywake_blockage,
        )

        # Small time series for gradient test
        n_timestamps = 20
        np.random.seed(456)
        ws = np.random.uniform(8.0, 12.0, size=n_timestamps)
        wd = np.random.uniform(0, 360, size=n_timestamps)

        # PyWake gradients
        pw_dx, pw_dy = wfm.aep_gradients(x=x, y=y, wd=wd, ws=ws, time=True)

        # Pixwake gradients
        pixwake_model = pixwake_model_cls()
        turbine = _create_pixwake_turbine(ct_curve, power_curve, RD=RD, HH=HH)
        sim = WakeSimulation(turbine, pixwake_model, fpi_damp=1.0)

        grad_fn = jax.jit(
            jax.value_and_grad(
                lambda xx, yy: sim(xx, yy, jnp.array(ws), jnp.array(wd)).aep(),
                argnums=(0, 1),
            )
        )

        val, (px_dx, px_dy) = grad_fn(jnp.asarray(x), jnp.asarray(y))
        px_dx.block_until_ready()
        px_dy.block_until_ready()

        # Check gradients are finite
        assert np.isfinite(px_dx).all(), (
            f"Non-finite gradients for {pixwake_model_cls.__name__}"
        )
        assert np.isfinite(px_dy).all(), (
            f"Non-finite gradients for {pixwake_model_cls.__name__}"
        )

        # Compare gradients
        rtol = 1e-2  # Slightly looser tolerance for gradients
        np.testing.assert_allclose(
            px_dx,
            pw_dx,
            rtol=rtol,
            atol=1e-6,
            err_msg=f"dx gradient mismatch for {pixwake_model_cls.__name__}",
        )
        np.testing.assert_allclose(
            px_dy,
            pw_dy,
            rtol=rtol,
            atol=1e-6,
            err_msg=f"dy gradient mismatch for {pixwake_model_cls.__name__}",
        )


class TestCombinedWakeAndBlockage:
    """Tests for combined wake deficit + blockage models.

    These tests verify that pixwake correctly combines wake and blockage effects
    to match PyWake's behavior when using both wake_deficitModel and
    blockage_deficitModel together.

    Uses close turbine spacing (2-3D) to ensure blockage effects are significant
    and visible in the results.
    """

    @pytest.fixture
    def close_spacing_curves(self):
        """Create ct and power curves for close-spacing tests."""
        ws = np.arange(0.0, 26.0, 1.0)
        # Higher Ct for stronger blockage effects
        ct_vals = np.where(ws < 4, 0.0, np.where(ws > 25, 0.0, 0.85))
        power_vals = np.where(
            ws < 4, 0.0, np.where(ws > 25, 0.0, np.minimum(3000, 50 * ws**2))
        )
        ct_curve = np.stack([ws, ct_vals], axis=1)
        power_curve = np.stack([ws, power_vals], axis=1)
        return ct_curve, power_curve

    @pytest.mark.parametrize(
        "pixwake_blockage_cls,pywake_blockage_cls",
        [
            (SelfSimilarityBlockageDeficit, PyWakeSelfSimilarityDeficit),
            (SelfSimilarityBlockageDeficit2020, PyWakeSelfSimilarityDeficit2020),
        ],
        ids=["original", "2020"],
    )
    def test_gaussian_with_blockage_close_spacing(
        self, close_spacing_curves, pixwake_blockage_cls, pywake_blockage_cls
    ):
        """Test Gaussian wake model combined with blockage at close spacing (3D).

        Uses Bastankhah Gaussian model with blockage.
        """
        ct_curve, power_curve = close_spacing_curves
        RD, HH = 80.0, 100.0
        k = 0.04  # Wake expansion coefficient

        # Close spacing: 3D between turbines
        spacing = 3.0 * RD
        x, y = _create_turbine_layout(4, 4, spacing=spacing)

        windTurbines = _create_pywake_turbines(
            len(x), ct_curve, power_curve, RD=RD, HH=HH
        )

        site = Hornsrev1Site()

        # PyWake: Gaussian wake + self-similarity blockage
        pywake_wake = PyWakeBastankhahGaussianDeficit(k=k)
        # Blockage model needs LinearSum since it produces negative values (speedups)
        pywake_blockage = pywake_blockage_cls(superpositionModel=LinearSum())
        wfm = All2AllIterative(
            site,
            windTurbines,
            wake_deficitModel=pywake_wake,
            superpositionModel=SquaredSum(),
            blockage_deficitModel=pywake_blockage,
        )

        n_timestamps = 30
        np.random.seed(101)
        ws = np.random.uniform(8.0, 12.0, size=n_timestamps)
        wd = np.random.uniform(0, 360, size=n_timestamps)

        sim_res = wfm(x=x, y=y, wd=wd, ws=ws, time=True)
        pywake_ws_eff = sim_res["WS_eff"].values
        pywake_aep = sim_res.aep().sum().values

        # Pixwake: Gaussian wake + self-similarity blockage
        # Note: use_radius_mask=False to match PyWake behavior
        pixwake_wake = BastankhahGaussianDeficit(k=k, use_radius_mask=False)
        pixwake_blockage = pixwake_blockage_cls()
        turbine = _create_pixwake_turbine(ct_curve, power_curve, RD=RD, HH=HH)
        sim = WakeSimulation(
            turbine, pixwake_wake, blockage=pixwake_blockage, fpi_damp=1.0
        )

        pixwake_sim_res = sim(
            jnp.asarray(x), jnp.asarray(y), jnp.asarray(ws), jnp.asarray(wd)
        )
        pixwake_aep = pixwake_sim_res.aep()

        # Verify blockage effects are present
        sim_wake_only = WakeSimulation(turbine, pixwake_wake, fpi_damp=1.0)
        wake_only_res = sim_wake_only(
            jnp.asarray(x), jnp.asarray(y), jnp.asarray(ws), jnp.asarray(wd)
        )

        ws_diff = jnp.abs(pixwake_sim_res.effective_ws - wake_only_res.effective_ws)
        max_blockage_effect = jnp.max(ws_diff)
        assert max_blockage_effect > 0.01, (
            f"Blockage effect too small ({max_blockage_effect:.6f}). "
            "Ensure turbine spacing is close enough for blockage to be significant."
        )

        rtol = 1e-3
        atol = 1e-6
        np.testing.assert_allclose(
            pixwake_sim_res.effective_ws.T,
            pywake_ws_eff,
            rtol=rtol,
            atol=atol,
            err_msg=f"Wind speed mismatch for Gaussian + {pixwake_blockage_cls.__name__}",
        )
        np.testing.assert_allclose(
            pixwake_aep,
            pywake_aep,
            rtol=rtol,
            atol=atol,
            err_msg=f"AEP mismatch for Gaussian + {pixwake_blockage_cls.__name__}",
        )

    @pytest.mark.parametrize(
        "pixwake_blockage_cls,pywake_blockage_cls",
        [
            (SelfSimilarityBlockageDeficit, PyWakeSelfSimilarityDeficit),
            (SelfSimilarityBlockageDeficit2020, PyWakeSelfSimilarityDeficit2020),
        ],
        ids=["original", "2020"],
    )
    def test_combined_gradients(
        self, close_spacing_curves, pixwake_blockage_cls, pywake_blockage_cls
    ):
        """Test that AEP gradients match for combined Gaussian wake+blockage models."""
        ct_curve, power_curve = close_spacing_curves
        RD, HH = 80.0, 100.0
        k = 0.04  # Wake expansion coefficient for Gaussian

        # Small layout for faster gradient computation
        spacing = 3.0 * RD
        x, y = _create_turbine_layout(3, 3, spacing=spacing)

        windTurbines = _create_pywake_turbines(
            len(x), ct_curve, power_curve, RD=RD, HH=HH
        )

        site = Hornsrev1Site()

        # Use Gaussian wake model instead of NOJ
        pywake_wake = PyWakeBastankhahGaussianDeficit(k=k)
        pywake_blockage = pywake_blockage_cls(superpositionModel=LinearSum())
        wfm = All2AllIterative(
            site,
            windTurbines,
            wake_deficitModel=pywake_wake,
            superpositionModel=SquaredSum(),
            blockage_deficitModel=pywake_blockage,
        )

        n_timestamps = 15
        np.random.seed(202)
        ws = np.random.uniform(8.0, 12.0, size=n_timestamps)
        wd = np.random.uniform(0, 360, size=n_timestamps)

        # PyWake gradients
        pw_dx, pw_dy = wfm.aep_gradients(x=x, y=y, wd=wd, ws=ws, time=True)

        # Pixwake gradients - use Gaussian wake model
        pixwake_wake = BastankhahGaussianDeficit(k=k, use_radius_mask=False)
        pixwake_blockage = pixwake_blockage_cls()
        turbine = _create_pixwake_turbine(ct_curve, power_curve, RD=RD, HH=HH)
        sim = WakeSimulation(
            turbine, pixwake_wake, blockage=pixwake_blockage, fpi_damp=1.0
        )

        grad_fn = jax.jit(
            jax.value_and_grad(
                lambda xx, yy: sim(xx, yy, jnp.array(ws), jnp.array(wd)).aep(),
                argnums=(0, 1),
            )
        )

        val, (px_dx, px_dy) = grad_fn(jnp.asarray(x), jnp.asarray(y))
        px_dx.block_until_ready()
        px_dy.block_until_ready()

        # Check gradients are finite
        assert np.isfinite(px_dx).all(), (
            f"Non-finite dx gradients for Gaussian + {pixwake_blockage_cls.__name__}"
        )
        assert np.isfinite(px_dy).all(), (
            f"Non-finite dy gradients for Gaussian + {pixwake_blockage_cls.__name__}"
        )

        # Compare gradients
        rtol = 1e-3
        atol = 1e-6
        np.testing.assert_allclose(
            px_dx,
            pw_dx,
            rtol=rtol,
            atol=atol,
            err_msg=f"dx gradient mismatch for Gaussian + {pixwake_blockage_cls.__name__}",
        )
        np.testing.assert_allclose(
            px_dy,
            pw_dy,
            rtol=rtol,
            atol=atol,
            err_msg=f"dy gradient mismatch for Gaussian + {pixwake_blockage_cls.__name__}",
        )

    def test_blockage_effect_magnitude(self, close_spacing_curves):
        """Test that blockage effects have realistic magnitudes at close spacing.

        At 2D spacing with Ct=0.85, blockage should reduce upwind turbine
        effective wind speeds by approximately 2-4%.

        This tests blockage-only mode (blockage as primary deficit model).
        """
        ct_curve, power_curve = close_spacing_curves
        RD, HH = 80.0, 100.0

        # Very close spacing: 2D
        spacing = 2.0 * RD

        # Row of 5 turbines aligned with wind
        n_turbines = 5
        x = np.arange(n_turbines) * spacing
        y = np.zeros(n_turbines)

        turbine = _create_pixwake_turbine(ct_curve, power_curve, RD=RD, HH=HH)

        # Blockage-only mode: use blockage model as the primary deficit model
        # This properly tests blockage magnitude including upstream reduction and
        # downstream speedup
        blockage = SelfSimilarityBlockageDeficit()
        sim_blockage = WakeSimulation(turbine, blockage, fpi_damp=1.0)

        ws = np.array([10.0])
        wd = np.array([270.0])  # Wind from west

        result = sim_blockage(
            jnp.asarray(x), jnp.asarray(y), jnp.asarray(ws), jnp.asarray(wd)
        )

        ws_eff = result.effective_ws.squeeze()

        # The most upwind turbine (x=0) should experience blockage from all downstream turbines
        # Expected blockage reduction: ~2-4% at 2D spacing with Ct=0.85
        upwind_reduction = (10.0 - ws_eff[0]) / 10.0 * 100  # percentage

        # The most downwind turbine experiences speedup (in blockage-only mode)
        downwind_speedup = (ws_eff[-1] - 10.0) / 10.0 * 100  # percentage

        # Verify blockage is in expected range
        assert 1.0 < upwind_reduction < 6.0, (
            f"Upwind blockage reduction {upwind_reduction:.2f}% outside expected 1-6% range"
        )
        assert 0.5 < downwind_speedup < 4.0, (
            f"Downwind speedup {downwind_speedup:.2f}% outside expected 0.5-4% range"
        )
