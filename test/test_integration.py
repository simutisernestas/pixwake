"""Integration tests for pixwake combining multiple components."""

import jax.numpy as jnp
import numpy as np
import pytest

from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import (
    BastankhahGaussianDeficit,
    NOJDeficit,
    SelfSimilarityBlockageDeficit,
)
from pixwake.turbulence import CrespoHernandez


@pytest.fixture
def simple_turbine():
    """Create a simple test turbine."""
    ws = jnp.array([3.0, 5.0, 10.0, 15.0, 25.0])
    power = jnp.array([0.0, 100.0, 1000.0, 1500.0, 1500.0])
    ct = jnp.array([0.0, 0.8, 0.8, 0.4, 0.2])
    return Turbine(
        rotor_diameter=80.0,
        hub_height=70.0,
        power_curve=Curve(ws=ws, values=power),
        ct_curve=Curve(ws=ws, values=ct),
    )


class TestCombinedModels:
    """Tests for combined wake + blockage + turbulence."""

    def test_wake_plus_turbulence(self, simple_turbine):
        """Test wake deficit combined with turbulence model."""
        sim = WakeSimulation(
            simple_turbine,
            deficit=BastankhahGaussianDeficit(),
            turbulence=CrespoHernandez(),
        )
        wt_x = jnp.array([0.0, 400.0])
        wt_y = jnp.array([0.0, 0.0])

        result = sim(wt_x, wt_y, ws_amb=10.0, wd_amb=270.0, ti_amb=0.06)

        # Should have effective TI computed
        assert result.effective_ti is not None
        # Downstream turbine should have higher TI
        assert result.effective_ti[0, 1] > result.effective_ti[0, 0]

    def test_wake_plus_blockage(self, simple_turbine):
        """Test wake deficit combined with blockage model."""
        sim = WakeSimulation(
            simple_turbine,
            deficit=BastankhahGaussianDeficit(),
            blockage=SelfSimilarityBlockageDeficit(),
        )
        wt_x = jnp.array([0.0, 400.0])
        wt_y = jnp.array([0.0, 0.0])

        result = sim(wt_x, wt_y, ws_amb=10.0, wd_amb=270.0, ti_amb=0.06)

        # Both turbines should have reduced effective wind speed
        assert jnp.all(result.effective_ws < 10.0)

    def test_all_three_combined(self, simple_turbine):
        """Test wake + blockage + turbulence all together."""
        sim = WakeSimulation(
            simple_turbine,
            deficit=BastankhahGaussianDeficit(),
            turbulence=CrespoHernandez(),
            blockage=SelfSimilarityBlockageDeficit(),
        )
        wt_x = jnp.array([0.0, 400.0, 800.0])
        wt_y = jnp.array([0.0, 0.0, 0.0])

        result = sim(wt_x, wt_y, ws_amb=10.0, wd_amb=270.0, ti_amb=0.06)

        # Check all outputs are valid
        assert jnp.all(jnp.isfinite(result.effective_ws))
        assert jnp.all(jnp.isfinite(result.effective_ti))
        # Power should be computable
        power = result.power()
        assert jnp.all(jnp.isfinite(power))


class TestLargeFarm:
    """Tests for large wind farms."""

    def test_10x10_farm(self, simple_turbine):
        """Test a 100 turbine farm in a 10x10 grid."""
        # Create 10x10 grid
        x_pos, y_pos = jnp.meshgrid(
            jnp.arange(10) * 500.0,
            jnp.arange(10) * 500.0,
        )
        wt_x = x_pos.ravel()
        wt_y = y_pos.ravel()

        sim = WakeSimulation(simple_turbine, deficit=NOJDeficit())
        result = sim(wt_x, wt_y, ws_amb=10.0, wd_amb=270.0)

        assert result.effective_ws.shape == (1, 100)
        assert jnp.all(jnp.isfinite(result.effective_ws))

    def test_multiple_wind_conditions(self, simple_turbine):
        """Test with many wind conditions."""
        wt_x = jnp.array([0.0, 400.0, 800.0])
        wt_y = jnp.array([0.0, 0.0, 0.0])

        # 36 wind directions, 5 wind speeds = 180 cases
        wd = jnp.tile(jnp.arange(0, 360, 10), 5)
        ws = jnp.repeat(jnp.array([6.0, 8.0, 10.0, 12.0, 14.0]), 36)

        sim = WakeSimulation(simple_turbine, deficit=BastankhahGaussianDeficit())
        result = sim(wt_x, wt_y, ws_amb=ws, wd_amb=wd, ti_amb=0.06)

        assert result.effective_ws.shape == (180, 3)
        assert jnp.all(jnp.isfinite(result.effective_ws))


class TestMultiTurbineTypes:
    """Tests for farms with multiple turbine types."""

    def test_two_turbine_types(self):
        """Test a farm with two different turbine types."""
        ws = jnp.array([3.0, 10.0, 25.0])

        # Small turbine
        small = Turbine(
            rotor_diameter=60.0,
            hub_height=50.0,
            power_curve=Curve(ws=ws, values=jnp.array([0.0, 500.0, 500.0])),
            ct_curve=Curve(ws=ws, values=jnp.array([0.0, 0.8, 0.2])),
        )

        # Large turbine
        large = Turbine(
            rotor_diameter=100.0,
            hub_height=80.0,
            power_curve=Curve(ws=ws, values=jnp.array([0.0, 2000.0, 2000.0])),
            ct_curve=Curve(ws=ws, values=jnp.array([0.0, 0.8, 0.2])),
        )

        # Create simulation with library
        sim = WakeSimulation([small, large], deficit=NOJDeficit())

        # 4 turbines: small, large, small, large
        wt_x = jnp.array([0.0, 500.0, 1000.0, 1500.0])
        wt_y = jnp.array([0.0, 0.0, 0.0, 0.0])
        wt_types = [small.type_id, large.type_id, small.type_id, large.type_id]

        result = sim(wt_x, wt_y, ws_amb=10.0, wd_amb=270.0, wt_types=wt_types)

        assert result.effective_ws.shape == (1, 4)
        power = result.power()
        # Large turbines should produce more power
        assert power[0, 1] > power[0, 0]  # large > small (upstream)


class TestJSONSerialization:
    """Tests for JSON serialization of SimulationResult."""

    def test_roundtrip(self, simple_turbine):
        """Test that JSON serialization preserves data."""
        sim = WakeSimulation(simple_turbine, deficit=NOJDeficit())
        wt_x = jnp.array([0.0, 400.0])
        wt_y = jnp.array([0.0, 0.0])

        result = sim(wt_x, wt_y, ws_amb=10.0, wd_amb=270.0, ti_amb=0.06)

        # Serialize and deserialize
        json_str = result.to_json()
        restored = result.from_json(json_str)

        # Check all fields match
        np.testing.assert_allclose(restored.wt_x, result.wt_x)
        np.testing.assert_allclose(restored.wt_y, result.wt_y)
        np.testing.assert_allclose(restored.wd, result.wd)
        np.testing.assert_allclose(restored.ws, result.ws)
        np.testing.assert_allclose(restored.effective_ws, result.effective_ws)
        np.testing.assert_allclose(restored.ti, result.ti)

    def test_json_is_valid(self, simple_turbine):
        """Test that output is valid JSON."""
        import json

        sim = WakeSimulation(simple_turbine, deficit=NOJDeficit())
        result = sim(jnp.array([0.0]), jnp.array([0.0]), ws_amb=10.0, wd_amb=270.0)

        json_str = result.to_json()
        # Should not raise
        data = json.loads(json_str)
        assert "turbine" in data
        assert "effective_ws" in data


class TestTurbineValidation:
    """Tests for Turbine validation."""

    def test_valid_turbine_passes(self, simple_turbine):
        """Valid turbine should pass validation."""
        simple_turbine.validate()  # Should not raise

    def test_negative_diameter_fails(self):
        """Negative rotor diameter should fail validation."""
        ws = jnp.array([3.0, 10.0, 25.0])
        turbine = Turbine(
            rotor_diameter=-80.0,  # Invalid
            hub_height=70.0,
            power_curve=Curve(ws=ws, values=jnp.array([0.0, 1000.0, 1000.0])),
            ct_curve=Curve(ws=ws, values=jnp.array([0.0, 0.8, 0.2])),
        )
        with pytest.raises(ValueError, match="Rotor diameter must be positive"):
            turbine.validate()

    def test_mismatched_curve_length_fails(self):
        """Mismatched curve lengths should fail validation."""
        turbine = Turbine(
            rotor_diameter=80.0,
            hub_height=70.0,
            power_curve=Curve(
                ws=jnp.array([3.0, 10.0, 25.0]),
                values=jnp.array([0.0, 1000.0]),  # Wrong length
            ),
            ct_curve=Curve(
                ws=jnp.array([3.0, 10.0, 25.0]),
                values=jnp.array([0.0, 0.8, 0.2]),
            ),
        )
        with pytest.raises(ValueError, match="Power curve"):
            turbine.validate()


class TestResultMethods:
    """Tests for SimulationResult computed properties."""

    def test_gross_power(self, simple_turbine):
        """Test gross power calculation."""
        sim = WakeSimulation(simple_turbine, deficit=NOJDeficit())
        wt_x = jnp.array([0.0, 400.0])
        wt_y = jnp.array([0.0, 0.0])

        result = sim(wt_x, wt_y, ws_amb=10.0, wd_amb=270.0)

        gross = result.gross_power()
        actual = result.power()

        # Gross should be >= actual (no wake effects)
        assert jnp.all(gross >= actual - 1e-6)
        # Upstream turbine should have gross == actual (no upstream wakes)
        np.testing.assert_allclose(gross[0, 0], actual[0, 0], rtol=1e-3)

    def test_wake_losses(self, simple_turbine):
        """Test wake loss calculation."""
        sim = WakeSimulation(simple_turbine, deficit=NOJDeficit())
        wt_x = jnp.array([0.0, 400.0])
        wt_y = jnp.array([0.0, 0.0])

        result = sim(wt_x, wt_y, ws_amb=10.0, wd_amb=270.0)

        losses = result.wake_losses()

        # Losses should be in [0, 1]
        assert jnp.all(losses >= 0)
        assert jnp.all(losses <= 1)
        # Downstream turbine should have higher losses
        assert losses[0, 1] > losses[0, 0]

    def test_farm_wake_loss(self, simple_turbine):
        """Test farm-level wake loss calculation."""
        sim = WakeSimulation(simple_turbine, deficit=NOJDeficit())
        wt_x = jnp.array([0.0, 400.0, 800.0])
        wt_y = jnp.array([0.0, 0.0, 0.0])

        result = sim(wt_x, wt_y, ws_amb=10.0, wd_amb=270.0)

        farm_loss = result.farm_wake_loss()

        # Should be a scalar per case
        assert farm_loss.shape == (1,)
        # Should be in [0, 1]
        assert 0 <= farm_loss[0] <= 1


class TestReprMethods:
    """Tests for __repr__ methods."""

    def test_curve_repr(self):
        """Test Curve repr."""
        curve = Curve(
            ws=jnp.array([3.0, 10.0, 25.0]),
            values=jnp.array([0.0, 1000.0, 1000.0]),
        )
        repr_str = repr(curve)
        assert "Curve" in repr_str
        assert "3.0" in repr_str
        assert "25.0" in repr_str

    def test_turbine_repr(self, simple_turbine):
        """Test Turbine repr."""
        repr_str = repr(simple_turbine)
        assert "Turbine" in repr_str
        assert "80" in repr_str  # Diameter

    def test_simulation_repr(self, simple_turbine):
        """Test WakeSimulation repr."""
        sim = WakeSimulation(
            simple_turbine,
            deficit=BastankhahGaussianDeficit(),
            turbulence=CrespoHernandez(),
        )
        repr_str = repr(sim)
        assert "WakeSimulation" in repr_str
        assert "BastankhahGaussianDeficit" in repr_str
        assert "CrespoHernandez" in repr_str


class TestErrorHandling:
    """Tests for proper error handling."""

    def test_mismatched_wt_positions(self, simple_turbine):
        """Mismatched wt_x and wt_y should raise ValueError."""
        sim = WakeSimulation(simple_turbine, deficit=NOJDeficit())
        with pytest.raises(ValueError, match="shape"):
            sim(jnp.array([0.0, 1.0]), jnp.array([0.0]), ws_amb=10.0, wd_amb=270.0)

    def test_mismatched_wind_conditions(self, simple_turbine):
        """Mismatched ws and wd should raise ValueError."""
        sim = WakeSimulation(simple_turbine, deficit=NOJDeficit())
        with pytest.raises(ValueError, match="shape"):
            sim(
                jnp.array([0.0]),
                jnp.array([0.0]),
                ws_amb=jnp.array([10.0, 11.0]),
                wd_amb=jnp.array([270.0]),
            )

    def test_mismatched_wt_types_length(self, simple_turbine):
        """Mismatched wt_types length should raise ValueError."""
        sim = WakeSimulation([simple_turbine], deficit=NOJDeficit())
        with pytest.raises(ValueError, match="wt_types length"):
            sim(
                jnp.array([0.0, 1.0]),
                jnp.array([0.0, 0.0]),
                ws_amb=10.0,
                wd_amb=270.0,
                wt_types=[simple_turbine.type_id],  # Only 1, but 2 turbines
            )
