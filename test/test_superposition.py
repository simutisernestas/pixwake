import jax.numpy as jnp
import numpy as np
import pytest

from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import NiayifarGaussianDeficit, NOJDeficit
from pixwake.superposition import LinearSum, SqrMaxSum, SquaredSum
from pixwake.turbulence.crespo import CrespoHernandez


@pytest.fixture(scope="module")
def turbine():
    return Turbine(
        rotor_diameter=100.0,
        hub_height=100.0,
        power_curve=Curve(ws=jnp.array([0.0, 20.0]), values=jnp.array([0.0, 3000.0])),
        ct_curve=Curve(ws=jnp.array([0.0, 20.0]), values=jnp.array([0.8, 0.8])),
    )


class TestSuperpositionClasses:
    """Tests for the superposition classes themselves."""

    def test_squared_sum_basic(self):
        ambient = jnp.array([10.0, 10.0])
        added = jnp.array([[3.0, 4.0], [5.0, 0.0]])
        superposition = SquaredSum()
        result = superposition(ambient, added)
        # sqrt(3^2 + 4^2) = 5, sqrt(5^2 + 0^2) = 5
        np.testing.assert_allclose(result, jnp.array([5.0, 5.0]), rtol=1e-5)

    def test_sqr_max_sum_basic(self):
        ambient = jnp.array([3.0, 3.0])
        added = jnp.array([[3.0, 4.0], [5.0, 0.0]])
        superposition = SqrMaxSum()
        result = superposition(ambient, added)
        # sqrt(3^2 + max(3,4)^2) = sqrt(9+16) = 5
        # sqrt(3^2 + max(5,0)^2) = sqrt(9+25) ~= 5.83
        np.testing.assert_allclose(result, jnp.array([5.0, jnp.sqrt(34.0)]), rtol=1e-5)


class TestDeficitSuperposition:
    """Tests for superposition in wake deficit models."""

    def test_noj_default_superposition_is_squared_sum(self):
        model = NOJDeficit()
        assert isinstance(model.superposition, SquaredSum)

    def test_niayifar_default_superposition_is_squared_sum(self):
        model = NiayifarGaussianDeficit()
        assert isinstance(model.superposition, SquaredSum)

    def test_noj_custom_superposition(self):
        custom_superposition = LinearSum()
        model = NOJDeficit(superposition=custom_superposition)
        assert model.superposition is custom_superposition

    def test_deficit_custom_superposition_changes_result(self, turbine):
        """Verify that using a different superposition changes the results."""
        xs = jnp.array([0.0, 500.0, 1000.0])
        ys = jnp.zeros(3)
        ws = 10.0
        wd = 270.0

        # Default superposition (SquaredSum)
        default_model = NOJDeficit()
        sim_default = WakeSimulation(turbine, default_model)
        result_default = sim_default(xs, ys, ws, wd)

        # Custom superposition (LinearSum)
        custom_model = NOJDeficit(superposition=LinearSum())
        sim_custom = WakeSimulation(turbine, custom_model)
        result_custom = sim_custom(xs, ys, ws, wd)

        # The results should be different
        assert not jnp.allclose(
            result_default.effective_ws, result_custom.effective_ws
        ), "Custom superposition should change the results"


class TestTurbulenceSuperposition:
    """Tests for superposition in turbulence models."""

    def test_crespo_default_superposition_is_sqr_max_sum(self):
        model = CrespoHernandez()
        assert isinstance(model.superposition, SqrMaxSum)

    def test_crespo_custom_superposition(self):
        custom_superposition = SquaredSum()
        model = CrespoHernandez(superposition=custom_superposition)
        assert model.superposition is custom_superposition

    def test_turbulence_custom_superposition_changes_result(self, turbine):
        """Verify that using a different superposition changes the results."""
        xs = jnp.array([0.0, 500.0, 1000.0])
        ys = jnp.zeros(3)
        ws = 10.0
        wd = 270.0
        ti = 0.1

        deficit_model = NiayifarGaussianDeficit()

        # Default superposition (SqrMaxSum)
        default_turbulence = CrespoHernandez()
        sim_default = WakeSimulation(turbine, deficit_model, default_turbulence)
        result_default = sim_default(xs, ys, ws, wd, ti)

        # Custom superposition (SquaredSum)
        custom_turbulence = CrespoHernandez(superposition=SquaredSum())
        sim_custom = WakeSimulation(turbine, deficit_model, custom_turbulence)
        result_custom = sim_custom(xs, ys, ws, wd, ti)

        # The results should be different for most cases
        # (depends on the configuration, but should differ in general)
        assert result_default.effective_ti is not None
        assert result_custom.effective_ti is not None

        assert not jnp.allclose(
            result_default.effective_ti, result_custom.effective_ti
        ), "Custom superposition should change the turbulence results"
