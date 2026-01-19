"""Tests for pixwake.jax_utils module."""

import jax.numpy as jnp
import numpy as np

from pixwake.jax_utils import default_float_type, get_float_eps, ssqrt


class TestDefaultFloatType:
    """Tests for default_float_type function."""

    def test_returns_valid_float_type(self):
        """Should return either float32 or float64 depending on JAX config."""
        dtype = default_float_type()
        assert dtype in (jnp.float32, jnp.float64)

    def test_returns_jax_dtype(self):
        """Return value is a JAX-compatible dtype."""
        dtype = default_float_type()
        # Should be usable for creating arrays
        arr = jnp.zeros(5, dtype=dtype)
        assert arr.dtype == dtype


class TestGetFloatEps:
    """Tests for get_float_eps function."""

    def test_returns_positive_value(self):
        """Machine epsilon should be positive."""
        eps = get_float_eps()
        assert eps > 0

    def test_returns_small_value(self):
        """Machine epsilon should be small but non-zero."""
        eps = get_float_eps()
        assert eps < 1e-5  # Should be much smaller than this
        assert eps > 0

    def test_eps_consistent_with_dtype(self):
        """Epsilon should match the expected precision."""
        eps = get_float_eps()
        dtype = default_float_type()
        expected_eps = float(jnp.finfo(dtype).eps)
        assert eps == expected_eps


class TestSsqrt:
    """Tests for gradient-stable square root function."""

    def test_positive_values(self):
        """ssqrt should match jnp.sqrt for positive values."""
        x = jnp.array([1.0, 4.0, 9.0, 16.0])
        result = ssqrt(x)
        expected = jnp.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_zero_value(self):
        """ssqrt of zero should return small positive value (eps)."""
        result = ssqrt(jnp.array([0.0]))
        # Should return sqrt(eps) which is small but positive
        assert result[0] > 0
        assert result[0] < 1e-3

    def test_negative_values(self):
        """ssqrt should handle negative values gracefully (return sqrt(eps))."""
        x = jnp.array([-1.0, -100.0])
        result = ssqrt(x)
        # Should return sqrt(eps) for negative inputs
        assert jnp.all(result > 0)
        assert jnp.all(result < 1e-3)

    def test_small_negative_from_float_errors(self):
        """ssqrt should handle small negative values from floating-point errors."""
        # This can happen in calculations like a**2 - b**2 when a ≈ b
        x = jnp.array([-1e-15, -1e-10])
        result = ssqrt(x)
        assert jnp.all(jnp.isfinite(result))
        assert jnp.all(result > 0)

    def test_mixed_values(self):
        """ssqrt should handle mixed positive/negative arrays."""
        x = jnp.array([4.0, -1.0, 9.0, 0.0, -0.5])
        result = ssqrt(x)
        assert jnp.all(jnp.isfinite(result))
        assert jnp.all(result > 0)
        # Positive values should give expected sqrt
        np.testing.assert_allclose(result[0], 2.0, rtol=1e-6)
        np.testing.assert_allclose(result[2], 3.0, rtol=1e-6)

    def test_gradient_stability(self):
        """ssqrt gradient should not produce NaN near zero."""
        import jax

        def f(x):
            return ssqrt(x).sum()

        # Gradient at very small positive value
        grad_fn = jax.grad(f)
        small_x = jnp.array([1e-10, 1e-15, 1e-20])
        grads = grad_fn(small_x)
        assert jnp.all(jnp.isfinite(grads))

    def test_preserves_shape(self):
        """ssqrt should preserve input array shape."""
        x = jnp.ones((3, 4, 5))
        result = ssqrt(x)
        assert result.shape == x.shape
