"""Tests for pixwake.utils module."""

import jax.numpy as jnp
import numpy as np

from pixwake.utils import ct2a_madsen, ct2a_mom1d


class TestCt2aMadsen:
    """Tests for Madsen polynomial approximation of Ct to induction."""

    def test_zero_ct(self):
        """Zero thrust coefficient should give zero induction."""
        ct = jnp.array([0.0])
        a = ct2a_madsen(ct)
        np.testing.assert_allclose(a, 0.0, atol=1e-10)

    def test_typical_ct_values(self):
        """Test with typical operating Ct values."""
        ct = jnp.array([0.4, 0.6, 0.8])
        a = ct2a_madsen(ct)
        # Induction should be positive and less than 1
        assert jnp.all(a > 0)
        assert jnp.all(a < 1)
        # Higher Ct should give higher induction
        assert a[0] < a[1] < a[2]

    def test_high_ct(self):
        """Test with high Ct values near theoretical limit."""
        ct = jnp.array([0.9, 0.95, 1.0])
        a = ct2a_madsen(ct)
        # Should still return finite positive values
        assert jnp.all(jnp.isfinite(a))
        assert jnp.all(a > 0)

    def test_array_shapes(self):
        """Test that output shape matches input shape."""
        ct_1d = jnp.array([0.5, 0.6, 0.7])
        ct_2d = jnp.ones((3, 4)) * 0.5

        a_1d = ct2a_madsen(ct_1d)
        a_2d = ct2a_madsen(ct_2d)

        assert a_1d.shape == ct_1d.shape
        assert a_2d.shape == ct_2d.shape

    def test_custom_coefficients(self):
        """Test with custom polynomial coefficients."""
        ct = jnp.array([0.5])
        # Default coefficients
        a_default = ct2a_madsen(ct)
        # Custom coefficients (doubled)
        custom_coefs = (0.492, 0.1172, 0.1766)
        a_custom = ct2a_madsen(ct, ct2ap=custom_coefs)
        # Custom should give different (doubled) result
        np.testing.assert_allclose(a_custom, a_default * 2, rtol=1e-6)

    def test_monotonicity(self):
        """Induction should increase monotonically with Ct."""
        ct = jnp.linspace(0.0, 0.9, 100)
        a = ct2a_madsen(ct)
        diffs = jnp.diff(a)
        assert jnp.all(diffs >= 0)


class TestCt2aMom1d:
    """Tests for 1D momentum theory Ct to induction conversion."""

    def test_zero_ct(self):
        """Zero thrust coefficient should give zero induction."""
        ct = jnp.array([0.0])
        a = ct2a_mom1d(ct)
        np.testing.assert_allclose(a, 0.0, atol=1e-10)

    def test_betz_limit(self):
        """At Betz optimal Ct=8/9, induction should be 1/3."""
        ct = jnp.array([8.0 / 9.0])
        a = ct2a_mom1d(ct)
        np.testing.assert_allclose(a, 1.0 / 3.0, rtol=1e-6)

    def test_ct_equals_one(self):
        """At Ct=1, induction should be close to 0.5."""
        ct = jnp.array([1.0])
        a = ct2a_mom1d(ct)
        np.testing.assert_allclose(a, 0.5, rtol=1e-3)

    def test_ct_clamping(self):
        """Ct values above 1 should be clamped to 1."""
        ct = jnp.array([1.5, 2.0, 10.0])
        a = ct2a_mom1d(ct)
        # All should give same result as ct=1
        expected = ct2a_mom1d(jnp.array([1.0]))[0]
        np.testing.assert_allclose(a, expected, rtol=1e-6)

    def test_typical_ct_values(self):
        """Test with typical operating Ct values."""
        ct = jnp.array([0.4, 0.6, 0.8])
        a = ct2a_mom1d(ct)
        # Induction should be positive and less than 0.5
        assert jnp.all(a > 0)
        assert jnp.all(a < 0.5)
        # Higher Ct should give higher induction
        assert a[0] < a[1] < a[2]

    def test_array_shapes(self):
        """Test that output shape matches input shape."""
        ct_1d = jnp.array([0.5, 0.6, 0.7])
        ct_2d = jnp.ones((3, 4)) * 0.5

        a_1d = ct2a_mom1d(ct_1d)
        a_2d = ct2a_mom1d(ct_2d)

        assert a_1d.shape == ct_1d.shape
        assert a_2d.shape == ct_2d.shape

    def test_monotonicity(self):
        """Induction should increase monotonically with Ct (up to 1)."""
        ct = jnp.linspace(0.0, 1.0, 100)
        a = ct2a_mom1d(ct)
        diffs = jnp.diff(a)
        assert jnp.all(diffs >= 0)

    def test_numerical_stability_small_ct(self):
        """Should handle very small Ct values without numerical issues."""
        ct = jnp.array([1e-10, 1e-8, 1e-6])
        a = ct2a_mom1d(ct)
        assert jnp.all(jnp.isfinite(a))
        assert jnp.all(a >= 0)


class TestCt2aComparison:
    """Compare the two Ct to induction methods."""

    def test_both_give_zero_for_zero_ct(self):
        """Both methods should give zero induction for zero Ct."""
        ct = jnp.array([0.0])
        a_madsen = ct2a_madsen(ct)
        a_mom1d = ct2a_mom1d(ct)
        np.testing.assert_allclose(a_madsen, 0.0, atol=1e-10)
        np.testing.assert_allclose(a_mom1d, 0.0, atol=1e-10)

    def test_similar_at_low_ct(self):
        """Both methods should give similar results at low Ct."""
        ct = jnp.array([0.1, 0.2, 0.3])
        a_madsen = ct2a_madsen(ct)
        a_mom1d = ct2a_mom1d(ct)
        # Should be within 20% of each other at low Ct
        np.testing.assert_allclose(a_madsen, a_mom1d, rtol=0.2)

    def test_both_monotonic(self):
        """Both methods should be monotonically increasing."""
        ct = jnp.linspace(0.1, 0.9, 50)
        a_madsen = ct2a_madsen(ct)
        a_mom1d = ct2a_mom1d(ct)
        assert jnp.all(jnp.diff(a_madsen) > 0)
        assert jnp.all(jnp.diff(a_mom1d) > 0)
