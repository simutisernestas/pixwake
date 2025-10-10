import jax.numpy as jnp
import numpy as np
import pytest

from pixwake.rotor_avg_models import AreaOverlapAvgModel


@pytest.mark.parametrize(
    "R1, R2, d, expected",
    [
        (1.0, 1.0, 0.0, 1.0),  # Full overlap
        (1.0, 1.0, 2.0, 0.0),  # No overlap
        (1.0, 1.0, 1.0, 0.3910022),  # Partial overlap
    ],
)
def test_area_overlap_avg_model(R1, R2, d, expected):
    """Test the AreaOverlapAvgModel with known inputs and expected outputs."""
    model = AreaOverlapAvgModel()
    factor = model._cal_overlapping_area_factor(
        jnp.array(R1), jnp.array(R2), jnp.array(d)
    )
    np.testing.assert_allclose(factor, expected, rtol=1e-5)