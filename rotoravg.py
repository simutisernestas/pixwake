from py_wake.rotor_avg_models import CGIRotorAvg
from numpy import newaxis as na
import numpy as np

np.random.seed(42)


class StandaloneCGIRotorAvg:
    """
    Standalone implementation of CGI (Composite Gauss Integration) Rotor Averaging Model
    This implementation removes inheritance complexity and clearly explains each component.
    CGI uses predefined integration points and weights to approximate rotor-averaged values.

    Composite Gauss Integration Rotor Averaging Model

    This model computes rotor-averaged quantities by evaluating a function at
    specific points across the rotor disk and combining them with weights.

    The CGI method uses predetermined node locations and weights optimized for
    circular rotor averaging. Different numbers of points provide different
    accuracy-performance tradeoffs.
    """

    def __init__(self, n_points=7):
        """
        Initialize CGI rotor averaging model

        Parameters
        ----------
        n_points : int
            Number of integration points: 4, 7, 9, or 21
            More points = higher accuracy but slower computation
        """
        self.n_points = n_points

        # Get predefined node positions and weights for this configuration
        self.nodes_x, self.nodes_y, self.weights = self._get_cgi_nodes_and_weights(
            n_points
        )

    def _get_cgi_nodes_and_weights(self, n):
        """
        Get CGI integration points and weights for circular rotor

        These are predetermined optimal locations for integrating over a circle.
        Coordinates are normalized to rotor radius (range: -1 to 1)

        Returns
        -------
        nodes_x : array
            X-coordinates of integration points (normalized by radius)
        nodes_y : array
            Y-coordinates of integration points (normalized by radius)
        weights : array
            Integration weights (sum to 1.0)
        """
        pm = np.array([[-1, -1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, 1]])
        x, y, w = {
            4: pm * [0.5, 0.5, 1 / 4],
            7: np.r_[
                [
                    [0, 0, 1 / 4],
                    [-np.sqrt(2 / 3), 0, 1 / 8],
                    [np.sqrt(2 / 3), 0, 1 / 8],
                ],
                pm * [np.sqrt(1 / 6), np.sqrt(1 / 2), 1 / 8],
            ],
            9: np.r_[
                [
                    [0, 0, 1 / 6],
                    [-1, 0, 1 / 24],
                    [1, 0, 1 / 24],
                    [0, -1, 1 / 24],
                    [0, 1, 1 / 24],
                ],
                pm * [1 / 2, 1 / 2, 1 / 6],
            ],
            21: np.r_[
                [[0, 0, 1 / 9]],
                [
                    [
                        np.sqrt((6 - np.sqrt(6)) / 10) * np.cos(2 * np.pi * k / 10),
                        np.sqrt((6 - np.sqrt(6)) / 10) * np.sin(2 * np.pi * k / 10),
                        (16 + np.sqrt(6)) / 360,
                    ]
                    for k in range(1, 11)
                ],
                [
                    [
                        np.sqrt((6 + np.sqrt(6)) / 10) * np.cos(2 * np.pi * k / 10),
                        np.sqrt((6 + np.sqrt(6)) / 10) * np.sin(2 * np.pi * k / 10),
                        (16 - np.sqrt(6)) / 360,
                    ]
                    for k in range(1, 11)
                ],
            ],
        }[n].T
        return (np.array(x), np.array(y), np.array(w))

    def __call__(self, func, D_dst_ijl, hcw_ijlk, dw_ijlk, dh_ijlk, **kwargs):
        """
        Compute rotor-averaged value by evaluating func at integration points

        Parameters
        ----------
        func : callable
            Function to evaluate at each rotor point
            Should accept **kwargs and return array with shape (..., n_points)
        D_dst_ijl : array
            Destination (rotor) diameter [m]
            Shape: (i turbines, j points, l wind directions)
        hcw_ijlk : array
            Horizontal crosswind distance from source to destination [m]
            Shape: (i, j, l, k wind speeds)
        dw_ijlk : array
            Downwind distance [m]
            Shape: (i, j, l, k)
        dh_ijlk : array
            Vertical distance [m]
            Shape: (i, j, l, k)
        **kwargs : dict
            Additional arguments passed to func

        Returns
        -------
        result_ijlk : array
            Rotor-averaged result
            Shape: (i, j, l, k)
        """
        # Get rotor radius (half diameter)
        # D_dst_ijl has shape (i, j, l), we need (i, j, l, k, 1) for broadcasting
        R_ijlk1 = D_dst_ijl[:, :, :, na, na] / 2.0

        # Calculate positions of integration points on rotor
        # nodes_x, nodes_y are normalized (-1 to 1), multiply by radius
        # Add new axis for broadcasting: (n_points,) -> (1, 1, 1, 1, n_points)
        node_x_offset = self.nodes_x[na, na, na, na, :] * R_ijlk1
        node_y_offset = self.nodes_y[na, na, na, na, :] * R_ijlk1

        # Adjust crosswind and vertical distances to integration points
        # hcw_ijlk has shape (i, j, l, k), expand to (i, j, l, k, n_points)
        hcw_ijlk_expanded = hcw_ijlk[:, :, :, :, na]
        dh_ijlk_expanded = dh_ijlk[:, :, :, :, na]

        # Add offsets to get distance to each integration point
        hcw_at_nodes = hcw_ijlk_expanded + node_x_offset  # horizontal offset
        dh_at_nodes = dh_ijlk_expanded + node_y_offset  # vertical offset

        # Downwind distance doesn't change across rotor
        dw_at_nodes = np.broadcast_to(dw_ijlk[:, :, :, :, na], hcw_at_nodes.shape)

        # Prepare kwargs for function evaluation at all points
        eval_kwargs = kwargs.copy()
        eval_kwargs.update(
            {"hcw_ijlk": hcw_at_nodes, "dh_ijlk": dh_at_nodes, "dw_ijlk": dw_at_nodes}
        )

        # Evaluate function at all integration points
        # Expected output shape: (i, j, l, k, n_points)
        values_at_nodes = func(**eval_kwargs)

        # Weight and sum to get rotor average
        # weights has shape (n_points,), reshape to (1, 1, 1, 1, n_points)
        weights_broadcast = self.weights[na, na, na, na, :]

        # Weighted sum over last dimension (integration points)
        rotor_averaged = np.sum(values_at_nodes * weights_broadcast, axis=-1)

        return rotor_averaged


# Test to verify it matches PyWake
if __name__ == "__main__":
    np.random.seed(42)  # For reproducible test

    n_points = 21
    standalone_model = StandaloneCGIRotorAvg(n_points)

    _cache = None

    # Dummy function that returns random values at each point (eg. wind speed)
    # This would be calling a deficit model in practice (or turbulence).
    def dummy_func(**kwargs):
        global _cache
        # Gets expanded in the rotor avg model and adds a n_points dimension
        hcw_shape = kwargs["hcw_ijlk"].shape  # (i, j, l, k, n_points)
        if _cache is None:
            _cache = np.random.uniform(8, 9, hcw_shape)
        return _cache

    # Test inputs matching your example
    D_dst_ijl = np.ones((1, 1, 1)) * 120  # 120m diameter
    hcw_ijlk = np.zeros((1, 1, 1, 1))  # At rotor center all points
    dw_ijlk = np.ones((1, 1, 1, 1)) * 200  # 200m downstream
    dh_ijlk = np.zeros((1, 1, 1, 1))  # Same height for all points

    result = standalone_model(dummy_func, D_dst_ijl, hcw_ijlk, dw_ijlk, dh_ijlk)

    print(f"Standalone CGI Rotor Averaged Result: {result}")
    print(f"Shape: {result.shape}")
    print(f"Number of integration points used: {n_points}")
    print(f"Integration weights sum: {standalone_model.weights.sum():.6f}")
    print(f"Value mean: {_cache.mean()}")

    cgi_ravg = CGIRotorAvg(n_points)
    pw_res = cgi_ravg(
        dummy_func,
        D_dst_ijl,
        hcw_ijlk=hcw_ijlk,
        dw_ijlk=dw_ijlk,
        dh_ijlk=dh_ijlk,
    )
    assert np.allclose(result, pw_res), "Results do not match PyWake CGI model!"

    import warnings

    warnings.filterwarnings("ignore")

    from py_wake.examples.data.hornsrev1 import V80, Hornsrev1Site
    from py_wake import BastankhahGaussian

    # this is for debugger run inspecting what happens inside the rotor avg model
    windTurbines = V80()
    R = windTurbines.diameter() / 2
    site = Hornsrev1Site()
    m = CGIRotorAvg(21)
    wfm = BastankhahGaussian(site, windTurbines, rotorAvgModel=m)
    ws_eff = wfm([0, 200], [0, 0], wd=270, ws=10).WS_eff_ilk[1, 0, 0]
