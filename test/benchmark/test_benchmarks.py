import os
import subprocess


def test_running_small_benchmark_pywake():
    test_dir = os.path.dirname(__file__)
    subprocess.run(
        [
            "python",
            test_dir + "/against_pywake.py",
            "--n_turbines",
            "8",
            "--spacings",
            "5",
            "--target",
            "pywake",
        ],
        check=True,
    )


def test_running_small_benchmark_pixwake():
    test_dir = os.path.dirname(__file__)
    subprocess.run(
        [
            "python",
            test_dir + "/against_pywake.py",
            "--n_turbines",
            "8",
            "--spacings",
            "5",
            "--target",
            "pixwake",
        ],
        check=True,
    )
