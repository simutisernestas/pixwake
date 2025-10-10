import os
import subprocess


def test_running_small_benchmark_against_pywake():
    test_dir = os.path.dirname(__file__)
    subprocess.run(
        [
            "python",
            test_dir + "/against_pywake.py",
            "--n_turbines",
            "8",
            "--spacings",
            "5",
        ],
        check=True,
    )
