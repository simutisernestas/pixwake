import os
import subprocess
import time


def test_running_small_benchmark_pywake():
    test_dir = os.path.dirname(__file__)
    run_id = int(time.time()).__str__()
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
            "--run_id",
            run_id,
        ],
        check=True,
    )

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
            "--run_id",
            run_id,
        ],
        check=True,
    )

    subprocess.run(
        ["python", test_dir + "/against_pywake.py", "--plot", "--run_id", run_id],
        check=True,
    )
