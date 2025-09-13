import pytest

pytest.skip("integration test requires selenium and network", allow_module_level=True)


import re
import subprocess
import sys


def test_cli_prints_200_rows():
    cmd = [
        sys.executable,
        "-m",
        "highest_volatility",
        "--top-n",
        "300",
        "--print-top",
        "200",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    # Count data lines that start with an uppercase ticker-like token
    lines = [ln for ln in r.stdout.splitlines() if re.match(r"^[A-Z0-9\.-]+\s+", ln)]
    assert len(lines) == 200, f"Expected 200 rows, got {len(lines)}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"

