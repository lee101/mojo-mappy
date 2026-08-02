import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def build_kernel():
    root = Path(__file__).parents[1]
    subprocess.run(["bash", "build/build.sh"], cwd=root, check=True)
