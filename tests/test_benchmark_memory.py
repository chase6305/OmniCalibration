import os
import sys
from pathlib import Path

import pytest

from scripts.benchmark_proximity import _isolated_python


@pytest.mark.skipif(sys.platform != "linux", reason="Linux exec high-water accounting")
def test_memory_worker_does_not_count_the_loaded_parent_image(tmp_path):
    # Touch a modest parent-only allocation. A direct fork/exec can carry this
    # resident image into ru_maxrss even when the worker only imports resource.
    payload = bytearray(64 * 1024 * 1024)
    payload[::4096] = b"x" * (len(payload) // 4096)
    status = dict(
        line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines()
        if ":" in line
    )
    parent_rss = int(status["VmRSS"].split()[0]) * 1024
    code = "import resource\nprint(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)"
    worker_peak = int(_isolated_python(code, [], tmp_path, os.environ.copy()))
    assert payload[0] == ord("x")
    assert worker_peak < parent_rss - 32 * 1024 * 1024
