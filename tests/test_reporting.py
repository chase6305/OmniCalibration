import json

import numpy as np
import pytest

from omnicalib.reporting import write_json


def test_json_reports_are_strict_and_numpy_compatible(tmp_path):
    path = tmp_path / "result.json"
    write_json(path, {"array": np.array([1.0, np.inf]), "accepted": np.bool_(True)})
    assert json.loads(path.read_text()) == {"array": [1.0, None], "accepted": True}


def test_serialization_failure_preserves_previous_checkpoint(tmp_path):
    path = tmp_path / "result.json"
    write_json(path, {"completed": 3})
    with pytest.raises(TypeError):
        write_json(path, {"invalid": object()})
    assert json.loads(path.read_text()) == {"completed": 3}
    assert list(tmp_path.iterdir()) == [path]
