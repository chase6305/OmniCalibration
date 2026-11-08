import copy
import hashlib
import json

import pytest

from scripts.benchmark import verify_resume_inputs


@pytest.mark.parametrize("changed", ["config", "urdf", "mesh", "plan"])
def test_changed_inputs_cannot_reuse_a_frozen_batch(tmp_path, changed):
    assets = tmp_path / "asset"
    assets.mkdir()
    urdf, mesh = assets / "robot.urdf", assets / "mesh.glb"
    urdf.write_bytes(b"original URDF")
    mesh.write_bytes(b"original mesh")
    configs = tmp_path / "configs"
    configs.mkdir()
    config = configs / "model.json"
    config.write_text(json.dumps({"asset_root": "asset", "urdf": "robot.urdf"}))

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    plan = {"seeds": [7], "cases": [{"id": "one", "config": "configs/model.json"}]}
    previous = {
        "source_sha256": "source",
        "plan": copy.deepcopy(plan),
        "seeds": [7],
        "environment": {"test": "environment"},
        "input_configs": {
            "configs/model.json": {
                "sha256": digest(config),
                "asset_root": str(assets),
                "urdf_sha256": digest(urdf),
            }
        },
    }
    output = tmp_path / "batch"
    (output / "runs").mkdir(parents=True)
    (output / "runs" / "one.json").write_text(
        json.dumps(
            {
                "source_sha256": "source",
                "result": {"mesh_sources": [{"path": str(mesh), "sha256": digest(mesh)}]},
            }
        )
    )
    args = [output, previous, plan, [7], "source", previous["environment"], tmp_path]
    verify_resume_inputs(*args)
    if changed == "plan":
        plan["purpose"] = "changed experiment"
    else:
        path = {"config": config, "urdf": urdf, "mesh": mesh}[changed]
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="differs"):
        verify_resume_inputs(*args)
