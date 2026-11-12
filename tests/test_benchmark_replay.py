import json

from scripts.benchmark_replay import _replay_artifacts


def test_replay_attempts_preserve_previous_reports_and_input_provenance(tmp_path):
    output = tmp_path / "timing.json"
    output.write_text('{"previous_complete_result": true}\n')
    first_manifest = {"candidate_source_sha256": "first", "dataset_sha256": "same dataset"}
    first = _replay_artifacts(output, first_manifest)
    (first / "baseline_1.json").write_text('{"saved_before_interruption": true}\n')
    second_manifest = {**first_manifest, "candidate_source_sha256": "second"}
    second = _replay_artifacts(output, second_manifest)
    assert first != second
    assert json.loads((first / "manifest.json").read_text()) == first_manifest
    assert json.loads((second / "manifest.json").read_text()) == second_manifest
    assert json.loads((first / "baseline_1.json").read_text()) == {"saved_before_interruption": True}
    assert json.loads(output.read_text()) == {"previous_complete_result": True}
