import json

import pytest

from medical_imaging_qa.rules import QARules


def test_rules_load_from_json(tmp_path):
    path = tmp_path / "rules.json"
    path.write_text(json.dumps({"min_component_voxels": 20}), encoding="utf-8")
    rules = QARules.from_json(path)
    assert rules.min_component_voxels == 20


def test_rules_reject_unknown_key(tmp_path):
    path = tmp_path / "rules.json"
    path.write_text(json.dumps({"unknown_rule": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown QA rule"):
        QARules.from_json(path)
