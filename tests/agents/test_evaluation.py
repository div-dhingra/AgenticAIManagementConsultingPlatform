import importlib.util
from pathlib import Path


def test_unmeasured_quality_is_null_not_perfect():
    path = Path("scripts/evaluate_consulting.py")
    spec = importlib.util.spec_from_file_location("evaluate_consulting", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    case = {
        "id": "x",
        "expected_domains": ["technology"],
        "request": {"question": "Cisco router"},
        "gold_evidence_ids": [],
        "human_review": None,
    }
    metrics = module.score([{"case_id": "x", "response": None}], [case], "run-id", "live")
    assert metrics["human_citation_support"]["value"] is None
    assert metrics["human_response_quality"]["mean"] is None
    assert metrics["final_answer_evidence_hit"]["value"] is None
    assert metrics["completed_requests"]["value"] == 0
