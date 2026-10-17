"""Repeatable training + evaluation pipeline (entry point: scripts/train_retail.py)."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

from .evaluate import EVAL_SEED, evaluate, find_demo_cases, render_markdown
from .model import (
    MIN_CONFIDENCE,
    MIN_EVIDENCE_COUNT,
    MIN_LIFT,
    MODEL_VERSION,
    SPLIT_SEED,
    TEST_FRACTION,
    RetailModel,
    split_ids,
    utc_timestamp,
)
from .preprocess import file_sha256, load_transactions

MODEL_FILENAME = "model.json"


def display_path(path: Path, root: Path) -> str:
    """Repo-relative path when possible; never record a personal absolute path."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def train_and_evaluate(csv_path: Path, artifact_dir: Path, report_dir: Path, repo_root: Path, k: int = 5) -> dict[str, Any]:
    import mlxtend
    import pandas

    data = load_transactions(csv_path)
    sha = file_sha256(csv_path)
    train_ids, test_ids = split_ids(list(data.transactions))
    created_at = utc_timestamp()
    version = f"{MODEL_VERSION}+{sha[:8]}"

    input_info = {
        "path": display_path(csv_path, repo_root),
        "sha256": sha,
        "transactions": len(data.transactions),
        "products": len(data.catalog),
        "max_basket_length": data.max_basket_length,
        "duplicate_items_removed": data.duplicate_items_removed,
        "issues": data.issues,
    }
    eval_config = {"k": k, "hide_one_seed": EVAL_SEED, "split": "held-out test IDs only"}
    model = RetailModel.train(
        data.transactions,
        data.catalog,
        train_ids,
        max_len=data.max_basket_length,
        model_version=version,
        metadata={
            "created_at": created_at,
            "input": input_info,
            "split": {"seed": SPLIT_SEED, "test_fraction": TEST_FRACTION, "train_ids": train_ids, "test_ids": test_ids},
            "rule_settings": {
                "miner": "mlxtend.fpgrowth, then exact recount",
                "min_support": f"{MIN_EVIDENCE_COUNT}/N_training",
                "consequents": "single product",
                "antecedent_match": "equals full cart as a set",
                "min_evidence_count": MIN_EVIDENCE_COUNT,
                "min_confidence": MIN_CONFIDENCE,
                "min_lift_exclusive": MIN_LIFT,
                "ranking": "confidence * (lift - 1) desc, evidence_count desc, product_id asc",
            },
            "evaluation_config": eval_config,
            "library_versions": {"python": platform.python_version(), "mlxtend": mlxtend.__version__, "pandas": pandas.__version__},
        },
    )
    model.save(artifact_dir / MODEL_FILENAME)

    report = {
        "model_version": version,
        "generated_at": created_at,
        "input": {"path": input_info["path"], "sha256": sha},
        "input_issues": data.issues,
        "n_training": model.n_training,
        "n_rules": len(model.rules),
        **evaluate(model, data.transactions, test_ids, k=k),
        "demo_cases": find_demo_cases(model),
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (report_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    return report
