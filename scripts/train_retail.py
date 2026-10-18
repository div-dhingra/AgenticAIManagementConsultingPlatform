"""Train and evaluate the retail co-purchase model.

Usage:
    python scripts/train_retail.py                       # uses data/raw/retail/market_basket_transactions.csv
    python scripts/train_retail.py --import-csv PATH     # copy a CSV into the repo first, then train
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from consulting_platform.retail.train import train_and_evaluate

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = Path("raw") / "retail" / "market_basket_transactions.csv"


def main() -> int:
    data_dir = REPO_ROOT / os.environ.get("DATA_DIR", "data")
    artifact_dir = REPO_ROOT / os.environ.get("ARTIFACT_DIR", "artifacts")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--import-csv", type=Path, help="copy this CSV to the repo data directory before training")
    parser.add_argument("--csv", type=Path, default=data_dir / DEFAULT_CSV, help="repo-relative training CSV")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    if args.import_csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.import_csv, args.csv)
        print(f"Imported {args.import_csv.name} -> {args.csv.relative_to(REPO_ROOT)}")
    if not args.csv.exists():
        print(f"Training CSV not found: {args.csv}. Use --import-csv PATH.", file=sys.stderr)
        return 1

    report = train_and_evaluate(args.csv, artifact_dir / "retail", REPO_ROOT / "evaluation" / "retail", REPO_ROOT, k=args.k)
    m, b = report["model"], report["popularity_baseline"]
    print(f"Model {report['model_version']}: {report['n_training']} training baskets, {report['n_rules']} rules")
    if report.get("input_issues"):
        print(f"Input issues: {report['input_issues']}")
    print(f"Hit rate@{args.k}: model {m['hit_rate_at_k']:.3f} (coverage {m['coverage']:.3f}) vs popularity {b['hit_rate_at_k']:.3f}")
    print("Wrote artifacts/retail/model.json and evaluation/retail/report.{json,md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
