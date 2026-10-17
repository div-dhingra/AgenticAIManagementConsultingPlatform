"""Reproducible consulting evaluation; missing/manual metrics stay explicitly null."""

import argparse
import asyncio
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from consulting_platform.agents.service import create_consulting_service, keyword_route
from consulting_platform.cli import load_factory
from consulting_platform.config import Settings
from consulting_platform.contracts import (
    ConsultRequest,
    ConsultResponse,
    ServiceUnavailableError,
    UpstreamModelError,
    UpstreamTimeoutError,
)


def ratio(numerator, denominator):
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
    }


def score(rows: list[dict], cases: list[dict], run_id: str, mode: str) -> dict:
    lookup = {c["id"]: c for c in cases}
    exact = tp = fp = fn = valid_refs = checked_refs = expected_status = correct_status = 0
    gold_hits = gold_count = human_supported = human_reviewed = baseline_exact = 0
    relevance, quality, latencies = [], [], []
    successes = 0
    for row in rows:
        case = lookup[row["case_id"]]
        gold = set(case["expected_domains"])
        baseline_exact += {d.value for d in keyword_route(case["request"]["question"])} == gold
        if not row.get("response"):
            fn += len(gold)
            if "expected_status" in case:
                expected_status += 1
            continue
        response = ConsultResponse.model_validate(row["response"])
        successes += 1
        predicted = {d.value for d in response.domains}
        exact += predicted == gold
        tp += len(predicted & gold)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
        if "expected_status" in case:
            expected_status += 1
            correct_status += response.status == case["expected_status"]
        citation_ids = {c.evidence_id for c in response.citations}
        references = [
            i for item in [*response.findings, *response.recommendations] for i in item.evidence_ids
        ]
        checked_refs += len(references)
        valid_refs += sum(i in citation_ids for i in references)
        if case.get("gold_evidence_ids") and mode == "live":
            gold_count += 1
            gold_hits += bool(citation_ids & set(case["gold_evidence_ids"]))
        human = case.get("human_review")
        if human and human.get("run_id") == run_id and mode == "live":
            support = human.get("citation_support") or {}
            n, d = support.get("supported", 0), support.get("reviewed", 0)
            if not 0 <= n <= d:
                raise ValueError("Invalid human citation support counts")
            human_supported += n
            human_reviewed += d
            for key, output in [
                ("recommendation_relevance", relevance),
                ("response_quality", quality),
            ]:
                value = human.get(key)
                if value is not None:
                    if not 1 <= value <= 5:
                        raise ValueError("Human ratings must be 1..5")
                    output.append(value)
        latencies.append(row["latency_seconds"])
    return {
        "interpretation": "synthetic integration checks only"
        if mode == "demo"
        else "live evaluation; inspect missing labels",
        "completed_requests": ratio(successes, len(rows)),
        "routing_exact_match": ratio(exact, len(rows)),
        "routing_micro_precision": ratio(tp, tp + fp),
        "routing_micro_recall": ratio(tp, tp + fn),
        "keyword_baseline_exact_match": ratio(baseline_exact, len(rows)),
        "expected_status_accuracy": ratio(correct_status, expected_status),
        "citation_reference_validity": ratio(valid_refs, checked_refs),
        "final_answer_evidence_hit": ratio(gold_hits, gold_count),
        "human_citation_support": ratio(human_supported, human_reviewed),
        "human_recommendation_relevance": {
            "mean": statistics.mean(relevance) if relevance else None,
            "reviewed": len(relevance),
        },
        "human_response_quality": {
            "mean": statistics.mean(quality) if quality else None,
            "reviewed": len(quality),
        },
        "latency_seconds": {
            "mean": statistics.mean(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
    }


async def evaluate(service, cases):
    rows = []
    for case in cases:
        start = time.monotonic()
        try:
            response = await service.run_consultation(
                ConsultRequest.model_validate(case["request"]),
                request_id=str(uuid4()),
            )
            row = {"case_id": case["id"], "response": response.model_dump(mode="json")}
        except (ServiceUnavailableError, UpstreamTimeoutError, UpstreamModelError) as exc:
            row = {"case_id": case["id"], "response": None, "error": type(exc).__name__}
        rows.append({**row, "latency_seconds": time.monotonic() - start})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("evaluation/consulting/cases.jsonl"))
    parser.add_argument("--mode", choices=["live", "demo"], default="demo")
    parser.add_argument("--split", choices=["development", "test"], default="development")
    parser.add_argument("--knowledge-factory")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--score-existing", type=Path)
    args = parser.parse_args()
    all_cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    if args.score_existing:
        report = json.loads(args.score_existing.read_text())
        report["metrics"] = score(report["rows"], all_cases, report["run_id"], report["mode"])
        output = args.output or args.score_existing.with_suffix(".scored.json")
    else:
        cases = [c for c in all_cases if c["split"] == args.split]
        settings = Settings.from_env()
        settings.mode = args.mode
        knowledge = load_factory(args.knowledge_factory) if args.knowledge_factory else None
        service = create_consulting_service(knowledge, settings=settings)
        run_id = str(uuid4())
        rows = asyncio.run(evaluate(service, cases))
        report = {
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "split": args.split,
            "provider": "fake" if args.mode == "demo" else settings.provider,
            "model": "deterministic fixture" if args.mode == "demo" else settings.model,
            "rows": rows,
            "metrics": score(rows, cases, run_id, args.mode),
        }
        output = args.output or Path(f"evaluation/consulting/runs/{run_id}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(output), "metrics": report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
