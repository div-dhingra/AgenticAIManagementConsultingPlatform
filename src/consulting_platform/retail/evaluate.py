"""Leave-one-out evaluation of the retail model on the held-out split."""

from __future__ import annotations

import random
import statistics
import time
from typing import Any

from .model import MIN_CONFIDENCE, MIN_EVIDENCE_COUNT, MIN_LIFT, RetailModel

EVAL_SEED = 42


def hide_one(tid: str, basket: list[str], seed: int = EVAL_SEED) -> tuple[list[str], str]:
    """Deterministically hide one item; return (query cart in original order, hidden item)."""
    index = random.Random(f"{seed}:{tid}").randrange(len(basket))
    return basket[:index] + basket[index + 1 :], basket[index]


def _rate(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def evaluate(model: RetailModel, transactions: dict[str, list[str]], test_ids: list[str], k: int = 5) -> dict[str, Any]:
    popular = model.popularity_ranking()
    queries = [(tid, *hide_one(tid, transactions[tid])) for tid in test_ids if len(transactions[tid]) >= 2]
    skipped = len(test_ids) - len(queries)

    hits = returned = covered = 0
    per_query_precision: list[float] = []
    recommended_products: set[str] = set()
    latencies_ms: list[float] = []
    by_size: dict[int, dict[str, int]] = {}
    base_hits = 0
    base_products: set[str] = set()

    for _, cart, hidden in queries:
        start = time.perf_counter()
        suggestions = model.recommend(cart, k)
        latencies_ms.append((time.perf_counter() - start) * 1000)

        ids = [s.product_id for s in suggestions]
        hit = int(hidden in ids)
        hits += hit
        returned += len(ids)
        recommended_products.update(ids)
        if ids:
            covered += 1
            per_query_precision.append(hit / len(ids))
        bucket = by_size.setdefault(len(cart), {"queries": 0, "covered": 0, "hits": 0})
        bucket["queries"] += 1
        bucket["covered"] += int(bool(ids))
        bucket["hits"] += hit

        baseline = [sku for sku in popular if sku not in cart][:k]
        base_hits += int(hidden in baseline)
        base_products.update(baseline)

    n = len(queries)
    catalog_size = len(model.catalog)
    latencies_sorted = sorted(latencies_ms)
    return {
        "config": {
            "k": k,
            "protocol": "For each held-out basket with >=2 items, hide one item chosen by "
            f"random.Random('{EVAL_SEED}:<transaction_id>') and query with the rest.",
            "thresholds": {"min_evidence_count": MIN_EVIDENCE_COUNT, "min_confidence": MIN_CONFIDENCE, "min_lift_exclusive": MIN_LIFT},
            "tuning": "None. Thresholds are fixed team defaults; held-out baskets were not used to choose them.",
        },
        "n_test_baskets": len(test_ids),
        "n_queries": n,
        "skipped_single_item_baskets": skipped,
        "model": {
            "hit_rate_at_k": _rate(hits, n),
            "hits": hits,
            "coverage": _rate(covered, n),
            "covered_queries": covered,
            "hit_rate_on_covered_queries": _rate(hits, covered),
            "precision_at_k_micro": _rate(hits, returned),
            "precision_at_k_micro_definition": "total hits / total returned items; queries with an empty list add 0 to both",
            "precision_at_k_macro_covered": statistics.fmean(per_query_precision) if per_query_precision else None,
            "precision_at_k_macro_covered_definition": "mean of hits/len(list) over queries with a non-empty list",
            "mean_list_length": _rate(returned, n),
            "catalog_coverage": len(recommended_products) / catalog_size,
            "by_query_cart_size": {
                str(size): {**b, "coverage": _rate(b["covered"], b["queries"]), "hit_rate_at_k": _rate(b["hits"], b["queries"])}
                for size, b in sorted(by_size.items())
            },
        },
        "popularity_baseline": {
            "description": "Top-k most frequent training items not already in the cart (evaluation only; never used in the UI).",
            "hit_rate_at_k": _rate(base_hits, n),
            "hits": base_hits,
            "coverage": 1.0 if n else None,
            "precision_at_k": _rate(base_hits, n * k),
            "precision_at_k_definition": "hits / (queries * k); the baseline always returns k items",
            "catalog_coverage": len(base_products) / catalog_size,
        },
        "latency_ms_in_process": {
            "mean": statistics.fmean(latencies_ms) if latencies_ms else None,
            "p50": latencies_sorted[len(latencies_sorted) // 2] if latencies_sorted else None,
            "p95": latencies_sorted[int(len(latencies_sorted) * 0.95) - 1] if latencies_sorted else None,
            "max": latencies_sorted[-1] if latencies_sorted else None,
            "note": "model.recommend() only, excluding HTTP and serialization",
        },
    }


def find_demo_cases(model: RetailModel) -> dict[str, Any]:
    """Real carts from the trained model: the strongest supported single-item cart,
    and a two-item cart whose items each have suggestions but no full-cart pattern."""
    supported = None
    for sku in model.catalog:
        suggestions = model.recommend([sku], 5)
        if suggestions and (supported is None or suggestions[0].score > supported[1]):
            supported = (sku, suggestions[0].score, [s.product_id for s in suggestions])

    unsupported = None
    skus = list(model.catalog)
    for i, a in enumerate(skus):
        if not model.recommend([a], 5):
            continue
        for b in skus[i + 1 :]:
            if model.recommend([b], 5) and not model.recommend([a, b], 5):
                unsupported = [a, b]
                break
        if unsupported:
            break

    return {
        "supported_cart": [supported[0]] if supported else None,
        "supported_cart_top_suggestions": supported[2] if supported else None,
        "unsupported_cart": unsupported,
    }


def render_markdown(report: dict[str, Any]) -> str:
    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.1%}"

    m, b, lat = report["model"], report["popularity_baseline"], report["latency_ms_in_process"]
    lines = [
        f"# Retail recommender evaluation ({report['model_version']})",
        "",
        f"Generated {report['generated_at']} from `{report['input']['path']}` (sha256 `{report['input']['sha256'][:12]}…`).",
        f"Training baskets: {report['n_training']}. Held-out queries: {report['n_queries']} "
        f"(skipped single-item baskets: {report['skipped_single_item_baskets']}). k = {report['config']['k']}.",
        "",
        "| Metric | Full-cart rules | Popularity baseline |",
        "|---|---|---|",
        f"| Hit rate @k (hidden item recommended) | {pct(m['hit_rate_at_k'])} ({m['hits']}/{report['n_queries']}) | {pct(b['hit_rate_at_k'])} ({b['hits']}/{report['n_queries']}) |",
        f"| Coverage (queries with any suggestion) | {pct(m['coverage'])} | {pct(b['coverage'])} |",
        f"| Precision @k | {pct(m['precision_at_k_micro'])} (micro: hits / returned items) | {pct(b['precision_at_k'])} (hits / (queries × k)) |",
        f"| Hit rate on covered queries | {pct(m['hit_rate_on_covered_queries'])} | n/a |",
        f"| Catalog coverage | {pct(m['catalog_coverage'])} | {pct(b['catalog_coverage'])} |",
        "",
        "Model results by query cart size:",
        "",
        "| Items in query cart | Queries | Coverage | Hit rate @k |",
        "|---|---|---|---|",
    ]
    for size, row in m["by_query_cart_size"].items():
        lines.append(f"| {size} | {row['queries']} | {pct(row['coverage'])} | {pct(row['hit_rate_at_k'])} |")
    lines += [
        "",
        f"In-process latency per query: mean {lat['mean']:.3f} ms, p95 {lat['p95']:.3f} ms, max {lat['max']:.3f} ms.",
        "",
        "Notes:",
        f"- {report['config']['protocol']}",
        f"- {report['config']['tuning']}",
        "- The popularity baseline is for comparison only and is never used to fill UI recommendations.",
        "- Confidence and lift are co-purchase statistics, not evidence of compatibility or business impact.",
    ]
    if report.get("demo_cases"):
        d = report["demo_cases"]
        lines += ["", "Demo carts taken from this model:", f"- Supported: `{d['supported_cart']}` → {d['supported_cart_top_suggestions']}",
                  f"- Unsupported combination: `{d['unsupported_cart']}`"]
    return "\n".join(lines) + "\n"
