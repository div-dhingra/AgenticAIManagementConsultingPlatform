"""Full-cart association-rule model: training, prediction and JSON persistence.

Training mines frequent itemsets with mlxtend FP-growth on the training split only,
then recomputes every count exactly. A suggestion for cart C is a rule C -> p whose
antecedent equals the whole cart as a set (never a subset or the last item).
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL_VERSION = "retail-v1"
ARTIFACT_FORMAT = 1

# Team-default policy from shared contract v1.0 (not tuned on any data).
MIN_EVIDENCE_COUNT = 3
MIN_CONFIDENCE = 0.10
MIN_LIFT = 1.0  # strict: lift must be > 1.0
SPLIT_SEED = 42
TEST_FRACTION = 0.2


@dataclass(frozen=True)
class Rule:
    antecedent: tuple[str, ...]  # sorted SKUs
    consequent: str
    joint_count: int  # count(antecedent ∪ {consequent}) in training baskets
    antecedent_count: int  # count(antecedent) in training baskets


@dataclass(frozen=True)
class Suggestion:
    product_id: str
    score: float
    support: float
    confidence: float
    lift: float
    evidence_count: int
    antecedent_count: int
    product_count: int


def split_ids(ids: list[str], seed: int = SPLIT_SEED, test_fraction: float = TEST_FRACTION) -> tuple[list[str], list[str]]:
    """Deterministic train/test split of transaction IDs."""
    shuffled = sorted(ids, key=int)
    random.Random(seed).shuffle(shuffled)
    n_test = round(len(shuffled) * test_fraction)
    test = sorted(shuffled[:n_test], key=int)
    train = sorted(shuffled[n_test:], key=int)
    return train, test


def mine_itemset_counts(baskets: list[list[str]], min_count: int, max_len: int) -> dict[frozenset[str], int]:
    """Frequent itemsets via mlxtend FP-growth, with exact integer counts."""
    import pandas as pd
    from mlxtend.frequent_patterns import fpgrowth
    from mlxtend.preprocessing import TransactionEncoder

    n = len(baskets)
    encoder = TransactionEncoder()
    frame = pd.DataFrame(encoder.fit(baskets).transform(baskets), columns=encoder.columns_)
    # Mine slightly below the threshold so float rounding in support can never drop a
    # count==min_count itemset; the exact recount below enforces the real cut-off.
    itemsets = fpgrowth(frame, min_support=(min_count - 0.5) / n, use_colnames=True, max_len=max_len)

    basket_sets = [frozenset(b) for b in baskets]
    counts: dict[frozenset[str], int] = {}
    for items in itemsets["itemsets"]:
        key = frozenset(items)
        count = sum(1 for basket in basket_sets if key <= basket)
        if count >= min_count:
            counts[key] = count
    return counts


def build_rules(itemset_counts: dict[frozenset[str], int]) -> list[Rule]:
    """Every single-consequent rule derivable from the frequent itemsets."""
    rules = []
    for itemset, joint in itemset_counts.items():
        if len(itemset) < 2:
            continue
        for consequent in itemset:
            antecedent = itemset - {consequent}
            rules.append(Rule(tuple(sorted(antecedent)), consequent, joint, itemset_counts[antecedent]))
    rules.sort(key=lambda r: (len(r.antecedent), r.antecedent, r.consequent))
    return rules


class RetailModel:
    def __init__(
        self,
        *,
        catalog: dict[str, str],
        n_training: int,
        item_counts: dict[str, int],
        rules: list[Rule],
        model_version: str = MODEL_VERSION,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if n_training <= 0:
            raise ValueError("n_training must be positive")
        self.catalog = dict(sorted(catalog.items()))
        self.n_training = n_training
        self.item_counts = item_counts
        self.rules = rules
        self.model_version = model_version
        self.metadata = metadata or {}
        self._index: dict[frozenset[str], list[Rule]] = {}
        for rule in rules:
            if rule.consequent not in self.catalog or not set(rule.antecedent) <= set(self.catalog):
                raise ValueError(f"rule references an unknown SKU: {rule}")
            self._index.setdefault(frozenset(rule.antecedent), []).append(rule)

    # ------------------------------------------------------------------ training

    @classmethod
    def train(
        cls,
        transactions: dict[str, list[str]],
        catalog: dict[str, str],
        train_ids: list[str],
        *,
        min_count: int = MIN_EVIDENCE_COUNT,
        max_len: int | None = None,
        model_version: str = MODEL_VERSION,
        metadata: dict[str, Any] | None = None,
    ) -> "RetailModel":
        baskets = [transactions[tid] for tid in train_ids]
        max_len = max_len or max(len(b) for b in baskets)
        counts = mine_itemset_counts(baskets, min_count, max_len)
        item_counts = {sku: 0 for sku in catalog}
        for basket in baskets:
            for sku in basket:
                item_counts[sku] += 1
        return cls(
            catalog=catalog,
            n_training=len(baskets),
            item_counts=item_counts,
            rules=build_rules(counts),
            model_version=model_version,
            metadata={**(metadata or {}), "max_itemset_length": max_len, "min_count": min_count},
        )

    # ---------------------------------------------------------------- prediction

    def recommend(self, cart: list[str], top_k: int) -> list[Suggestion]:
        """Ranked suggestions whose rule antecedent equals the full cart."""
        cart_set = frozenset(cart)
        if not cart_set:
            return []
        suggestions = []
        for rule in self._index.get(cart_set, []):
            if rule.consequent in cart_set or rule.joint_count < MIN_EVIDENCE_COUNT:
                continue
            product_count = self.item_counts[rule.consequent]
            confidence = rule.joint_count / rule.antecedent_count
            lift = confidence / (product_count / self.n_training)
            if confidence < MIN_CONFIDENCE or not lift > MIN_LIFT:
                continue
            suggestions.append(
                Suggestion(
                    product_id=rule.consequent,
                    score=confidence * (lift - 1.0),
                    support=rule.joint_count / self.n_training,
                    confidence=confidence,
                    lift=lift,
                    evidence_count=rule.joint_count,
                    antecedent_count=rule.antecedent_count,
                    product_count=product_count,
                )
            )
        suggestions.sort(key=lambda s: (-s.score, -s.evidence_count, s.product_id))
        return suggestions[:top_k]

    def popularity_ranking(self) -> list[str]:
        """Training-only popularity order, for the evaluation baseline only."""
        return sorted(self.item_counts, key=lambda sku: (-self.item_counts[sku], sku))

    # --------------------------------------------------------------- persistence

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_format": ARTIFACT_FORMAT,
            "model_version": self.model_version,
            "n_training": self.n_training,
            "catalog": self.catalog,
            "item_counts": self.item_counts,
            "rules": [[list(r.antecedent), r.consequent, r.joint_count, r.antecedent_count] for r in self.rules],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetailModel":
        if data.get("artifact_format") != ARTIFACT_FORMAT:
            raise ValueError(f"unsupported retail artifact format {data.get('artifact_format')!r}")
        rules = [Rule(tuple(a), c, int(j), int(n)) for a, c, j, n in data["rules"]]
        return cls(
            catalog=data["catalog"],
            n_training=int(data["n_training"]),
            item_counts={k: int(v) for k, v in data["item_counts"].items()},
            rules=rules,
            model_version=data["model_version"],
            metadata=data.get("metadata", {}),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=1, sort_keys=False), encoding="utf-8")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "RetailModel":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
