import itertools
import json
from pathlib import Path

import pytest

from consulting_platform.contracts import RetailRequest
from consulting_platform.retail.evaluate import hide_one
from consulting_platform.retail.fixtures import DEMO_CATALOG, build_demo_model
from consulting_platform.retail.model import RetailModel, split_ids
from consulting_platform.retail.service import RetailModelService, UnknownProductError

REPO = Path(__file__).resolve().parents[2]
ARTIFACT = REPO / "artifacts" / "retail" / "model.json"
CSV = REPO / "data" / "raw" / "retail" / "market_basket_transactions.csv"
RID = "00000000-0000-4000-8000-000000000000"

CATALOG = {sku: f"Product {sku}" for sku in "ABCDEFG"}


def make_model(baskets: list[list[str]]) -> RetailModel:
    transactions = {str(i): b for i, b in enumerate(baskets, start=1)}
    return RetailModel.train(transactions, CATALOG, list(transactions))


def brute_force_counts(baskets: list[list[str]], min_count: int, max_len: int) -> dict[frozenset, int]:
    counts: dict[frozenset, int] = {}
    for basket in baskets:
        for size in range(1, min(len(basket), max_len) + 1):
            for combo in itertools.combinations(sorted(set(basket)), size):
                key = frozenset(combo)
                counts[key] = counts.get(key, 0) + 1
    return {k: v for k, v in counts.items() if v >= min_count}


def test_confidence_and_lift_come_from_exact_counts():
    # A in 10 baskets; B in 5 (4 with A); C in 20; N = 25
    baskets = [["A", "B"]] * 4 + [["A", "C"]] * 6 + [["B", "C"]] + [["C"]] * 13 + [["D", "E"]]
    model = make_model(baskets)
    [b] = [s for s in model.recommend(["A"], 10) if s.product_id == "B"]
    assert (b.evidence_count, b.antecedent_count, b.product_count) == (4, 10, 5)
    assert b.confidence == pytest.approx(4 / 10)
    assert b.support == pytest.approx(4 / 25)
    assert b.lift == pytest.approx((4 / 10) / (5 / 25))
    assert b.score == pytest.approx(b.confidence * (b.lift - 1))
    # A->C: confidence 0.6 but C is in 20/25 baskets, so lift 0.75 < 1 and it is excluded
    assert "C" not in [s.product_id for s in model.recommend(["A"], 10)]


def test_minimum_evidence_count_of_three():
    model = make_model([["A", "B"]] * 2 + [["A"]] * 3 + [["C"]] * 5)
    assert model.recommend(["A"], 5) == []


def test_full_cart_is_required_not_a_subset():
    # A->B is strongly supported, but {A, C} never co-occurs with B.
    baskets = [["A", "B"]] * 5 + [["A", "C", "D"]] * 4 + [["E"]] * 10
    model = make_model(baskets)
    assert [s.product_id for s in model.recommend(["A"], 5)][0] == "B"
    recs = model.recommend(["A", "C"], 5)
    assert [s.product_id for s in recs] == ["D"]
    assert recs[0].antecedent_count == 4
    # order of the cart does not matter for matching
    assert model.recommend(["C", "A"], 5) == recs


def test_never_recommends_cart_items_and_ranking_ties():
    baskets = [["A", "B", "C"]] * 3 + [["F"]] * 9
    model = make_model(baskets)
    recs = model.recommend(["A"], 5)
    assert [s.product_id for s in recs] == ["B", "C"]  # equal score + evidence -> product_id asc
    assert all(s.product_id != "A" for s in recs)


@pytest.mark.skipif(not CSV.exists(), reason="copy the retail CSV with scripts/train_retail.py --import-csv")
def test_fpgrowth_counts_match_brute_force_on_real_data():
    from consulting_platform.retail.model import build_rules, mine_itemset_counts
    from consulting_platform.retail.preprocess import load_transactions

    data = load_transactions(CSV)
    train_ids, _ = split_ids(list(data.transactions))
    baskets = [data.transactions[t] for t in train_ids]
    mined = mine_itemset_counts(baskets, 3, data.max_basket_length)
    assert mined == brute_force_counts(baskets, 3, data.max_basket_length)
    assert all(r.joint_count >= 3 and r.antecedent_count >= r.joint_count for r in build_rules(mined))


def test_split_is_deterministic_and_disjoint():
    ids = [str(i) for i in range(1, 1001)]
    train, test = split_ids(ids)
    assert (train, test) == split_ids(list(reversed(ids)))
    assert len(test) == 200 and not set(train) & set(test)


def test_hide_one_is_deterministic():
    assert hide_one("7", ["A", "B", "C"]) == hide_one("7", ["A", "B", "C"])


def test_artifact_round_trip(tmp_path):
    model = build_demo_model()
    path = tmp_path / "model.json"
    model.save(path)
    loaded = RetailModel.load(path)
    assert loaded.recommend(["FX-PANEL"], 5) == model.recommend(["FX-PANEL"], 5)
    assert json.loads(path.read_text())["n_training"] == 100


@pytest.mark.skipif(not ARTIFACT.exists(), reason="run scripts/train_retail.py first")
def test_trained_artifact_has_no_test_leakage():
    from consulting_platform.retail.preprocess import load_transactions

    model = RetailModel.load(ARTIFACT)
    split = model.metadata["split"]
    assert not set(split["train_ids"]) & set(split["test_ids"])
    data = load_transactions(CSV)
    train_sets = [frozenset(data.transactions[t]) for t in split["train_ids"]]
    assert model.n_training == len(train_sets)
    for rule in model.rules[::25]:
        antecedent = frozenset(rule.antecedent)
        assert rule.antecedent_count == sum(antecedent <= b for b in train_sets)
        assert rule.joint_count == sum(antecedent | {rule.consequent} <= b for b in train_sets)
    for sku, count in model.item_counts.items():
        assert count == sum(sku in b for b in train_sets)


def test_service_statuses_and_validation():
    service = RetailModelService(build_demo_model(), mode="demo")
    assert service.recommend(RetailRequest(cart_items=[]), request_id=RID).status == "empty_cart"

    ok = service.recommend(RetailRequest(cart_items=["FX-PANEL"], top_k=1), request_id=RID)
    assert ok.status == "ok" and len(ok.recommendations) == 1
    rec = ok.recommendations[0]
    assert rec.evidence_count >= 3 and rec.antecedent_count == 40 and rec.lift > 1

    none = service.recommend(RetailRequest(cart_items=["FX-PANEL", "FX-NVR"]), request_id=RID)
    assert none.status == "no_supported_pattern" and none.recommendations == []
    assert none.cart_items == ["FX-PANEL", "FX-NVR"]

    with pytest.raises(UnknownProductError) as excinfo:
        service.recommend(RetailRequest(cart_items=["FX-PANEL", "NOPE"]), request_id=RID)
    assert excinfo.value.details == [("cart_items.1", "Unknown product SKU.")]

    catalog = service.get_catalog(request_id=RID)
    assert [p.product_id for p in catalog.products] == sorted(DEMO_CATALOG)
