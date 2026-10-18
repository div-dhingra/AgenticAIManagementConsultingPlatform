from pathlib import Path

import pytest

from consulting_platform.retail.preprocess import (
    CatalogConflictError,
    load_transactions,
    parse_item,
)

HEADER = "transaction_id,item_1,item_2,item_3,item_4,item_5\n"


def write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "baskets.csv"
    path.write_text(HEADER + body, encoding="utf-8")
    return path


def test_parse_item_splits_name_and_sku():
    assert parse_item("Bosch B5512 Control Panel (SKU: B5512)") == ("B5512", "Bosch B5512 Control Panel")
    assert parse_item(" Dahua 8-Channel NVR (SKU: NVR4208-8P-4KS2) ") == ("NVR4208-8P-4KS2", "Dahua 8-Channel NVR")
    assert parse_item("No sku here") is None
    assert parse_item("Empty (SKU: )") is None


def test_blank_cells_ignored_and_duplicates_removed(tmp_path):
    data = load_transactions(write(tmp_path, "1,A Thing (SKU: A),B Thing (SKU: B),A Thing (SKU: A),,\n"))
    assert data.transactions == {"1": ["A", "B"]}
    assert data.catalog == {"A": "A Thing", "B": "B Thing"}
    assert data.duplicate_items_removed == 1
    assert data.issues == []


def test_malformed_rows_are_reported_and_skipped(tmp_path):
    body = (
        "1,A Thing (SKU: A),,,,\n"
        "x2,A Thing (SKU: A),,,,\n"  # bad id
        "1,B Thing (SKU: B),,,,\n"  # duplicate id
        "3,garbage,,,,\n"  # unparseable item
        "4,,,,,\n"  # empty basket
    )
    data = load_transactions(write(tmp_path, body))
    assert list(data.transactions) == ["1"]
    assert len(data.issues) == 4


def test_conflicting_names_for_one_sku_raise(tmp_path):
    with pytest.raises(CatalogConflictError):
        load_transactions(write(tmp_path, "1,A Thing (SKU: A),,,,\n2,Other Name (SKU: A),,,,\n"))


def test_wrong_header_raises(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("id,a,b\n1,x,y\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_transactions(path)
