"""Parse the market-basket CSV into validated baskets and a SKU catalog."""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

ITEM_PATTERN = re.compile(r"^(?P<name>.*?\S)\s*\(SKU:\s*(?P<sku>[^()]*?)\s*\)\s*$")
EXPECTED_COLUMNS = ["transaction_id", "item_1", "item_2", "item_3", "item_4", "item_5"]


class CatalogConflictError(ValueError):
    """The same SKU appears with two different product names."""


@dataclass
class PreprocessResult:
    transactions: dict[str, list[str]]  # transaction_id -> ordered, deduplicated SKUs
    catalog: dict[str, str]  # SKU -> product name
    issues: list[str] = field(default_factory=list)
    duplicate_items_removed: int = 0

    @property
    def max_basket_length(self) -> int:
        return max((len(items) for items in self.transactions.values()), default=0)


def parse_item(cell: str) -> tuple[str, str] | None:
    """Return (sku, name) for 'Name (SKU: X)', or None if the cell is malformed."""
    match = ITEM_PATTERN.match(cell.strip())
    if not match or not match.group("sku"):
        return None
    return match.group("sku"), match.group("name").strip()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_transactions(path: Path) -> PreprocessResult:
    """Read the CSV. Malformed rows are reported in `issues` and skipped; SKU name
    conflicts raise, because they would silently create two product identities."""
    transactions: dict[str, list[str]] = {}
    catalog: dict[str, str] = {}
    issues: list[str] = []
    duplicates = 0

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None or [h.strip() for h in header] != EXPECTED_COLUMNS:
            raise ValueError(f"unexpected CSV header {header!r}; expected {EXPECTED_COLUMNS}")

        for line_no, row in enumerate(reader, start=2):
            if not any(cell.strip() for cell in row):
                continue
            if len(row) > len(EXPECTED_COLUMNS):
                issues.append(f"line {line_no}: {len(row)} columns, expected {len(EXPECTED_COLUMNS)}; row skipped")
                continue
            tid = row[0].strip()
            if not tid.isdigit():
                issues.append(f"line {line_no}: invalid transaction_id {tid!r}; row skipped")
                continue
            if tid in transactions:
                issues.append(f"line {line_no}: duplicate transaction_id {tid}; row skipped")
                continue

            basket: list[str] = []
            bad_cell = False
            for cell in row[1:]:
                if not cell.strip():
                    continue
                parsed = parse_item(cell)
                if parsed is None:
                    issues.append(f"line {line_no}: unparseable item {cell.strip()!r}; row skipped")
                    bad_cell = True
                    break
                sku, name = parsed
                known = catalog.get(sku)
                if known is not None and known != name:
                    raise CatalogConflictError(f"SKU {sku} maps to both {known!r} and {name!r} (line {line_no})")
                catalog[sku] = name
                if sku in basket:
                    duplicates += 1
                    continue
                basket.append(sku)
            if bad_cell:
                continue
            if not basket:
                issues.append(f"line {line_no}: transaction {tid} has no items; row skipped")
                continue
            transactions[tid] = basket

    return PreprocessResult(transactions, dict(sorted(catalog.items())), issues, duplicates)
