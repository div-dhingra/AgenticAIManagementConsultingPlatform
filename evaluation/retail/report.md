# Retail recommender evaluation (retail-v1+3d48b248)

Generated 2026-10-17T22:25:41+00:00 from `data/raw/retail/market_basket_transactions.csv` (sha256 `3d48b2480bee…`).
Training baskets: 800. Held-out queries: 200 (skipped single-item baskets: 0). k = 5.

| Metric | Full-cart rules | Popularity baseline |
|---|---|---|
| Hit rate @k (hidden item recommended) | 4.0% (8/200) | 17.0% (34/200) |
| Coverage (queries with any suggestion) | 27.0% | 100.0% |
| Precision @k | 3.2% (micro: hits / returned items) | 3.4% (hits / (queries × k)) |
| Hit rate on covered queries | 14.8% | n/a |
| Catalog coverage | 97.5% | 17.5% |

Model results by query cart size:

| Items in query cart | Queries | Coverage | Hit rate @k |
|---|---|---|---|
| 1 | 49 | 100.0% | 16.3% |
| 2 | 44 | 11.4% | 0.0% |
| 3 | 54 | 0.0% | 0.0% |
| 4 | 53 | 0.0% | 0.0% |

In-process latency per query: mean 0.002 ms, p95 0.009 ms, max 0.014 ms.

Notes:
- For each held-out basket with >=2 items, hide one item chosen by random.Random('42:<transaction_id>') and query with the rest.
- None. Thresholds are fixed team defaults; held-out baskets were not used to choose them.
- The popularity baseline is for comparison only and is never used to fill UI recommendations.
- Confidence and lift are co-purchase statistics, not evidence of compatibility or business impact.

Demo carts taken from this model:
- Supported: `['5800CO']` → ['WS4916', 'XRN-2010', 'HS2016NK', '60-652-95R', '60-746-01-95R']
- Unsupported combination: `['0407-001', '0806-001']`
