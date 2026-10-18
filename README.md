# Agentic AI Management Consulting Platform — runbook

One Python app with two tabs:

- **Consulting**: the Streamlit UI calls `POST /consult`. The API forwards the request to Person A's LangGraph workflow, which uses Person B's knowledge index.
- **Retail Recommendations**: the UI calls `GET /catalog` and `POST /recommend`. These are backed by a co-purchase model trained on the supplied basket CSV.

Ownership follows shared contract v1.0:
- **Person A:** `contracts.py`, `config.py`, `agents/`, `pyproject.toml`
- **Person B:** `knowledge/`, `scripts/ingest.py`
- **Person C:** `retail/`, `api/`, `ui/`, `scripts/train_retail.py`, and this README

Agent/contract details live in [docs/INTEGRATION.md](docs/INTEGRATION.md) and [docs/AGENTS.md](docs/AGENTS.md).

## 1. Setup (Python 3.11)

```bash
conda create -n cmpe256 python=3.11 -y      # or python3.11 -m venv .venv
conda activate cmpe256
pip install -e ".[dev,app,retail]"          # add ,knowledge for Person B's ingestion/index
cp .env.example .env                         # then fill in the provider key locally; never commit it
```

Live consulting also needs:
- the provider credential named in `.env.example`, in your local `.env`
- Person B's knowledge index
- the `sentence-transformers/all-MiniLM-L6-v2` embedding model, which downloads once and needs network access

## 2. Data

Large data files are kept out of Git, so copy them in on each machine.

| What | Where in the repo | Who |
|---|---|---|
| `CMPE256_Hackathon_market_basket_analysis_Release.csv` | `data/raw/retail/market_basket_transactions.csv` (git-ignored) | C (`--import-csv`) |
| SEC 10-Ks, SOFI PDFs, Cisco/Agilent/GCP sources | `data/…` per B's `data/manifest.json` | B (`python scripts/ingest.py`) |

## 3. Train the retail model

```bash
python scripts/train_retail.py --import-csv /path/to/CMPE256_Hackathon_market_basket_analysis_Release.csv
python scripts/train_retail.py          # later runs, once the CSV is in data/raw/retail/
```

This writes:
- `artifacts/retail/model.json` (plain JSON, no pickle), which records:
  - model version, input SHA-256, training size, train/test IDs
  - rule settings, every rule's counts, the catalog, timestamp and evaluation config
- `evaluation/retail/report.json` and `report.md`

**Rebuild policy:** retrain whenever the CSV changes, since the model version includes the CSV checksum. Nothing retrains at request time. Restart the API after retraining.

### Model

Rules are mined with mlxtend FP-growth on the training split only: a seed-42 80/20 split, giving 800 training baskets. Every count is then recomputed exactly. A suggestion `p` for cart `C` needs a rule whose antecedent **equals the whole cart** as a set, plus:
- `count(C ∪ {p}) ≥ 3`
- confidence ≥ 0.10
- lift > 1

Ranking is by `confidence × (lift − 1)`, then evidence count, then SKU. If nothing passes, the API returns `no_supported_pattern` and an empty list. It never falls back to popular items.

### Measured results (seed 42, k = 5, 200 held-out baskets, one item hidden in each)

Each result is the share of held-out baskets where the hidden item appeared in the top 5 suggestions.

| | Full-cart rules | Popularity baseline (evaluation only) |
|---|---|---|
| Hit rate @5 | 4.0% (8/200) | 17.0% (34/200) |
| Coverage (any suggestion) | 27.0% | 100% |
| Precision @5 | 3.2% (hits / returned items) | 3.4% (hits / (queries × 5)) |
| Catalog coverage | 97.5% | 17.5% |
| Hit rate @5, 1-item carts | 16.3% (coverage 100%) | — |
| Coverage, 2-item carts | 11.4% | — |
| Coverage, 3–4-item carts | 0% | — |

Latency is about 0.01 ms per call in-process.

**Why the numbers are low:**
- The supplied baskets are close to random: 778 of the 780 possible product pairs co-occur, and pair lift rarely goes above 2.
- With 800 training baskets, almost no 3-item combination occurs 3 or more times, so carts of 3+ items correctly return "no supported pattern".

These are honest results under the team's fixed policy. The thresholds were not tuned on test data. The full table is in [evaluation/retail/report.md](evaluation/retail/report.md).

## 4. Run

```bash
# Terminal 1 — API on http://127.0.0.1:8000  (APP_MODE=live by default)
uvicorn consulting_platform.api.app:create_app --factory --host 127.0.0.1 --port 8000

# Terminal 2 — UI on http://localhost:8501
API_BASE_URL=http://127.0.0.1:8000 streamlit run src/consulting_platform/ui/app.py
```

| Variable | Default | Meaning |
|---|---|---|
| `APP_MODE` | `live` | `demo` runs Person A's agent graphs with their synthetic evidence and deterministic fake LLM, plus a synthetic retail fixture. The API and UI disclose it on every response. Live mode never silently falls back to demo. |
| `ARTIFACT_DIR` | `artifacts` | Where `retail/model.json` is read from. |
| `DATA_DIR` | `data` | Where the training CSV is read from. |
| `API_BASE_URL` | `http://127.0.0.1:8000` | Used by the UI. |
| `CONSULT_TIMEOUT_SECONDS` | `90` | Outer `/consult` deadline. The UI's HTTP timeout is 100 s. |

Settings are read by Person A's `consulting_platform.config.Settings` (environment variables override `.env`).

**Subsystems start independently.** A missing retail artifact only makes `/catalog` and `/recommend` return 503. A missing or failing agent workflow only makes `/consult` return 503. `/health` reports each subsystem and never calls a paid model.

**Integration point:** at startup, `src/consulting_platform/api/deps.py` calls Person B's no-argument factory `consulting_platform.knowledge.service:create_knowledge_service()`. It passes the result to Person A's `create_consulting_service(knowledge, settings=...)`.
- If B's factory has another name, change `KNOWLEDGE_FACTORY` in that file.
- If B's service or the provider key is missing, `/consult` returns 503 and `/health` explains why.

Before B's index is ready, you can explicitly use selected real passages with live
inference through the same UI:

```bash
APP_MODE=live CONSULT_EVIDENCE_FILE=fixtures/consulting/controlled_real_passages.json \
  .venv/bin/uvicorn consulting_platform.api.app:create_app --factory --port 8000
```

Restart the API, refresh the UI and its status, choose a controlled example in
Consulting, and click **Ask**. This uses real model API calls and always discloses
the limited evidence. Retail is unchanged. Unset/empty `CONSULT_EVIDENCE_FILE` to
return to B's factory; do not combine it with demo mode. See
[controlled tests](docs/CONTROLLED_TESTS.md) for details.

## 5. API summary

| Route | Success | Notable errors |
|---|---|---|
| `GET /health` | 200 `HealthResponse` (`ok`/`degraded`) | — |
| `POST /consult` | 200 `ConsultResponse` | 422, 503, 504 timeout, 502 provider error |
| `GET /catalog` | 200 `CatalogResponse`, sorted by SKU | 503 if no retail artifact |
| `POST /recommend` | 200 `RetailResponse` (`ok`/`empty_cart`/`no_supported_pattern`) | 422 for unknown, repeated or extra fields; 503 |

- Every error uses the shared `ErrorResponse` envelope, including FastAPI's own 404, 405 and 422 responses.
- Every response carries an `X-Request-ID` header that matches `request_id`.
- Stack traces and provider error text are never returned.

## 6. Tests and evaluation

```bash
pytest -q                               # retail, API and UI smoke tests; no paid calls
python scripts/train_retail.py          # retrains and rewrites evaluation/retail/report.*
```

**Manual UI checklist**
1. The sidebar shows mode, per-service status and per-domain coverage.
2. Retail:
   - add `5800CO` and see 5 ranked suggestions
   - add the top suggestion and see suggestions for the 2-item cart
   - add a third item and see "No suggestion for this exact cart"
   - remove an item and see the suggestions refresh
3. In demo mode, the "Demo fixture question" and the "Cross-domain example" both return answered responses with numbered citations and a trace, and a demo banner is visible.
4. With the API stopped, both tabs show "Cannot reach the API". When the API is up but returns an error, the error shows its request ID.

## 7. Demo outline (about 5 minutes)

1. Open with the sidebar health panel to show what is indexed per domain and what isn't.
2. Ask the cross-domain consulting preset. Walk through status, routed domains, ranked recommendations with prerequisites, cited evidence with pages, uncertainty and the trace.
3. Ask a missing-context question, e.g. "Which replacement parts should we order for our failing router?" with no model given. Show the `needs_clarification` result.
4. Retail, supported case: cart `5800CO` → WS4916 Smoke Detector, from 11 of 61 training baskets, lift 2.09.
5. Retail, unsupported case: cart `0407-001` + `0806-001` → no supported pattern. Explain abstention.
6. Close with the evaluation table and an honest read of the data's low signal.

## Known limitations

- Retail statistics measure co-purchase only. They do not show compatibility or business impact.
- Most carts of 2+ items get no suggestion because the data is sparse and close to random.
- Canvas states no explicit slide, video or repo format. Treat the demo outline as a useful asset, not a mandated deliverable.

### Consulting demo and architecture

- [Exact demo questions, expected decisions, negative cases and evaluation](docs/CONSULTING_TEST_GUIDE.md)
- [Individual agents, manager orchestration, project architecture, RAG and answer decisions](docs/ARCHITECTURE.md)

The full live UI includes examples for Technology, Clinical, SEC and SOFI, plus
cross-domain and router-troubleshooting presets. A router incident without diagnostic
evidence should request clarification and avoid ordering parts; an `Answered` badge
is not the expected outcome for every safety/abstention case.
