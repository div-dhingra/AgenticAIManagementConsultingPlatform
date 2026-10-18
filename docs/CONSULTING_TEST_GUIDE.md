# Consulting agent tests and demo

The assignment requires useful decisions, grounded citations, ranking, orchestration,
uncertainty and measurable evaluation. An `Answered` badge alone does not establish
any of these. Read the conclusion and verify that it answers the question asked.
The sidebar reports service/corpus availability, not accuracy or full coverage.

## Run the application

From the repository root, use the existing `.venv`. If setting up another machine:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,app,knowledge,retail]'
```

In `.env`, set `APP_MODE=live`, `LLM_PROVIDER=cohere`, your `COHERE_API_KEY`, and
`COHERE_MODEL=command-a-03-2025`. Leave role overrides blank to use the same model.
Unset `CONSULT_EVIDENCE_FILE` for the full local knowledge service. Do not print or
commit the key. Restart the backend after changing environment settings. Existing
exported terminal variables can override `.env`.

The current index already exists. Rebuild it only after changing documents/chunks:

```bash
.venv/bin/python scripts/build_index.py
```

Terminal 1:

```bash
.venv/bin/uvicorn consulting_platform.api.app:create_app --factory --port 8000
```

Terminal 2:

```bash
.venv/bin/streamlit run src/consulting_platform/ui/app.py
```

Open `http://localhost:8501`. Refresh status. Click an example button, then **Ask**.
Each preset resets the old company, model, year and requirements. Do not use demo
mode to judge real-model/source quality; demo mode uses synthetic deterministic
fixtures. Do not start a second backend on the same port.

## Main demo cases — exact questions

The following six questions have buttons in full live mode. Their text and context
are defined in `src/consulting_platform/examples.py`. Context not listed is empty.

### 1. Compare router ports — Technology and deterministic recommendation

> Compare Cisco C8300-2N2S-4T2X and C8300-2N2S-6T for at least 5 1G Ethernet ports.

Context: company `Cisco`; requirements `At least 5 1G Ethernet ports`.

Expected: **Answered**, Technology. The 4T2X has four 1G ports and fails this
requirement; the 6T has six and meets it. Recommend only the 6T for this comparison.
The conclusion must not imply overall hospital deployment suitability or treat
10G ports as extra 1G ports. Verify product identity, table/prose counts and page.

### 2. Cisco risks (2025) — SEC

> Summarize demand, inventory, and purchase-commitment risks disclosed in Cisco fiscal 2025 10-K.

Context: company `Cisco`; report years `2025`.

Expected: **Answered**, SEC. Explain all three requested risk topics in readable
language with Cisco FY2025 citations. An inventory growth figure alone does not
answer demand risk. No wrong-year filing, standalone liability invented from a
combined measure, investment recommendation or achieved return.

### 3. SOFI priorities (2024) — Corporate responsibility

> Explain SOFI 2024 priorities for financing food security and nutrition and improving data for evidence-based decisions.

Context: report years `2024`; company empty.

Expected: **Answered**, Corporate responsibility. Explain both financing priorities
and data/measurement needs using SOFI 2024. No Cisco company filter on SOFI, no
claim that a router investment has already improved nutrition outcomes.

### 4. Agilent laboratory options — Clinical

> According to the indexed Agilent product pages, what do Seahorse XF analyzers measure, and what do digital pathology slide scanners do? Distinguish these laboratory workflows; do not infer clinical approval or patient suitability.

Context: company `Agilent`.

Expected: **Answered**, Clinical, when both product-page passages validate.
Distinguish live-cell metabolic/bioenergetic measurements from digitizing pathology
slides. Cite the respective product pages. This demonstrates grounded laboratory
assistance; the current product pages do not establish patient suitability,
repair compatibility, or every instrument's quantitative capacity.

### 5. Router troubleshooting — missing diagnostic evidence

> Our Cisco enterprise router is experiencing intermittent connectivity failures. Which replacement parts should we order, and what actions should we take to prevent future outages? Our router is a Cisco Catalyst 8300.

Context: company `Cisco`; model `Cisco Catalyst 8300`.

Expected: **Needs clarification**, Technology. Start with **do not order parts yet**.
Ask for the exact chassis SKU and timestamped fault/diagnostic observations. Give
a proposed investigation and recurrence measures. A PSU substitution or SRST
feature is not a diagnosis. Do not display a confirmed replacement part or label
unrelated specifications as an answered repair question. This is the assignment's
example with incomplete customer facts; clarification is the correct behavior.

### 6. Cross-domain example — Technology + SEC + CSR

> We operate a hospital network with diabetes research labs. Compare Cisco C8300-2N2S-4T2X and C8300-2N2S-6T against our requirement for at least five 1G Ethernet ports per site. Assess only this documented port requirement; overall deployment suitability remains unverified. Summarize demand, inventory, and purchase-commitment risks disclosed in Cisco's fiscal 2025 10-K. Also summarize risks disclosed in Agilent's fiscal 2025 10-K; if filing evidence is unavailable, explicitly leave that assessment unresolved. Our proposed initiative is to connect sites for sharing nutrition-program data. Explain relevant food-security priorities from SOFI 2024, distinguishing documented priorities from hypothetical ways connectivity could support the initiative.

Context: requirements `At least 5 1G Ethernet ports per site`; report years
`2024, 2025`; company and model empty; max recommendations `3`.

Expected with the current supporting corpus: **Answered**, Technology, SEC and CSR.
The answer should read like a consulting recommendation with three clear parts:

1. Choose the 6T for the stated port requirement; the 4T2X fails it. Limit this
   conclusion to port count.
2. Explain the requested Cisco risks and Agilent filing risks separately, with
   the correct company and FY2025 citations.
3. Explain documented SOFI financing/data priorities. Propose assessing the
   nutrition-data workflow before spending, describe a small feasibility test,
   and measure data completeness/reporting delay against a baseline. Identify
   connectivity benefits as hypotheses, not established food-security outcomes.

Inspect Operational trace: routing chooses three domains; each retrieves and
validates evidence; ranking runs for the technical comparison; manager reviews,
synthesizes and checks completeness. Clinical is not required merely because the
customer runs labs: this question does not request a clinical instrument decision.

A natural answer can use short labeled sections and bullets. It need not be one
unbroken paragraph or imitate the sample screenshot's unrelated healthcare text.
Sources, detailed findings and diagnostics should support the main answer, not
substitute for it. A real missing sub-answer should still produce Partial.

## Additional validation cases

Start from an empty question/context (or load a preset and clear all unrelated
context). These are checks, not all necessary to show in the short demo video.

| Check | Exact input / change | Correct behavior |
|---|---|---|
| No model supplied | `Our Cisco router has intermittent failures. Which replacement parts should we order?` | Needs clarification for exact SKU and diagnostics; no component order. |
| Exact SKU is not a diagnosis | Troubleshooting question above; change model to `C8300-2N2S-6T` | Still Needs clarification for fault observations; no PSU order. Do not re-ask the supplied SKU. |
| User supplies observations | `Our Cisco C8300-2N2S-6T router has intermittent failures. Diagnostic logs show interface errors after each drop. Which replacement parts should we order?` | Do not re-ask for logs already described. No part inferred from counters alone; unresolved diagnosis/compatibility is explicit. Partial or Insufficient evidence is appropriate without an applicable repair source. |
| Clinical quantitative gap | `Which Agilent laboratory instrument supports at least 12 samples per run? Recommend one only if its capacity is documented.` Requirements: `At least 12 samples per run` | Abstain if the indexed pages lack that capacity. No synthetic Fixture Analyzer, guessed capacity or patient-suitability claim. |
| No candidate meets threshold | Port comparison above; change both question and requirement to `At least 7 1G Ethernet ports` | Explain that neither of these models meets the threshold; recommend neither. Known failure is an answer, not an unknown. |
| Unknown extra specification | Port comparison; also require `A documented purchase price below USD 1000` | Do not invent price or call a model fully eligible; disclose the unresolved constraint. |
| Wrong year / missing source | `What risks does Cisco disclose in its fiscal 2020 10-K?` Company Cisco, years 2020 | Insufficient evidence if no FY2020 filing exists; never substitute FY2025. |
| Conflicting customer context | Cisco FY2025 risk question with context company `Agilent`, year `2024` | Needs clarification before retrieval; resolve conflict, do not silently choose a company/year. |
| Google Cloud technical source | `According to the indexed Google Cloud documentation, how does retrieval-augmented generation ground an agent response, and what role does MCP play in connecting agents to tools?` Company Google, no report year | Technology; distinguish retrieval grounding from tool connectivity and cite the respective indexed sources. Do not answer from unsupported general knowledge. |
| SOFI 2023 drivers | `According to SOFI 2023, what major drivers contribute to food insecurity and malnutrition?` years 2023 | CSR with SOFI 2023 evidence; explain drivers, not just quote unrelated financing text. |
| Conflict and malnutrition | `According to SOFI 2023, how can conflict worsen food insecurity and malnutrition?` years 2023 | CSR; supported mechanisms and limitations, no individual patient diagnosis. |
| Food-price mechanism | `According to SOFI 2024, how do increased food prices affect the affordability of healthy diets?` years 2024 | CSR; directly explain price/affordability consequences with supporting passages, not a generic policy list. |
| Multiple-company coverage | `Summarize operational risks disclosed in Cisco and Agilent fiscal 2025 10-K filings.` years 2025, company empty | Both companies, separate citations, or explicitly identify the missing company's answer as unresolved. |
| Out of scope | `Write a birthday poem.` all context empty | No specialist recommendation or fabricated citations; explain supported consulting scope. |
| Re-submit behavior | Change model, observations or requirements, then Ask again | Uses the entire question and context currently shown. Previous answers are not chat memory. Keep the original problem when adding details. |
| Bad generated citation / source injection | Run the automated tests below | Forged IDs/quotes and instructions embedded in sources are rejected, not followed. |
| Service unavailable | Stop backend, then Ask | Clear API error and refreshed availability; no stale green readiness pretending success. |

## Repeatable commands

Offline regression tests, including diagnostic decision, grounding, API and UI:

```bash
HF_HUB_OFFLINE=1 .venv/bin/pytest -q --import-mode=importlib
```

Live acceptance run (uses the configured model API and full local index):

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/evaluate_consulting.py \
  --mode live \
  --knowledge-factory consulting_platform.knowledge.service:create_knowledge_service \
  --cases evaluation/consulting/demo_acceptance.jsonl \
  --output evaluation/consulting/runs/live_demo_checks.json
```

For one case add `--case-id router_diagnostics` or `--case-id cross_domain`.
Add `--repeat 2` to assess repeated-run stability. Model results can vary; a pass
is evidence for that saved run, not a guarantee. Reports preserve the actual
responses. Never keep retrying until an answer happens to look good and count
only that success.

Direct CLI alternative, with no web server needed:

```bash
.venv/bin/consult ask --mode live \
  --knowledge-factory consulting_platform.knowledge.service:create_knowledge_service \
  --question "Our Cisco router has intermittent failures. Which replacement parts should we order?" \
  --context-json '{"company":"Cisco","product_model":"Cisco Catalyst 8300"}'
```

## Assignment evaluation evidence

| Assignment capability | What to demonstrate / measure |
|---|---|
| Orchestration | Four single-domain cases plus cross-domain; compare selected domains to expected domains and report routing exact match/precision/recall. |
| Authoritative grounding + RAG | Open cited pages; check exact company/year/model and whether each cited passage supports its claim. Source counts alone are not quality scores. |
| Intelligent recommendation | Router eligible/ineligible examples, unknown-price abstention, no diagnostic guessing; check ranking relevance against explicit requirements. |
| Cross-domain reasoning | Cross-domain answer must resolve the product choice, vendor risks and proposed food-security workflow together, not just list three sets of facts. |
| Explainability | Inline citations, trace, uncertainty and explicit unresolved questions; preserve a saved response with the run ID. |
| Retrieval accuracy | Label relevant chunk/document IDs in the frozen corpus and measure retrieved hit@k/precision@k separately. The consulting runner's final-answer evidence hit is not retrieval accuracy. |
| Response quality + business value | Human-score saved outputs 1–5 for correctness, directly answering the decision, relevance, useful next steps and measurable outcomes. Unsupported causal claims, irrelevant specs and unexplained fragments lose credit even if reference IDs are valid. |

`evaluation/consulting/cases.jsonl` provides development and held-out cases.
Follow `evaluation/consulting/README.md` for gold evidence labels and human ratings.
Unlabeled quality metrics remain null. Automated regex/status checks do not replace
this semantic review. Retail has separate training/evaluation and UI checks; it
is not a substitute for any of the consulting cases above.

## Checks after the multipart-answer fix

The Clinical example must visibly answer **both** questions in separate sections.
An answer containing only Seahorse measurements fails, even if it says Answered.
The shorter version ending immediately after “what do digital pathology slide
scanners do?” must also work; the extra workflow/approval sentence is not required.

The cross-domain example already supplies the port requirement. Asking for extra
port requirements or calling that same proven port choice unresolved is a failure,
not an instruction to keep rephrasing the demo prompt.

The troubleshooting preset is deliberately different: no exact chassis SKU or
fault evidence is supplied. Needs clarification is correct for ordering parts.
The response must still explain what to do now and how to measure recurrence.
Clicking Ask again with unchanged inputs cannot supply device logs that the app
has never received. A full knowledge index contains documentation, not the current
health or diagnosis of the customer's actual router.
