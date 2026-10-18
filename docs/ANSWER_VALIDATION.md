# Complete, cited consulting answers

The manager checks the **requested questions**, not the number of warnings.
Routing creates a private checklist with separate questions for each company, risk
topic, comparison and requested implication. The reviewer must link each answer
to accepted finding/recommendation IDs. Company and report-year metadata are
checked again in code; rejected or omitted content cannot satisfy the checklist.
Each domain is reviewed separately to keep its claim/source set small. Covered
IDs must also appear in the accepted-content lists; inconsistent reviews require
a bounded format repair. The reviewer also checks whether routing missed a
question in the original request. A known negative answer (e.g. four ports fail
a five-port requirement) is resolved, not an unknown.

A fully answered request can still include routine extraction diagnostics,
explicit scope boundaries, and proposed actions whose impact has not been proven.
`partial` means an actual requested answer or required comparison remains
unresolved. `needs_clarification` means missing **customer** input. Missing sources
are not a request for the customer to upload specifications. `insufficient_evidence`
means no useful supported answer survived validation.

## Retrieval and guardrails

Each specialist performs the main search plus at most eight focused checklist
searches, preserving domain/company/year filters. Results are interleaved so a
single subtopic cannot occupy every slot. At most ten search hits plus four exact
neighboring chunks enter a retrieval round. Neighbor records keep their original
IDs, source text and page metadata. No source text is merged under an existing ID.
SEC extraction is isolated by company so one vendor cannot crowd out another.
Embedded Risk Factors headings are recognized for section-aware ranking, including
headings after a preceding paragraph in the same chunk.
If time and call budgets allow, one unresolved checklist question can drive a
second retrieval/analysis/review round. There is no unbounded search/chat loop. Each generation permits one format repair,
with at most three repairs and twelve total model calls across a request. A bad
optional draft with empty evidence supports is discarded without deleting valid
facts from the same response; no supporting citation is invented.

The local knowledge service adds optional `expand_context(chunks, max_chunks=4)`;
other implementations do not need this capability. The public HTTP request and
response contracts are unchanged. No index rebuild is needed for these changes.
Existing CPU embedding inference and its native-thread lock remain in place.

Longer, sentence-aware exact spans help preserve complete financial labels.
A combined inventory provision plus purchase-commitment liability cannot be
reported as the standalone liability. Semantic review still checks claims,
causality, numerical labels and periods; mechanical checks do not establish
financial accuracy by themselves. Factual financial disclosures use exact
attributed source excerpts instead of publishing an unchecked paraphrase of
financial measures. Cohere routing uses fixed nullable slots for the four domains,
preventing duplicate specialist keys in the provider wire format. The public
Python/API domain map is unchanged.

## Answer presentation

The response starts with the documented equipment decision, followed by vendor
risks and food-security priorities/implications, as relevant to the question.
All accepted findings are retained, including third and later findings in a domain.
Every finding has inline numbered citations. Proposed actions carry cited premises,
next steps and suggested measures; hypothetical benefits remain labeled as such.
The UI keeps specialist findings and operational diagnostics in expanders.
Exact source-parsed port counts and attributed filing excerpts are retained even
if a model selector omits them. Narrow completeness checks verify source identity,
year and requested topic words; other interpretations still require review.
A requested connectivity feasibility assessment can be constructed from a cited
data-needs premise. It compares options and measures a baseline, and does not
recommend spending or assert achieved nutrition outcomes.
Opaque SEC accession titles become readable company/year source labels.

The cross-domain preset now explicitly requests Agilent fiscal 2025 risk analysis.
With the full index, the expected answer assesses the two Cisco models against the
five-port requirement, addresses each requested Cisco risk topic and Agilent,
explains SOFI 2024 priorities, and offers a measured, hypothetical initiative.
The 6T model passes the five-port count and the 4T2X model fails it. Neither fact
establishes overall hospital deployment suitability. If an actual required part
cannot be supported, `partial` remains the correct result.

## Verification

Offline suite (no API key or provider requests):

```sh
HF_HUB_OFFLINE=1 .venv/bin/pytest -q --import-mode=importlib
.venv/bin/ruff check src tests scripts/evaluate_consulting.py
```

Repeated live acceptance run (uses the configured provider key and sends retrieved
source passages to that provider):

```sh
HF_HUB_OFFLINE=1 .venv/bin/python scripts/evaluate_consulting.py \
  --mode live \
  --knowledge-factory consulting_platform.knowledge.service:create_knowledge_service \
  --cases evaluation/consulting/demo_acceptance.jsonl \
  --repeat 2
```

For just the cross-domain case, add `--case-id cross_domain`. Reports go to the
ignored `evaluation/consulting/runs/` directory. Each row reports mechanical
acceptance checks: status, routing, expected topics, product choice, inline
citations, and proposed actions/measures. Check **every repeated row**. These checks
do not measure whether every claim is entailed by its source; that still requires
reading the citations. Human quality metrics remain unmeasured until reviewed.

The local full-index retrieval check found Cisco technology sources, both Cisco
and Agilent fiscal 2025 passages, and SOFI 2024 passages. Live verification uses explicit user approval to send those passages to Cohere.
Mechanical acceptance reports are stored locally alongside the evaluation results;
do not treat mechanical checks as proof of human-reviewed answer quality.

After pulling these changes, restart FastAPI and Streamlit. For the full knowledge
service keep `APP_MODE=live` and unset `CONSULT_EVIDENCE_FILE` in both `.env` and the
shell environment that launches the API. The selected-passages mode remains a
separate limited-coverage test.

The explicit demo profile is `COHERE_MODEL=command-a-03-2025` for all roles.
Two full cross-domain runs passed the mechanical acceptance checks (approximately
55 and 64 seconds) in `runs/live_command_a_cross_domain.json`. This is a small
implementation check, not a statistical reliability guarantee. Earlier A+ runs
exposed intermittent format/provider failures; provider errors can still occur.
Model compatibility is documented in [Cohere's model catalog](https://docs.cohere.com/v2/docs/models)
and [structured-output documentation](https://docs.cohere.com/v2/docs/structured-outputs).

Final local verification: 257 tests passed; lint, dependency and diff-whitespace
checks passed. The four other Command A presets also passed once each in
`evaluation/consulting/runs/live_command_a_presets.json`: router ports, Cisco
risks, SOFI priorities, and missing diagnostics (correctly needs clarification).
Together with the two cross-domain repeats, all six final live cases passed their
mechanical criteria. All 38 citation excerpts across these responses were checked
verbatim against the loaded index, as were the attributed financial quotations.
These are implementation checks, not human quality ratings or a guarantee that
provider calls never fail. Earlier diagnostic runs remain separate from these
final reports. No answer is loaded from an evaluation report at runtime.

## Customer decisions and presentation

Router incident requests now require a diagnostic account even when a model name
is present. A family name such as Catalyst 8300 is not an exact chassis SKU.
Without those inputs, the response starts with a no-order decision, gives a
proposed investigation, asks for the missing observations, and suggests comparing
outage frequency/duration before and after the eventual fix. This method is
explicitly a proposal, not a diagnosis or a procedure quoted from a datasheet.
Source product features remain background; they cannot count as a confirmed repair.
Component candidates additionally need an explicit grounded compatibility attribute.
If observations are supplied but evidence cannot establish a replacement, that
selection remains unresolved instead of asking for the same observations again.

SEC findings retain the exact attributed disclosure as proof. A qualitative
paraphrase is presented only if independently accepted by semantic review against
its quoted supports. The paraphrase cannot bypass review just because its literal
proof survives. Financial amounts/percentages remain outside this presentation
shortcut. Company names and report years are grouped rather than repeated for each
excerpt. The original proof and all cited source passages remain inspectable.

Agilent laboratory equipment questions belong to Clinical, including Seahorse XF
and pathology scanners; technical wording does not make them Technology requests.
The main live presets now cover all four specialists, cross-domain collaboration
and the missing-diagnostics behavior. Their question/context values are centralized
in `src/consulting_platform/examples.py` and clear old filters when selected.

See [the test guide](CONSULTING_TEST_GUIDE.md) for exact questions, expected
outcomes, commands and assignment evaluation coverage, and [the architecture
walkthrough](ARCHITECTURE.md) for the graphs and full system diagrams.

Specialist structured-output schemas now enumerate the exact chunk IDs available
in that extraction batch. The earlier live review exposed invented evidence IDs
that caused an entire company's otherwise plausible draft to be rejected. The
provider constraint reduces that failure; local ID/span validation still rejects
any invalid pair. It does not relax quote matching or invent a citation alias.

For viewing, [the standalone HTML architecture walkthrough](architecture.html)
loads local SVG files and needs no external rendering service. Mermaid sources
and actual compiled graph exports remain in `docs/graphs/`.

## Customer-answer validation on October 18, 2026

- **270 offline tests passed**, including API/UI flows, missing diagnostic facts
  with a family name or exact SKU, compatible-but-not-proven-faulty parts, false
  clinical/SEC routing, and rejection of an unaccepted financial paraphrase.
  Ruff and dependency checks passed. SVG exports parse successfully.
- The seven-case live run in ignored
  `evaluation/consulting/runs/live_final_customer_examples.json` completed all
  requests. Six met their mechanical expectations; the SOFI financing request
  exposed a spurious SEC subtask and therefore correctly surfaced a partial
  result for that faulty route. That failure is retained in the report.
- The SOFI-only route boundary was then corrected without discarding its financing
  subquestions. Its targeted live recheck answered successfully.
- The final three-case report,
  `evaluation/consulting/runs/live_final_natural_answers.json`, passes the Cisco
  risk, SOFI financing/data, and cross-domain acceptance checks. Latencies were
  about 29, 25 and 47 seconds. The final SEC answers contain independently reviewed
  paraphrases (six in the Cisco-only run, seven in the cross-domain run), with
  exact attributed proof retained separately. All 22 citation excerpts in these
  three responses were checked verbatim against the loaded index, allowing only
  line-wrap whitespace normalization.
- Router port ranking, Clinical overview, missing-model diagnostics and
  model-family-without-diagnostics passed their live checks in the seven-case run.
  The narrow later SOFI/prose changes do not alter their input contracts.

These are development and integration observations, not a held-out accuracy claim.
Formal retrieval quality, recommendation relevance and human response-quality
metrics still need labeled evidence and review of saved runs. The test guide
includes the rubric and explicitly leaves unmeasured metrics unclaimed. Detailed
clinical capacity and repair recommendations still require the relevant source
manuals; product landing pages are not a substitute for those specifications.

## Multipart questions and false clarifications

A later user run exposed two failures despite earlier passing samples:

- The router-port answer was correct but was followed by an unresolved port-choice
  question and a request for additional requirements outside the stated scope.
  Report years from the shared SEC/SOFI context could contaminate an undated
  technical answer requirement. Product questions now discard those unrelated
  years while retaining explicitly dated technical-document requests. A reviewer’s
  free-text port objection cannot override a mechanically verified port comparison,
  and extra-port clarification is excluded once that requested comparison is known.
- A Clinical response answered only the Seahorse question while reporting Answered.
  Explicit coordinated clinical questions are now preserved in the answer checklist,
  even when the router omits one. Describing what two instruments do is not itself a
  procurement-ranking request. Extraction uses two bounded subject batches with
  relevant source passages; final coverage requires findings for each requested
  subject. The output gives separate Seahorse and pathology-scanner sections.
  With a scanner source deliberately removed, regression tests require Partial
  rather than allowing a Seahorse finding to count as the scanner answer.

A failed follow-up also must not erase a previously grounded answer: findings are
retained for another review, with new supports preferred for repeated statements.
This preserves useful partial results without bypassing semantic review.

The router incident preset still intentionally requires clarification. A product
family and general intermittent failures do not establish which component failed.
The UI now explains that the diagnostic plan is actionable now while part selection
awaits the exact SKU and fault evidence. It does not turn missing diagnostic facts
into a fabricated order recommendation.

Validation for this multipart fix: **275 offline tests passed**, including the
reported false port clarification and a router that omits the scanner question.
The negative clinical case uses a broad document title but no scanner-function
passage; a Seahorse statement still cannot count as the scanner answer.

`evaluation/consulting/runs/live_multipart_completeness.json` records two runs each
of the cross-domain prompt, the full clinical preset and the user's shorter clinical
question: all six passed the mechanical route/status/content criteria, with no
clarification questions. Their 34 citation excerpts were checked against indexed
text. Inspection also exposed unnecessary submodel elaboration, so function-only
clinical questions now avoid unrequested named-submodel details and emphasize the
requested purpose. The final two clinical checks in
`evaluation/consulting/runs/live_clinical_function_answers.json` both passed and
explicitly describe live-cell metabolic measurements and glass-slide digitization.
All seven excerpts in those final responses matched the index verbatim apart from
line-wrap whitespace. These observations supplement, not replace, semantic review
and the documented human evaluation rubric.
