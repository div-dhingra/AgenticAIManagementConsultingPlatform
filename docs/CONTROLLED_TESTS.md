# Test the real model before the full knowledge service exists

Use `--evidence-file` to run the live agent graphs against a small, explicitly
selected set of real source passages. This requires your configured Cohere key
and makes real API calls. It does not use the fake demo model. No API/frontend
server or teammate's KB index is needed.

```bash
.venv/bin/consult ask --mode live \
  --evidence-file fixtures/consulting/controlled_real_passages.json \
  --question 'What operational risks does Cisco disclose in its 2025 filing?' \
  --context-json '{"company":"Cisco","report_years":[2025]}'
```

The response is `mode: live` because inference and evidence are real. A persistent
warning explains that passages are preselected and search quality/full coverage
are not being evaluated. Useful answers will normally be `partial` due to that
warning; this is intentional. Do not confuse it with a failed API request. No
automatic fallback to this source occurs when a production KB fails.

## Sources and provenance

- **Technology:** selected factual columns from Cisco Catalyst 8300 datasheet,
  Table 2, physical/printed page 6. The original HTTPS URL is in the chunk. The
  small table is transcribed; PDF layout is normalized into rows/columns.
- **SEC:** verbatim passage from the supplied Cisco fiscal 2025 10-K, physical
  page 13 / printed page 11, on operating-result fluctuations and contributing
  factors. This is a selected excerpt, not the entire risk section.
- **CSR:** verbatim passages from the supplied SOFI 2024 report, physical pages
  24 and 25 / printed pages xxii and xxiii, on drivers and financing measurement.
- **Clinical:** no real instrument-capability passages have been selected yet.
  Missing clinical evidence should lead to abstention, not invented instrument facts.

`fixtures/consulting/controlled_source_manifest.json` records original local PDF
filenames, SHA-256 hashes, and pages. Copies for manual audit are under the ignored
`fixtures/source_documents/` directory on the owner's machine. Teammates can use
their copies of the original PDFs to audit the same pages; raw PDFs are not needed
to execute the controlled test because the selected text is included in JSON.
The table's `-` must not be silently treated as a documented numeric zero.
`retrieval_score: 1.0` denotes manual inclusion, not measured similarity/confidence.

## Checks you can perform now

1. **Routing and cited findings:** run the first command. Only Cisco SEC evidence
   from 2025 should appear. Check each claim against the supplied passage.
2. **Real candidate extraction/ranking:**

   ```bash
   .venv/bin/consult ask --mode live \
     --evidence-file fixtures/consulting/controlled_real_passages.json \
     --question 'Compare Cisco C8300-2N2S-4T2X and C8300-2N2S-6T for a requirement of at least 5 1G Ethernet ports.' \
     --context-json '{"company":"Cisco","requirements":["At least 5 1G Ethernet ports"]}'
   ```

   The selected table gives 4 1G ports for C8300-2N2S-4T2X and 6 for
   C8300-2N2S-6T. For this single requirement the 6T model meets the minimum;
   the 4T2X model does not. Do not claim overall suitability for unstated needs.
   Check exact model identity, speed class, count, and quoted table headers.
   This is an expected check, not a claim that the live model has passed it.
3. **CSR source limitations:**

   ```bash
   .venv/bin/consult ask --mode live \
     --evidence-file fixtures/consulting/controlled_real_passages.json \
     --question 'According to SOFI 2024, can the report calculate the financing gap, and what measurement improvements are needed?' \
     --context-json '{"report_years":[2024]}'
   ```

   The selected page xxiii says available data cannot establish a realistic stock
   of financing or the financing gap, and calls for improved sources/methodologies.
   A precise numerical gap would exceed the evidence.
4. **Cross-domain coordination:** ask about Cisco router interfaces, Cisco vendor
   risks, and SOFI 2024 food-security financing priorities together. Findings should
   use technology/SEC/CSR sources, while new action sequences remain proposals.
5. **Missing evidence:** ask for Agilent instrument capacity. There are no clinical
   passages in this set, so no supported capacity or purchase recommendation exists.
6. **Clarification:** repeat the unknown-router failure question. A parts order
   must not be justified just by the datasheet's interface table.

Compare source text, extracted facts, constraints, final recommendations, and
citations manually. Valid JSON and existing citation IDs do not establish that
the meaning of a claim is supported. Save live outputs if you want repeatable
human review; live generations need not match exact wording between runs.

## Live parsing checks

The original SEC and router-comparison commands were rerun with Cohere after
correcting PDF quote layout handling and provider JSON shape decoding. One verified
SEC run returned five supported findings, and one verified comparison returned a
supported C8300-2N2S-6T option for the five-1G-port requirement. Both were `partial`
because evidence is deliberately limited. These are individual integration checks,
not an accuracy rate or guarantee of identical future outputs. Real model output
can still be rejected when claims/quotes/fields are unsupported.

If a response is empty, inspect `warnings` for evidence rejection categories. If
structured output fails after the bounded retry, the error now identifies the
model role and invalid field paths/types. No reinstall or API contract change is
needed for these fixes in an editable installation.

## What still depends on teammates

| Validation | Dependency |
|---|---|
| Model routing, extraction/ranking, evidence use, and cross-domain behavior on these selected sources | Neither teammate; configured model access only |
| Corpus ingestion, realistic search, retrieval hit@k, full domain/source coverage | Teammate 1's real KB, indexed artifacts, and evidence labels |
| HTTP request/error handling, UI behavior, and final application demo | Teammate 2's API/frontend integrated with teammate 1's KB |
| Retail recommendation quality | Teammate 2's retail model and held-out evaluation |

This controlled subset checks agent behavior on selected real evidence. It is
not a held-out benchmark, an exhaustive source evaluation, or a replacement for
the agreed KnowledgeService implementation.
