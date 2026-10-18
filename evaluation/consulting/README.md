# Consulting evaluation

`cases.jsonl` contains 24 proposed real-source questions: 12 development and 12
held-out test cases. They cover four domains, cross-domain questions, and missing
information/out-of-scope behavior. Human reviewers must confirm the expected
routes before using these as a benchmark. Do not tune on the test subset.

The gold evidence IDs and human ratings are deliberately unfilled. Ask the
knowledge teammate to label evidence IDs from the frozen corpus snapshot. Add
`human_review` objects only after reviewing the saved response for the exact
`run_id` being scored:

```json
{
  "run_id": "the-output-report-run-id",
  "citation_support": {"supported": 3, "reviewed": 4},
  "recommendation_relevance": 4,
  "response_quality": 4,
  "notes": "One causal statement exceeds the quoted evidence."
}
```

Rubric for relevance/quality: **1** incorrect/unusable; **2** major gaps;
**3** useful with material omissions; **4** grounded and actionable with minor
gaps; **5** fully addresses stated scope, evidence, uncertainty, and next actions.
Use null for a dimension that is not applicable. Report reviewed sample counts.
Citation support means semantic support for each claim, not merely that an ID
exists. No retrieval accuracy or human quality score should be inferred from
similarity scores, fake-model outputs, or unreviewed responses.

Run offline integration checks (synthetic fixtures, not an accuracy benchmark):

```bash
.venv/bin/python scripts/evaluate_consulting.py --mode demo --split development
```

Run a live evaluation after the KB teammate supplies an importable factory and
you configure your key in `.env`:

```bash
.venv/bin/python scripts/evaluate_consulting.py --mode live --split development \
  --knowledge-factory consulting_platform.knowledge.service:create_knowledge_service
```

The module/factory path above loads the implemented local knowledge service.
The evaluator records routing exact match, micro precision/recall, status checks,
citation-reference validity, observed latency, warnings, and errors. Retrieval
hit@5 remains the KB teammate's separate metric: this runner measures **final
answer evidence hit**, which can differ because agents select a subset of chunks.
It never relabels final citations as retrieval accuracy. Reports stay in the
ignored `runs/` directory and contain customer questions/source excerpts; use
assignment data rather than sensitive real customer data.

Use `--score-existing report.json` after annotating case human reviews for that
run. This recomputes metrics without making paid inference calls. Re-run test
cases only when evaluating a frozen implementation; save corpus/config versions
alongside the report. The synthetic smoke report in this directory is labeled
demo-only; it is not evidence of live model or real-corpus accuracy.

See `docs/CONSULTING_TEST_GUIDE.md` for exact UI questions, expected decisions,
negative cases, and assignment coverage. `demo_acceptance.jsonl` holds the
repeatable demo inputs; these are development checks, not a held-out benchmark.
