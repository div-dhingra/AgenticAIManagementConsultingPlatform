# Shared integration handoff

Import Pydantic models/protocols/exceptions from
`consulting_platform.contracts`. JSON schemas are exported in `schemas/v1/`.
Public JSON keys and response shapes remain contract version 1.0 from all three
plan files. The extraction/ranking objects are private agent implementation types.

## API teammate

Create the service once at application startup, using the KB teammate's instance:

```python
from consulting_platform.agents import create_consulting_service
from consulting_platform.config import Settings
from consulting_platform.contracts import ConsultRequest

settings = Settings.from_env()
consulting_service = create_consulting_service(knowledge_service, settings=settings)

# Inside your POST /consult handler, after generating a UUID request_id:
result = await consulting_service.run_consultation(
    ConsultRequest.model_validate(request_body), request_id=request_id,
)
response_body = result.model_dump(mode="json")
```

The API supplies the UUID and `X-Request-ID`. Its validation/exception handlers
must return the shared ErrorResponse envelope. Map `ServiceUnavailableError` to
503, `UpstreamTimeoutError` to 504, and `UpstreamModelError` to 502. Keep normal
domain outcomes at 200. Unexpected failures are 500, without exposing stack traces.
Service construction can fail for missing credentials: catch that on startup,
mark consulting unavailable, and let retail/health start. `/health` must not invoke
the model. Read Settings credential/config presence and B's `get_status()` instead.

In demo mode, `create_consulting_service(settings=Settings(mode="demo"))` works
without any other dependencies. `agents.mocks.FakeConsultingService` is another
drop-in for route tests. Do not enable demo in live mode after catching an error.
The provider key never appears in a customer request or response.

No changes are needed to `/recommend`, `/catalog`, or the retail model for agents
to work. Those implementations remain yours. Shared RetailRequest cannot validate
unknown SKUs by itself; your service/route must validate IDs against its catalog
and return the contracted 422 error for unknown IDs.

For development before the real KB is ready, the CLI also accepts `--mode live
--evidence-file fixtures/consulting/controlled_real_passages.json`. This runs real
provider calls on selected real passages and always warns that search/full source
coverage are not evaluated. It is an explicit local testing path, not an outage
fallback or a new HTTP parameter. See `docs/CONTROLLED_TESTS.md`.

## Knowledge teammate

Implement the exact protocol in `contracts.py`:

```python
async def retrieve(query, domains, filters=None, top_k=5):
    # Return RetrievalResult, with EvidenceChunk values and actual metadata.
    ...

def get_status():
    # Return KnowledgeStatus, describing all four domains.
    ...
```

`Domain` is a string enum, and filters are a `RetrievalFilters` object. Empty
filter lists mean unrestricted. Categories combine with AND; values within each
category combine with OR. The agent wrapper checks returned filters again.
Avoid silently widening years or company filters. Keep globally stable unique
chunk IDs; an ID reused for different content is an error.

Use HTTPS source URLs or repo-relative paths, one-based physical PDF pages,
explicit nullable fields, and finite retrieval scores in [0,1]. No personal
absolute paths. Text must preserve source wording and table context so exact
quotes and candidate units can be validated. No structured candidate catalog or
attribute-extraction method is required from your module.

Healthy no-match returns `status="ok", chunks=[]`. A missing/corrupt requested
index returns `unavailable` or raises `ServiceUnavailableError`. Partial results
carry warnings. `get_status()` reports actual loaded artifacts; source URLs alone
are not an indexed corpus. Return `mode="live"` for real artifacts. The agent
rejects accidental demo/live mixing.

For CLI/evaluation convenience expose a no-argument local factory such as
`consulting_platform.knowledge.service:create_knowledge_service`; choose the
actual module name together and pass it with `--knowledge-factory`. This hook is
for trusted developer startup, never a customer-controlled API argument.

## Immediate handoff artifacts

1. Share this source package, pyproject.toml, requirements.lock.txt, and
   `schemas/v1/` with both teammates.
2. Share `fixtures/consulting/contract_examples.json` for request/response bodies.
3. Share `fixtures/consulting/evidence.json` only as visibly synthetic test data.
4. Keep secrets and large source/index artifacts out of the shared code history.

API readiness, evaluation reports, and the interface should disclose actual
corpus coverage. Real-corpus retrieval and human answer-quality metrics remain
pending until B's artifacts and reviewed evidence labels are supplied.
