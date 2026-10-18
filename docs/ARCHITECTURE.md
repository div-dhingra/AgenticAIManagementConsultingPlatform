# Project architecture

Open [the offline visual walkthrough](architecture.html) for four rendered SVG diagrams,
or open the individual [specialist](graphs/specialist_workflow.svg),
[multi-agent](graphs/multi_agent.svg), [overall project](graphs/project_architecture.svg)
and [RAG pipeline](graphs/knowledge_pipeline.svg) diagrams. Regenerate the visual
exports locally with `.venv/bin/python scripts/render_architecture.py`.

These diagrams describe the implemented prototype. Open this Markdown in a Mermaid-capable preview or on GitHub. Each diagram also has an editable `.mmd` file. The five existing `manager.mmd`, `technology.mmd`, `clinical.mmd`, `sec.mmd`, and `corporate_responsibility.mmd` files are generated from the actual compiled LangGraphs with `.venv/bin/consult graphs`.

## Specialist Workflow

All four specialists compile this same four-node topology with separate state and policies. Only analyze_extract calls the model; retrieval, literal checks and ranking are application code. SEC may extract separate companies concurrently within that node; Clinical may extract two independent subjects in separate bounded batches. These are bounded workflows, not arbitrary tool-selection loops.

```mermaid
flowchart LR
    A[Domain task + customer context] --> B[retrieve: focused searches + hard filters]
    B --> C[analyze_extract: LLM facts, candidates, proposed actions]
    C --> D[validate_evidence: exact spans, identities, values, units]
    D --> E[rank_candidates: deterministic requirements comparison]
    E --> F[Return grounded domain result to manager]
    KB[(Local hybrid knowledge index)] --> B
    P[Domain policy: Technology / Clinical / SEC / CSR] -.-> C
    P -.-> D
    N[Ranking skips when no product comparison is requested] -.-> E
```

## Multi Agent

Only selected specialists execute. Their graphs run concurrently, then the manager reviews each domain and checks each requested subquestion. At most one targeted follow-up search is allowed if the call and time budgets permit. The manager has one routing node; it does not need one reasoning node per specialist. Synthesis selects existing supported IDs; the final composer writes the customer response.

```mermaid
flowchart TD
    U[Customer question + optional context] --> V[validate_input: conflicting context?]
    V --> R[route: LLM selects domains + subquestion checklist]
    R --> T[Technology specialist graph]
    R --> C[Clinical specialist graph]
    R --> S[SEC specialist graph]
    R --> F[CSR / Food-security specialist graph]
    T --> E[validate_evidence: per-domain reviews + coverage]
    C --> E
    S --> E
    F --> E
    E --> Q{Missing source answer and budget/time left?}
    Q -->|Once only| H[targeted_follow_up: refine one domain search]
    H --> E
    Q -->|No| Y[synthesize: select IDs + sequence actions]
    Y --> Z[finalize: fault gate, coverage, natural answer + citations]
    Z --> O[Answered / Needs clarification / Partial / Insufficient evidence]
```

## Project Architecture

Consulting and retail share the UI and HTTP server but use separate services. Retail uses learned basket association rules, not an LLM-only recommender. The configured hosted model handles consulting routing/extraction/review/selection; it does not receive API secrets as prompt text. Readiness measures availability, not proven answer quality.

```mermaid
flowchart TD
    USER[Enterprise customer / consultant] --> UI[Streamlit: Consulting + Retail tabs]
    UI -->|HTTP JSON| API[FastAPI: validation, startup, deadlines, errors]
    API -->|POST /consult| MAN[Consulting manager + specialist LangGraphs]
    API -->|GET /health| HEALTH[Service and corpus readiness]
    API -->|POST /recommend + catalog| RET[Retail service: association rules + ranking]
    CSV[Basket CSV] --> TRAIN[Offline retail training / evaluation]
    TRAIN --> ART[(Retail artifact)]
    ART --> RET
    MAN -->|Shared KnowledgeService protocol| KB[Local retrieval: strict metadata + hybrid ranking]
    KB --> INDEX[(Chunks + embeddings + TF-IDF index)]
    MAN --> LLM[Cohere adapter / explicitly selected Mistral adapter]
    LLM --> PROVIDER[External model API]
    MAN --> RESULT[Grounded answer, recommendations, citations, uncertainty]
    RET --> PRODUCTS[Basket recommendations + scores + explanations]
    RESULT --> API
    PRODUCTS --> API
    NOTE[Retail association-rule recommendation does not need an LLM] -.-> RET
```

## Knowledge Pipeline

Ingestion downloads/reads source documents before indexing. Runtime agents retrieve stored text and exact spans; having a URL in metadata does not give an agent a web-browsing tool. Rebuild the index after source/chunk changes, not after prompt or UI changes. CPU embedding execution is serialized inside retrieval to avoid the earlier native concurrency crash.

```mermaid
flowchart LR
    DOC[Authoritative PDFs / retrieved product pages] --> EX[Extract document text + citation metadata]
    EX --> CH[Chunk text with stable IDs + document/page/company/year]
    CH --> EM[Offline CPU embeddings + TF-IDF build]
    EM --> IX[(Local index)]
    Q[Domain subquestions + strict metadata filters] --> SEARCH[Hybrid search + optional adjacent passages]
    IX --> SEARCH
    SEARCH --> SP[Exact quote spans with IDs]
    SP --> GEN[Specialist extraction]
    GEN --> VAL[Literal + semantic evidence validation]
    VAL --> OUT[Answer with linked source / indexed document + page]
    NO[Source URLs identify provenance; agents do not browse links at answer time] -.-> SEARCH
```

## Answer Decisions

Missing customer facts and missing source coverage are different. Supplying a model family does not supply diagnostics. A user-reported diagnostic account is still a reported premise, not independently verified fault evidence. A source may support a product feature without answering why a customer outage occurred. Fault-specific parts advice must pass compatibility and semantic checks; none can be inferred from co-occurrence.

```mermaid
flowchart TD
    A[Reviewed claims + actions + customer inputs] --> D{Router incident?}
    D -->|Missing SKU or diagnostic account| C[Needs clarification: no parts order, diagnostic plan]
    D -->|Inputs supplied| F[Require documented fault relevance + compatibility before part selection]
    D -->|No incident| G[Check every requested subquestion / company / year]
    F --> G
    G --> H{Any useful grounded content?}
    H -->|No| I[Insufficient evidence]
    H -->|Yes| J{Unresolved requested answer?}
    J -->|Yes| K[Partial: answer supported parts and name gaps]
    J -->|No| L[Answered: clear decision + evidence + next steps]
    P[Warnings alone do not mean Partial; hypotheses stay labeled proposals] -.-> J
```

## Code map

- `agents/service.py`: manager graph and fault/coverage gates.
- `agents/specialists.py`: individual specialist graphs.
- `agents/diagnostics.py`: diagnostic intake and proposed investigation.
- `agents/answer.py`: customer answer from accepted claims/actions.
- `knowledge/service.py` and `retrieval.py`: runtime retrieval and citation context.
- `knowledge/index.py`: offline index build.
- `api/app.py`: API factory and service wiring.
- `ui/app.py`: two-tab interface and examples.
- `retail/`: training, rule ranking, artifact-backed recommendations.
