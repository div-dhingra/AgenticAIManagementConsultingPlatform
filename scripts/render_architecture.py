"""Render the explanatory architecture diagrams locally, without a web service.

Mermaid sources in docs/graphs and compiled LangGraph exports remain the editable
node/edge reference. This renderer provides an offline visual walkthrough.
"""

import textwrap
from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "graphs"


def render(name, title, subtitle, nodes, edges, width=1400, height=1000):
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        '<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#52647b"/></marker></defs>',
        f'<rect width="{width}" height="{height}" fill="#f6f8fc"/>',
        f'<text x="45" y="52" fill="#152840" font-family="Arial,sans-serif" font-size="29" font-weight="bold">{escape(title)}</text>',
        f'<text x="45" y="86" fill="#52647b" font-family="Arial,sans-serif" font-size="17">{escape(subtitle)}</text>',
    ]
    for start, end, label in edges:
        a, b = nodes[start], nodes[end]
        ax, ay, aw, ah = a[:4]
        bx, by, bw, bh = b[:4]
        if ay == by:
            points = (
                [(ax + aw, ay + ah / 2), (bx, by + bh / 2)]
                if ax < bx
                else [(ax, ay + ah / 2), (bx + bw, by + bh / 2)]
            )
        else:
            x1, x2 = ax + aw / 2, bx + bw / 2
            y1, y2 = (ay + ah, by) if ay < by else (ay, by + bh)
            mid = (y1 + y2) / 2
            points = [(x1, y1), (x1, mid), (x2, mid), (x2, y2)]
        svg.append(
            '<polyline points="'
            + " ".join(f"{x},{y}" for x, y in points)
            + '" fill="none" stroke="#52647b" stroke-width="2" marker-end="url(#arrow)"/>'
        )
        if label:
            x, y = points[len(points) // 2]
            svg.append(
                f'<text x="{x + 9}" y="{y - 8}" fill="#52647b" font-family="Arial,sans-serif" font-size="15">{escape(label)}</text>'
            )
    colors = {
        "model": ("#e9e2ff", "#7352b4"),
        "data": ("#e0f1ea", "#2f8264"),
        "gate": ("#fff0d5", "#b87a19"),
        "code": ("#e5efff", "#4277b4"),
    }
    for x, y, w, h, label, kind in nodes.values():
        fill, border = colors[kind]
        svg.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{border}" stroke-width="1.7"/>'
        )
        lines = [
            s for line in label.split("\n") for s in textwrap.wrap(line, width=max(16, int(w / 10)))
        ]
        first = y + h / 2 - (len(lines) - 1) * 12
        for i, line in enumerate(lines):
            svg.append(
                f'<text x="{x + w / 2}" y="{first + i * 24}" text-anchor="middle" dominant-baseline="middle" fill="#152840" font-family="Arial,sans-serif" font-size="18">{escape(line)}</text>'
            )
    svg.append(
        f'<text x="45" y="{height - 30}" fill="#52647b" font-family="Arial,sans-serif" font-size="16">Purple: model step · Blue: application code · Green: data/service · Amber: checks and decisions</text></svg>'
    )
    (OUT / f"{name}.svg").write_text("\n".join(svg))


def main():
    render(
        "specialist_workflow",
        "Individual specialist workflow",
        "Each of the four domain agents has its own state and policy, using the same four graph nodes.",
        {
            "task": (50, 160, 240, 100, "Domain task + customer context", "data"),
            "retrieve": (
                370,
                160,
                260,
                100,
                "1. Retrieve\nFocused searches + hard filters",
                "code",
            ),
            "extract": (
                720,
                160,
                260,
                100,
                "2. Analyze / extract\nLLM facts, candidates, actions",
                "model",
            ),
            "validate": (
                1080,
                160,
                260,
                100,
                "3. Validate evidence\nExact spans, units, identities",
                "gate",
            ),
            "rank": (1080, 355, 260, 100, "4. Rank candidates\nDeterministic comparison", "code"),
            "result": (
                720,
                355,
                260,
                100,
                "Return grounded result\nManager reviews it next",
                "data",
            ),
            "kb": (370, 355, 260, 100, "KnowledgeService\nStored document passages", "data"),
            "policy": (
                370,
                560,
                610,
                100,
                "Technology: fault and compatibility checks\nClinical: laboratory evidence\nSEC: company / year / risk\nCSR: documented priorities vs hypotheses",
                "gate",
            ),
        },
        [
            ("task", "retrieve", ""),
            ("retrieve", "extract", ""),
            ("extract", "validate", ""),
            ("validate", "rank", ""),
            ("rank", "result", ""),
            ("kb", "retrieve", "retrieve"),
            ("policy", "extract", "domain policy"),
        ],
        height=780,
    )
    render(
        "multi_agent",
        "Manager and specialist agents",
        "Only selected domain graphs run; execution is parallel, bounded and evidence checked.",
        {
            "input": (510, 120, 350, 85, "Question + customer context", "data"),
            "route": (
                510,
                260,
                350,
                85,
                "Validate inputs + route\nDomains and subquestions",
                "model",
            ),
            "tech": (45, 420, 270, 95, "Technology specialist\nCisco / Google equipment", "model"),
            "clinical": (
                390,
                420,
                270,
                95,
                "Clinical specialist\nAgilent laboratory equipment",
                "model",
            ),
            "sec": (735, 420, 270, 95, "SEC specialist\nCompany / year filings", "model"),
            "csr": (1080, 420, 270, 95, "CSR specialist\nFood security / SOFI", "model"),
            "review": (
                510,
                600,
                350,
                95,
                "Per-domain evidence review\nCheck every subquestion",
                "gate",
            ),
            "follow": (
                1000,
                735,
                350,
                95,
                "One targeted source follow-up\nOnly if time / calls permit",
                "code",
            ),
            "synth": (
                510,
                760,
                350,
                95,
                "Synthesize validated IDs\nNo new factual claims",
                "model",
            ),
            "final": (
                510,
                915,
                350,
                95,
                "Finalize clear answer\nFault gate, citations, status",
                "gate",
            ),
        },
        [
            ("input", "route", ""),
            ("route", "tech", "selected"),
            ("route", "clinical", "selected"),
            ("route", "sec", "selected"),
            ("route", "csr", "selected"),
            ("tech", "review", ""),
            ("clinical", "review", ""),
            ("sec", "review", ""),
            ("csr", "review", ""),
            ("review", "follow", "optional"),
            ("follow", "review", "re-review"),
            ("review", "synth", ""),
            ("synth", "final", ""),
        ],
        height=1090,
    )
    render(
        "project_architecture",
        "Overall project architecture",
        "Two experiences share one UI and API; consulting and market-basket recommendations use separate services.",
        {
            "user": (500, 120, 370, 85, "Enterprise customer / consultant", "data"),
            "ui": (500, 265, 370, 90, "Streamlit\nConsulting + Retail tabs", "code"),
            "api": (500, 425, 370, 90, "FastAPI\nHTTP contracts, health, errors", "code"),
            "agents": (
                80,
                600,
                350,
                100,
                "POST /consult\nManager + four specialist graphs",
                "model",
            ),
            "retail": (
                970,
                600,
                350,
                100,
                "POST /recommend + catalog\nAssociation-rule ranking",
                "code",
            ),
            "provider": (520, 780, 350, 100, "Cohere model API\nExplicit Mistral option", "model"),
            "knowledge": (
                80,
                780,
                350,
                100,
                "KnowledgeService\nHybrid retrieval + metadata",
                "data",
            ),
            "index": (80, 960, 350, 90, "Local chunks + embeddings\nTF-IDF index", "data"),
            "artifact": (
                970,
                780,
                350,
                100,
                "Retail model artifact\nTrained from basket CSV",
                "data",
            ),
        },
        [
            ("user", "ui", ""),
            ("ui", "api", "HTTP JSON"),
            ("api", "agents", "consulting"),
            ("api", "retail", "retail"),
            ("agents", "provider", "model calls"),
            ("agents", "knowledge", "retrieve"),
            ("knowledge", "index", ""),
            ("retail", "artifact", "load"),
        ],
        height=1130,
    )
    render(
        "knowledge_pipeline",
        "Document ingestion and RAG",
        "Runtime agents read indexed text. A provenance URL is not an automatic browsing tool.",
        {
            "docs": (
                70,
                160,
                330,
                105,
                "Authoritative PDFs / product pages\nRead / download before indexing",
                "data",
            ),
            "extract": (
                530,
                160,
                330,
                105,
                "Extract + chunk\nStable IDs, company, year, page",
                "code",
            ),
            "build": (990, 160, 330, 105, "Offline index build\nCPU embeddings + TF-IDF", "code"),
            "search": (
                990,
                405,
                330,
                105,
                "Runtime hybrid search\nHard domain / company / year filters",
                "code",
            ),
            "question": (
                530,
                405,
                330,
                105,
                "Focused domain subquestions\nOptional nearby passages",
                "data",
            ),
            "spans": (
                990,
                650,
                330,
                105,
                "Exact quote spans\nOriginal document + page IDs",
                "data",
            ),
            "model": (
                530,
                650,
                330,
                105,
                "Specialist + manager review\nReject unsupported claims",
                "model",
            ),
            "answer": (
                70,
                650,
                330,
                105,
                "Grounded customer answer\nCitations and explicit limits",
                "gate",
            ),
        },
        [
            ("docs", "extract", ""),
            ("extract", "build", ""),
            ("build", "search", "index"),
            ("question", "search", "query"),
            ("search", "spans", ""),
            ("spans", "model", ""),
            ("model", "answer", ""),
        ],
        height=890,
    )
    paths = ["specialist_workflow", "multi_agent", "project_architecture", "knowledge_pipeline"]
    (OUT.parent / "architecture.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Consulting platform architecture</title><style>body{font-family:Arial,sans-serif;background:#eaf0f6;margin:25px}main{max-width:1400px;margin:auto}section{background:white;padding:15px;margin-bottom:30px;border-radius:12px}img{width:100%;height:auto}a{color:#245db0}</style><main><h1>Consulting platform architecture</h1><p>Offline diagrams. See ARCHITECTURE.md for node descriptions and editable Mermaid sources.</p>'
        + "".join(
            f'<section><a href="graphs/{p}.svg">Open full-size SVG</a><img src="graphs/{p}.svg" alt="{p.replace("_", " ")}"></section>'
            for p in paths
        )
        + "</main>"
    )


if __name__ == "__main__":
    main()
