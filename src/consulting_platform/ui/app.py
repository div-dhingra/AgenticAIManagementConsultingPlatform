"""Streamlit UI: Consulting and Retail Recommendations tabs, both over HTTP.

Run:  streamlit run src/consulting_platform/ui/app.py
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from consulting_platform.examples import CROSS_DOMAIN_QUESTION, EXAMPLES, example_form_values
from consulting_platform.ui import client, retail_view
from consulting_platform.ui.consulting_view import (
    citation_title,
    diagnostic_warning,
    readable_rationale,
)

CROSS_DOMAIN_PRESET = CROSS_DOMAIN_QUESTION
DEMO_PRESET = "Which fixture analyzer supports at least 12 samples per run?"

STATUS_STYLE = {
    "answered": ("green", "Answered", None),
    "partial": ("orange", "Partial answer", "Some requested parts remain unresolved; the answer below identifies them."),
    "needs_clarification": ("blue", "Needs clarification", "Add the requested details to your question or optional context, then click Ask again. Each submission uses the inputs currently shown."),
    "insufficient_evidence": ("red", "Insufficient evidence", "The sources do not support an answer. Do not treat this as procurement guidance."),
}
DOMAIN_LABELS = {
    "technology": "Technology",
    "clinical": "Clinical",
    "sec": "SEC filings",
    "corporate_responsibility": "Corporate responsibility",
}

st.set_page_config(page_title="AI Consulting Coworker", layout="wide")

defaults = {
    "cart": [],
    "retail_result": None,
    "catalog_result": None,
    "health_result": None,
    "consult_busy": False,
    "consult_pending": None,
    "consult_result": None,
    "consult_input_error": None,
    "question": "",
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)


def show_api_error(result: client.ApiResult) -> None:
    suffix = f" (request ID `{result.request_id}`)" if result.request_id else ""
    st.error(f"{result.error_message}{suffix}")
    for detail in result.error_details:
        st.caption(f"• `{detail.get('field')}`: {detail.get('message')}")


def demo_banner(mode: str | None) -> None:
    if mode == "demo":
        st.warning("**Demo mode**: every result below comes from synthetic fixtures, not the real sources.", icon="🧪")


# --------------------------------------------------------------------------- sidebar: readiness


def refresh_health() -> None:
    st.session_state.health_result = client.get_health()


if st.session_state.health_result is None:
    refresh_health()

with st.sidebar:
    st.subheader("System status")
    st.button("Refresh status", on_click=refresh_health)
    health = st.session_state.health_result
    if not health.ok:
        show_api_error(health)
    else:
        h = health.data
        overall = ":green[ok]" if h["status"] == "ok" else ":orange[degraded]"
        st.markdown(f"Mode: **{h['mode']}** · Overall: **{overall}**")
        for name, state in h["services"].items():
            color = {"ready": "green", "partial": "orange"}.get(state, "red")
            st.markdown(f"- {name.capitalize()}: :{color}[{state}]")
        st.markdown("**Knowledge coverage by domain**")
        st.dataframe(
            [
                {
                    "Domain": DOMAIN_LABELS[d],
                    "Status": s["status"],
                    "Docs": s["document_count"],
                    "Chunks": s["chunk_count"],
                    "Notes": "; ".join(s["warnings"]),
                }
                for d, s in h["knowledge"]["domains"].items()
            ],
            hide_index=True,
            use_container_width=True,
        )
        for warning in h["warnings"]:
            st.caption(f"⚠️ {warning}")

st.title("AI Consulting Coworker")
health_mode = health.data["mode"] if health.ok else None
demo_banner(health_mode)

consult_tab, retail_tab = st.tabs(["Consulting", "Retail Recommendations"])


# --------------------------------------------------------------------------- consulting tab


def set_question(text: str) -> None:
    st.session_state.question = text


def set_example(case_id: str) -> None:
    st.session_state.update(example_form_values(case_id))


def set_controlled_example(question: str, company: str, years: str, requirements: str) -> None:
    st.session_state.update(
        question=question,
        ctx_company=company,
        ctx_years=years,
        ctx_requirements=requirements,
        ctx_model="",
        ctx_symptoms="",
        ctx_region="",
    )


def lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def submit_consult() -> None:
    if st.session_state.consult_busy:
        return
    st.session_state.consult_input_error = None
    try:
        years = [int(y) for y in st.session_state.ctx_years.replace(",", " ").split()]
    except ValueError:
        st.session_state.consult_input_error = "Report years must be whole numbers, e.g. 2025, 2026."
        return
    context = {
        "company": st.session_state.ctx_company.strip() or None,
        "product_model": st.session_state.ctx_model.strip() or None,
        "symptoms": lines(st.session_state.ctx_symptoms),
        "requirements": lines(st.session_state.ctx_requirements),
        "report_years": years,
        "region": st.session_state.ctx_region.strip() or None,
    }
    st.session_state.consult_pending = (st.session_state.question, context, st.session_state.ctx_max_recs)
    st.session_state.consult_busy = True


def render_consult(data: dict) -> None:
    demo_banner(data["mode"])
    color, label, note = STATUS_STYLE[data["status"]]
    domains = ", ".join(DOMAIN_LABELS[d] for d in data["domains"]) or "none"
    st.markdown(f"### :{color}[{label}]")
    st.caption(f"Domains: {domains}")
    if note:
        if data["status"] == "needs_clarification" and data["summary"].startswith("**Do not order replacement parts yet."):
            note = "The diagnostic plan below answers what you can do now. Selecting a replacement part still requires the exact router SKU and fault evidence listed below."
        st.info(note)
    st.markdown(data["summary"])
    for warning in data["warnings"]:
        if not diagnostic_warning(warning):
            st.warning(warning)

    if data["clarification_questions"]:
        st.markdown("**Questions we need answered**")
        for question in data["clarification_questions"]:
            st.markdown(f"- {question}")

    ref = {c["evidence_id"]: i for i, c in enumerate(data["citations"], start=1)}

    def refs(ids: list[str]) -> str:
        return " ".join(f"[{ref[i]}]" for i in ids if i in ref)

    if data["recommendations"]:
        st.markdown("#### Recommendations")
    for rec in data["recommendations"]:
        with st.container(border=True):
            st.markdown(f"**{rec['rank']}. {rec['title']}** {refs(rec['evidence_ids'])}")
            assessment = {"supported": "Supported by sources", "conditional": "Needs verification", "proposed": "Suggested action"}[rec["assessment"]]
            st.caption(f"{assessment} · {rec['priority'].capitalize()} priority")
            if rec["assessment"] == "conditional":
                st.caption("Conditional: depends on the prerequisites below being met.")
            elif rec["assessment"] == "proposed":
                st.caption("Proposed action motivated by the evidence; its impact is not proven by the sources.")
            st.write(readable_rationale(rec["rationale"]))
            for title, items in (("Before you proceed", rec["prerequisites"]), ("Next steps", rec["next_steps"])):
                if items:
                    st.markdown(f"*{title}*")
                    for item in items:
                        st.markdown(f"- {item}")
            if rec["success_metrics"]:
                with st.expander("How to evaluate this recommendation"):
                    for item in rec["success_metrics"]:
                        st.markdown(f"- {item}")

    if data["findings"]:
        with st.expander("Validated findings by specialist"):
            for finding in data["findings"]:
                tag = "supported" if finding["assessment"] == "supported" else "inference"
                st.markdown(f"- {finding['statement']} *({tag}, {DOMAIN_LABELS[finding['domain']]})* {refs(finding['evidence_ids'])}")

    if data["citations"]:
        with st.expander(f"Sources and quoted passages ({len(data['citations'])})"):
            st.markdown("#### Evidence")
            for i, c in enumerate(data["citations"], start=1):
                page = f"p. {c['page']}" if c["page"] else "page n/a"
                if c["printed_page"]:
                    page += f" (printed {c['printed_page']})"
                meta = " · ".join(str(x) for x in (c["company"], c["report_year"], page) if x)
                title = citation_title(c)
                uri = c["source_uri"]
                label = f"[{title}]({uri})" if uri.startswith(("https://", "http://")) else title
                st.markdown(f"**[{i}] {label}** — {meta}")
                st.caption(f"> {c['excerpt']}")
                if not uri.startswith(("https://", "http://")):
                    st.caption("Indexed source document; page reference shown above.")

    level = data["uncertainty"]["level"]
    st.markdown(f"#### Uncertainty: **{level}**")
    for reason in data["uncertainty"]["reasons"]:
        st.markdown(f"- {reason}")

    diagnostic_events = [t for t in data["trace"] if t["detail"].startswith(("Specialist diagnostic:", "Review diagnostic:"))]
    diagnostic_warnings = [w for w in data["warnings"] if diagnostic_warning(w)]
    if diagnostic_events or diagnostic_warnings:
        with st.expander("Evidence checks and specialist notes"):
            st.caption("Diagnostics describe model extraction and review. Source availability is reported in the answer above.")
            for event in diagnostic_events:
                domain = DOMAIN_LABELS.get(event["domain"], "Manager review")
                st.write(f"{domain}: {event['detail']}")
            for warning in diagnostic_warnings:
                st.write(warning)

    with st.expander("Operational trace"):
        st.dataframe(
            [{"Step": t["step"], "Domain": t["domain"] or "—", "Status": t["status"], "Detail": t["detail"]} for t in data["trace"] if t not in diagnostic_events],
            hide_index=True,
            use_container_width=True,
        )
    st.caption(f"Request ID `{data['request_id']}` · mode {data['mode']}")


with consult_tab:
    st.caption("Compare equipment against your requirements, understand company risks, and explore food-security priorities using cited documents.")
    controlled = health.ok and any(
        w.startswith("Controlled evidence test:") for w in health.data["warnings"]
    )
    if controlled:
        st.warning("Live test on selected real passages. This uses your model API; full-document search and coverage are not being tested.")
        examples = st.columns(3)
        for column, label, args in [
            (examples[0], "Cisco risks (2025)", ("What operational risks does Cisco disclose in its 2025 filing?", "Cisco", "2025", "")),
            (examples[1], "Compare router ports", ("Compare Cisco C8300-2N2S-4T2X and C8300-2N2S-6T for at least 5 1G Ethernet ports.", "Cisco", "", "At least 5 1G Ethernet ports")),
            (examples[2], "SOFI financing (2024)", ("According to SOFI 2024, can the report calculate the financing gap, and what measurement improvements are needed?", "", "2024", "")),
        ]:
            column.button(label, on_click=set_controlled_example, args=args)
    elif health_mode != "demo":
        columns = st.columns(3)
        for index, case_id in enumerate(("cisco_ports", "cisco_risks", "sofi_priorities", "clinical_overview", "router_diagnostics")):
            columns[index % 3].button(EXAMPLES[case_id]["label"], on_click=set_example, args=(case_id,), help=EXAMPLES[case_id].get("help"))
    presets = st.columns([1, 1, 4])
    presets[0].button(
        "Cross-domain example",
        on_click=set_example,
        args=("cross_domain",),
    )
    if health_mode == "demo":
        presets[1].button("Demo fixture question", on_click=set_question, args=(DEMO_PRESET,))

    st.text_area("Customer question", key="question", height=110, max_chars=4000)
    with st.expander("Optional context (kept for follow-up questions; each request stands alone)"):
        c1, c2, c3 = st.columns(3)
        c1.text_input("Company", key="ctx_company")
        c2.text_input("Product model", key="ctx_model")
        c3.text_input("Region", key="ctx_region")
        c4, c5 = st.columns(2)
        c4.text_area("Symptoms (one per line)", key="ctx_symptoms", height=90)
        c5.text_area("Requirements with units (one per line)", key="ctx_requirements", height=90)
        c6, c7 = st.columns(2)
        c6.text_input("Report years (e.g. 2025, 2026)", key="ctx_years")
        c7.slider("Max recommendations", 1, 5, 3, key="ctx_max_recs")

    st.button(
        "Ask" if not st.session_state.consult_busy else "Working…",
        type="primary",
        disabled=st.session_state.consult_busy or not st.session_state.question.strip(),
        on_click=submit_consult,
    )
    if st.session_state.consult_input_error:
        st.error(st.session_state.consult_input_error)

    if st.session_state.consult_busy and st.session_state.consult_pending:
        question, context, max_recs = st.session_state.consult_pending
        with st.spinner("Routing to specialist agents and checking evidence (up to ~90 s)…"):
            st.session_state.consult_result = client.consult(question, context, max_recs)
        if not st.session_state.consult_result.ok and st.session_state.consult_result.status_code is None:
            # Cached green readiness must not survive a lost backend connection.
            refresh_health()
        st.session_state.consult_pending = None
        st.session_state.consult_busy = False
        st.rerun()

    result = st.session_state.consult_result
    if result is not None:
        st.divider()
        if result.ok:
            render_consult(result.data)
        else:
            show_api_error(result)


# --------------------------------------------------------------------------- retail tab


def refresh_recommendations() -> None:
    st.session_state.retail_result = client.recommend(list(st.session_state.cart))


def add_to_cart(sku: str | None) -> None:
    if sku and sku not in st.session_state.cart:
        st.session_state.cart.append(sku)
        st.session_state.last_added = sku
        st.session_state.product_picker = None
        refresh_recommendations()


def remove_from_cart(sku: str) -> None:
    st.session_state.cart = [s for s in st.session_state.cart if s != sku]
    st.session_state.last_added = None
    refresh_recommendations()


def clear_cart() -> None:
    st.session_state.cart = []
    st.session_state.last_added = None
    refresh_recommendations()


def load_catalog() -> None:
    st.session_state.catalog_result = client.get_catalog()


with retail_tab:
    st.markdown(retail_view.STYLE, unsafe_allow_html=True)
    if st.session_state.catalog_result is None or not st.session_state.catalog_result.ok:
        load_catalog()
    catalog_result = st.session_state.catalog_result
    if not catalog_result.ok:
        show_api_error(catalog_result)
        st.button("Retry", on_click=load_catalog)
    else:
        demo_banner(catalog_result.data["mode"])
        names = {p["product_id"]: p["name"] for p in catalog_result.data["products"]}
        st.session_state.cart = [s for s in st.session_state.cart if s in names]
        last_added = st.session_state.get("last_added")

        left, right = st.columns([2, 3], gap="large")
        with left:
            st.subheader("Cart")
            options = [sku for sku in names if sku not in st.session_state.cart]
            pick_col, add_col = st.columns([4, 1], vertical_alignment="bottom")
            pick_col.selectbox(
                "Search products",
                options,
                index=None,
                key="product_picker",
                placeholder="Type a name or SKU…",
                format_func=lambda sku: f"{names[sku]} ({sku})",
            )
            add_col.button(
                "",
                icon=":material/add_shopping_cart:",
                help="Add to cart",
                on_click=lambda: add_to_cart(st.session_state.product_picker),
                disabled=not options,
                use_container_width=True,
            )

            if not st.session_state.cart:
                st.caption("Your cart is empty. Search for a product to start.")
            for sku in st.session_state.cart:
                row = st.columns([6, 1], vertical_alignment="center")
                row[0].html(retail_view.cart_item(names[sku], sku, sku == last_added))
                row[1].button("", icon=":material/close:", key=f"rm-{sku}", help=f"Remove {sku}", on_click=remove_from_cart, args=(sku,))
            if st.session_state.cart:
                st.button("Clear cart", icon=":material/delete_sweep:", on_click=clear_cart)

        with right:
            st.subheader("Frequently bought together")
            if st.session_state.retail_result is None:
                refresh_recommendations()
            rec_result = st.session_state.retail_result
            if not rec_result.ok:
                show_api_error(rec_result)
            else:
                data = rec_result.data
                for warning in data["warnings"]:
                    st.warning(warning)
                if data["status"] == "empty_cart":
                    st.html(retail_view.empty_state("🛒", "Your cart is empty", data["explanation"], shimmer=True))
                elif data["status"] == "no_supported_pattern":
                    st.html(retail_view.empty_state("🔍", "No suggestion for this exact cart", data["explanation"]))
                else:
                    recs = data["recommendations"]
                    dark = (st.context.theme.type or "light") == "dark"
                    graph, height = retail_view.copurchase_graph(
                        [(names[s], s) for s in st.session_state.cart], recs, dark
                    )
                    components.html(graph, height=height)
                    st.caption(data["explanation"])
                    top_score = recs[0]["score"]
                    for i, rec in enumerate(recs):
                        card, button = st.columns([7, 1], vertical_alignment="center")
                        card.html(retail_view.recommendation_card(rec, i, top_score))
                        button.button(
                            "",
                            icon=":material/add_shopping_cart:",
                            key=f"add-{rec['product_id']}",
                            help=f"Add {rec['product_id']} to cart",
                            on_click=add_to_cart,
                            args=(rec["product_id"],),
                        )
                st.caption(
                    "Confidence and lift are co-purchase statistics from past transactions. They do not show that "
                    f"products are compatible or guarantee a business outcome. Model `{data['model_version']}`."
                )
