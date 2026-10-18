"""Demo inputs shared by the UI, CLI testing guide and evaluators; no answers."""

CROSS_DOMAIN_QUESTION = (
    "We operate a hospital network with diabetes research labs. Compare Cisco "
    "C8300-2N2S-4T2X and C8300-2N2S-6T against our requirement for at least five "
    "1G Ethernet ports per site. Assess only this documented port requirement; "
    "overall deployment suitability remains unverified. Summarize demand, inventory, "
    "and purchase-commitment risks disclosed in Cisco's fiscal 2025 10-K. Also "
    "summarize risks disclosed in Agilent's fiscal 2025 10-K; if filing evidence is "
    "unavailable, explicitly leave that assessment unresolved. Our proposed initiative "
    "is to connect sites for sharing nutrition-program data. Explain relevant "
    "food-security priorities from SOFI 2024, distinguishing documented priorities "
    "from hypothetical ways connectivity could support the initiative."
)

EXAMPLES = {
    "cisco_ports": {
        "label": "Compare router ports",
        "question": "Compare Cisco C8300-2N2S-4T2X and C8300-2N2S-6T for at least 5 1G Ethernet ports.",
        "context": {"company": "Cisco", "requirements": ["At least 5 1G Ethernet ports"]},
    },
    "cisco_risks": {
        "label": "Cisco risks (2025)",
        "question": "Summarize demand, inventory, and purchase-commitment risks disclosed in Cisco fiscal 2025 10-K.",
        "context": {"company": "Cisco", "report_years": [2025]},
    },
    "sofi_priorities": {
        "label": "SOFI priorities (2024)",
        "question": "Explain SOFI 2024 priorities for financing food security and nutrition and improving data for evidence-based decisions.",
        "context": {"report_years": [2024]},
    },
    "clinical_overview": {
        "label": "Agilent laboratory options",
        "question": "According to the indexed Agilent product pages, what do Seahorse XF analyzers measure, and what do digital pathology slide scanners do? Distinguish these laboratory workflows; do not infer clinical approval or patient suitability.",
        "context": {"company": "Agilent"},
    },
    "router_diagnostics": {
        "label": "Router troubleshooting",
        "help": "This example lacks an exact SKU and fault logs. Expect a diagnostic plan and requests for those details before any parts are selected.",
        "question": "Our Cisco enterprise router is experiencing intermittent connectivity failures. Which replacement parts should we order, and what actions should we take to prevent future outages? Our router is a Cisco Catalyst 8300.",
        "context": {"company": "Cisco", "product_model": "Cisco Catalyst 8300"},
    },
    "cross_domain": {
        "label": "Cross-domain example",
        "question": CROSS_DOMAIN_QUESTION,
        "context": {
            "requirements": ["At least 5 1G Ethernet ports per site"],
            "report_years": [2024, 2025],
        },
    },
}


def example_form_values(case_id):
    example = EXAMPLES[case_id]
    context = example["context"]
    # Clear every old filter: a previous company/year must not affect this example.
    return {
        "question": example["question"],
        "ctx_company": context.get("company", ""),
        "ctx_model": context.get("product_model", ""),
        "ctx_years": ", ".join(map(str, context.get("report_years", []))),
        "ctx_requirements": "\n".join(context.get("requirements", [])),
        "ctx_symptoms": "\n".join(context.get("symptoms", [])),
        "ctx_region": context.get("region", ""),
        "ctx_max_recs": 3,
        "consult_result": None,
        "consult_input_error": None,
    }
