"""Customer fault evidence is separate from product specifications.

These checks establish whether a diagnostic account was supplied, never that the
reported diagnosis is true. Source compatibility and semantic review still apply.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DiagnosticIntake:
    requested: bool
    exact_model: str | None
    observations_supplied: bool
    parts_requested: bool
    component_fault_reported: bool = False

    @property
    def questions(self):
        if not self.requested:
            return []
        questions = []
        if not self.exact_model:
            questions.append(
                "What is the exact router model/SKU (not just the Catalyst 8300 series)?"
            )
        if not self.observations_supplied:
            questions.append(
                "During a drop, does the router reboot or lose power, or do only interfaces/"
                "connections drop? Share timestamped fault logs, interface errors and any "
                "diagnostic or isolation-test results."
            )
        return questions


def diagnostic_intake(request):
    text = " ".join(
        [request.question, request.context.product_model or "", *request.context.symptoms]
    )
    # A request about disclosed vendor outages is not a customer router fault.
    hardware = re.search(r"\b(router|catalyst|c\d{4}|isr|asr)\b", text, re.I)
    incident = re.search(
        r"intermittent|fail(?:ure|ing|ed|s)?\b|outages?|troubleshoot|connectivity (?:drops|loss)",
        text,
        re.I,
    )
    incident_account = re.search(
        r"intermittent|troubleshoot|failed|failing|experienc|connectivity (?:drops|loss)"
        r"|\b(?:our|my)\b[^.!?]{0,90}(?:failures?|outages?)",
        text,
        re.I,
    )
    requested = bool(hardware and incident and incident_account)
    # C8300 is a family; C8300-2N2S-6T is an identifiable chassis SKU.
    model = re.search(r"\b(?:C\d{4}|ISR\d{4}|ASR\d{4})(?:-[A-Z0-9]+)+\b", text, re.I)
    # Merely saying 'logs' or 'intermittent failures' supplies no observations.
    observations = bool(
        re.search(
            r"(?:log|diagnostic|alarm|test|counter|error|crc|link|reboot|power supply|transceiver)"
            r"[^.!?\n]{0,90}(?:shows?|reports?|confirms?|failed|detected|increments?|increases?|"
            r"at \d|after|remains?|persists?|disappears?|follows?)"
            r"|(?:\d{1,2}:\d{2}(?::\d{2})?[^\n]{0,80}(?:down|reset|error|failed))",
            text,
            re.I,
        )
    )
    parts = bool(re.search(r"\b(parts?|components?|replacements?|order|procure)\b", text, re.I))
    fault_reported = bool(
        re.search(
            r"(?:diagnos\w*|test|isola\w*|logs?)[^.!?\n]{0,70}(?:confirms?|isolates?|identifies?)"
            r"[^.!?\n]{0,45}(?:power supply|PSU|transceiver|interface card|fan module)"
            r"[^.!?\n]{0,30}(?:fail\w*|fault\w*|defect\w*)",
            text,
            re.I,
        )
    )
    return DiagnosticIntake(
        requested, model[0] if model else None, observations, parts, fault_reported
    )


def compatible_with_reported_model(candidate, intake):
    """Require an explicit, source-grounded compatibility attribute, not proximity."""
    if not intake.exact_model or intake.questions or not intake.component_fault_reported:
        return False
    return any(
        re.search(r"compatib|supported.model|chassis.model", name, re.I)
        and intake.exact_model.casefold()
        in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)+", str(fact.value).casefold())
        for name, fact in candidate.attributes.items()
    )


def diagnostic_answer(intake, has_component=False):
    if has_component:
        lead = "Treat the component below as a conditional option, not an authorization to order."
    else:
        lead = "**Do not order replacement parts yet.** No faulty, compatible component has been established from the information available."
    return (
        lead
        + " A router family name and datasheet features do not diagnose intermittent connectivity.\n\n"
        "**Suggested diagnostic plan:** record when each drop occurs and whether it affects power, "
        "the whole router, one interface or the upstream connection. Have the network team correlate "
        "those observations with logs and interface/error counters, then use model-specific "
        "troubleshooting guidance to isolate the cause before selecting a replacement. This is a "
        "proposed investigation, not a diagnosis or a vendor-verified repair procedure.\n\n"
        "**Preventing recurrence:** choose corrective and monitoring actions after the cause is "
        "confirmed. Track outage frequency and duration before and after the change; verify that "
        "the fault no longer recurs. Do not treat a power-supply feature or a voice-survivability "
        "feature in a datasheet as evidence of what caused these drops."
    )
