"""Extract count facts from unambiguous retrieved tables, independently of LLM keys."""

import math
import re

from consulting_platform.agents.models import (
    AttributeFact,
    Candidate,
    Constraint,
    DomainTask,
    Support,
)


def _attribute_words(attribute: str) -> str:
    attribute = re.sub(r"([A-Z])([A-Z][a-z])", r"\1 \2", attribute)
    attribute = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", attribute)
    return re.sub(r"[_-]+", " ", attribute)


def port_speed(text: str) -> float | None:
    text = _attribute_words(text).casefold()
    speeds = {
        float(n) * (1000 if scale == "g" else 1)
        for n, scale in re.findall(
            r"(?<![a-z0-9])(\d+(?:\.\d+)?)\s*([gm])(?:bps|bit/s|bit|igabit|egabit|be|b|e)?(?![a-z0-9])",
            text,
        )
    }
    return next(iter(speeds)) if len(speeds) == 1 else None


def port_dimension(attribute: str, unit: str | None = None) -> float | None:
    if not re.search(r"\bports?\b", _attribute_words(attribute), re.I):
        return None
    speed = port_speed(attribute)
    unit_speed = port_speed(unit or "")
    if speed is not None and unit_speed is not None and speed != unit_speed:
        return None
    return speed if speed is not None else unit_speed


def canonical_port_task(task: DomainTask, requirements: list[str]) -> DomainTask:
    """Recognize complete simple count requirements, never partial complex clauses."""
    parsed, port_requirements = [], [r for r in requirements if re.search(r"\bports?\b", r, re.I)]
    for requirement in port_requirements:
        match = re.fullmatch(
            r"\s*(?:at least|minimum(?: of)?)\s+(\d+)\s+(\d+(?:\.\d+)?\s*[gm](?:bps|bit/s|bit|igabit|egabit|be|b|e)?)\s+(?:ethernet\s+)?ports?(?:\s+per\s+(?:site|router|device))?[.!]?\s*",
            requirement,
            re.I,
        )
        if not match:
            return task
        speed = port_speed(match[2])
        if speed is None:
            return task
        parsed.append(
            Constraint(
                id=f"port_minimum_{speed:g}_{len(parsed)}",
                attribute=f"ports_{speed:g}mbps",
                operator="gte",
                expected_value=int(match[1]),
                unit="ports",
            )
        )
    if parsed:
        other = [
            c
            for c in task.constraints
            if not re.search(r"\bports?\b", _attribute_words(c.attribute), re.I)
        ]
        return task.model_copy(update={"constraints": [*other, *parsed]})
    return task


def _model_key(text):
    return re.sub(r"\s*-\s*", "-", text).casefold()


def _prose_rows(chunks):
    """Read explicit model declarations with count × speed Ethernet port lists.

    The columns below are internal parsed facts. Every citation is the ORIGINAL
    prose span; no synthetic table or rewritten quotation is presented as evidence.
    Optional, negated and 'up to' descriptions are deliberately left unresolved.
    """
    model = re.compile(r"\b([a-z][a-z0-9]*(?:\s*-\s*[a-z0-9]+)+)\s+is\s+an?\s+", re.I)
    rate = r"\d+(?:\.\d+)?\s*(?:-\s*)?(?:[gm]bps|gigabit|megabit)"
    count_rate = rf"\d+\s*[x×]\s*{rate}(?:\s+ethernet)?"
    spec = re.compile(
        rf"{count_rate}(?:\s*(?:and|plus|,)\s*{count_rate})*\s+(?:embedded\s+)?(?:layer\s*3\s+)?ethernet\s+ports\b",
        re.I,
    )
    counts = re.compile(rf"(\d+)\s*[x×]\s*({rate})", re.I)
    rows = []
    for chunk in chunks:
        models = list(model.finditer(chunk.text))
        for i, declaration in enumerate(models):
            end = models[i + 1].start() if i + 1 < len(models) else len(chunk.text)
            body = chunk.text[declaration.end():end]
            match = spec.search(body)
            if not match or re.search(r"\b(?:not|optional|up to)\b", body[:match.end()], re.I):
                continue
            quote = chunk.text[declaration.start():declaration.end() + match.end()]
            if len(quote) > 400:
                continue
            columns, cells = ["Model"], [declaration[1]]
            for count in counts.finditer(match[0]):
                columns.append(count[2] + " port count")
                cells.append(count[1])
            rows.append((chunk, None, quote, columns, cells))
    return rows


def table_candidates(chunks, task: DomainTask, candidates, question: str, proven=None):
    """Return source-derived facts using EXACT task keys; never parse findings prose.

    Labeled pipe tables and explicit model/port prose are supported. Unknown,
    noninteger, conflicting, or dash-valued cells stay unknown. No hardcoded models.
    """
    if not task.compare_candidates or not task.constraints:
        return candidates, task, []
    constraints = []
    for c in task.constraints:
        dimension = port_dimension(c.attribute, c.unit)
        unit = (c.unit or "").casefold().strip()
        count_unit = (
            unit in {"", "port", "ports", "count", "number", "ethernet ports"}
            or re.fullmatch(
                r"\d+(?:\.\d+)?\s*[gm](?:bps|bit/s|bit|igabit|egabit|be|b|e)?\s+(?:ethernet\s+)?ports?",
                unit,
            )
            or port_speed(unit) is not None
            and not re.search(r"bps|bit", unit)
        )
        constraints.append(
            c.model_copy(
                update={
                    "unit": "ports",
                    "attribute": c.attribute
                    if port_speed(c.attribute) is not None
                    else f"ports_{dimension:g}mbps",
                }
            )
            if dimension is not None and count_unit
            else c
        )
    task = task.model_copy(update={"constraints": constraints})
    eligible_constraints = [
        c
        for c in constraints
        if c.unit == "ports" and port_dimension(c.attribute, c.unit) is not None
    ]
    if not eligible_constraints:
        return candidates, task, []
    rows = _prose_rows(chunks)
    for chunk in chunks:
        header, columns = None, []
        for line in chunk.text.splitlines():
            if "|" not in line:
                header = None
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if cells[0].casefold() in {"model", "product", "instrument"}:
                header, columns = line, cells
                continue
            if (
                header is None
                or len(cells) != len(columns)
                or not cells[0]
                or re.fullmatch(r"[-: ]+", cells[0])
            ):
                continue
            if len(line) > 400 or len(header) > 400:
                continue
            rows.append((chunk, header, line, columns, cells))
    named = {
        _model_key(cells[0])
        for _, _, _, _, cells in rows
        if re.search(r"(?<![\w-])" + re.escape(_model_key(cells[0])) + r"(?![\w-])", _model_key(question))
    }
    grouped = {}
    for chunk, header, line, columns, cells in rows:
        label = cells[0]
        if named and _model_key(label) not in named:
            continue
        item = grouped.setdefault(_model_key(label), {"label": label, "identity": [], "values": {}})
        identity = Support(evidence_id=chunk.chunk_id, quote=line)
        item["identity"].append(identity)
        for constraint in eligible_constraints:
            speed = port_dimension(constraint.attribute, constraint.unit)
            indexes = [
                i
                for i, column in enumerate(columns)
                if "port" in column.casefold() and port_speed(column) == speed
            ]
            value = None
            if len(indexes) == 1:
                try:
                    number = float(cells[indexes[0]])
                    if math.isfinite(number) and number >= 0 and number.is_integer():
                        value = number
                except ValueError:
                    pass
            supports = ([Support(evidence_id=chunk.chunk_id, quote=header)] if header else []) + [identity]
            item["values"].setdefault(constraint.attribute, []).append((value, supports))
    merged = {_model_key(c.label): c for c in candidates if not named or _model_key(c.label) in named}
    notes = []
    for key, item in grouped.items():
        previous = merged.get(key)
        attributes = (
            {
                k: v
                for k, v in previous.attributes.items()
                if k in {c.attribute for c in task.constraints}
            }
            if previous
            else {}
        )
        source_attributes = set()
        for attribute, observations in item["values"].items():
            values = {value for value, _ in observations}
            attributes.pop(attribute, None)
            if len(values) == 1 and None not in values:
                attributes[attribute] = AttributeFact(
                    value=observations[0][0],
                    unit="ports",
                    supports=[s for _, spans in observations for s in spans],
                )
                source_attributes.add(attribute)
            else:
                notes.append(
                    f"Port table comparison unresolved for {item['label']} / {attribute}: missing, ambiguous, or conflicting cells."
                )
        merged[key] = Candidate(
            id=item["label"],
            label=item["label"],
            identity_supports=item["identity"],
            attributes=attributes,
        )
        if proven is not None and {c.attribute for c in task.constraints}.issubset(
            source_attributes
        ):
            proven.add(merged[key].id)
    return list(merged.values()), task, notes
