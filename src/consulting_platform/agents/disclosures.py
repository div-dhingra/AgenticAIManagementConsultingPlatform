"""Readable, exact financial excerpts; no rewritten metrics or causal clauses."""

import re

from consulting_platform.agents.models import Support

STOP = {
    "the",
    "and",
    "our",
    "for",
    "with",
    "fiscal",
    "filing",
    "disclose",
    "reports",
    "cisco",
    "agilent",
}


def disclosure_support(draft, chunks):
    words = set(re.findall(r"[a-z]{3,}", draft.statement.casefold())) - STOP
    topics = [
        pattern
        for pattern in (r"inventor", r"purchase.commitment", r"demand")
        if re.search(pattern, draft.statement, re.I)
    ]
    options = []
    for support in draft.supports:
        score = len(words & set(re.findall(r"[a-z]{3,}", support.quote.casefold())))
        score += 10 * sum(bool(re.search(topic, support.quote, re.I)) for topic in topics)
        options.append((score, support))
    for evidence_id in dict.fromkeys(s.evidence_id for s in draft.supports):
        text = chunks[evidence_id].text
        # Sentences and individual bullets stay contiguous in the actual source.
        for match in re.finditer(r"[^.!?•]+[.!?](?=\s|$)|(?<=•)[^•]+(?=•)", text):
            quote = match[0].strip()
            if not 40 <= len(quote) <= 400 or len(quote.split()) < 7:
                continue
            if not re.search(r"[a-z]{3}", quote):
                continue
            score = len(words & set(re.findall(r"[a-z]{3,}", quote.casefold())))
            score += 10 * sum(bool(re.search(topic, quote, re.I)) for topic in topics)
            if score >= 2:
                options.append((score + 0.5, Support(evidence_id=evidence_id, quote=quote)))
    if options:
        return max(options, key=lambda pair: pair[0])[1]
    return max(
        draft.supports,
        key=lambda support: len(words & set(re.findall(r"[a-z]{3,}", support.quote.casefold()))),
    )
