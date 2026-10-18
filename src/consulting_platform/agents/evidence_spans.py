"""Short, exact source spans that models can cite by ID instead of retyping."""

import re


def evidence_spans(text: str, max_chars: int = 380) -> list[dict[str, str]]:
    spans, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            # Prefer a whole sentence/paragraph so a financial label does not
            # disappear while its amount is retained. Long sentences still have
            # overlapping verbatim windows, bounded by Support.quote's limit.
            sentences = list(re.finditer(r"[.!?](?:\s+|$)|\n\n", text[start:end]))
            boundary = (
                sentences
                if sentences and sentences[-1].end() > max_chars // 2
                else list(re.finditer(r"\s+", text[start:end]))
            )
            if boundary:
                end = start + (
                    boundary[-1].end() if boundary is sentences else boundary[-1].start()
                )
        quote = text[start:end].strip()
        if quote:
            spans.append({"span_id": f"s{len(spans)}", "quote": quote})
        if end >= len(text):
            break
        next_start = max(start + 1, end - 50)
        boundary = re.search(r"\s+", text[next_start:end])
        start = next_start + boundary.end() if boundary else end
    return spans
