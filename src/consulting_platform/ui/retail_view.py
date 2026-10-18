"""Animated presentation for the Retail Recommendations tab.

Pure HTML/CSS (plus SVG SMIL in the co-purchase graph): no JavaScript and no data
access. Interactions stay as Streamlit buttons, so the UI still talks only to the API.
"""

from __future__ import annotations

from html import escape

ACCENT = "#7c5cff"
ACCENT_2 = "#22c3a6"
GRAPH_SCALE = 0.9  # iframe height relative to the viewBox height; matches a ~550-700px column

# Injected with st.markdown(unsafe_allow_html=True): st.html sanitizes <style> away.
# Keep it free of blank lines so Markdown treats it as one HTML block.
STYLE = f"""<style>
@property --rc-p {{ syntax: '<integer>'; initial-value: 0; inherits: false; }}
@keyframes rc-rise {{ from {{ opacity: 0; transform: translateY(14px) scale(.98); }} to {{ opacity: 1; transform: none; }} }}
@keyframes rc-grow {{ from {{ width: 0; }} to {{ width: var(--w); }} }}
@keyframes rc-count {{ from {{ --rc-p: 0; }} to {{ --rc-p: var(--p); }} }}
@keyframes rc-pop {{ 0% {{ transform: scale(.85); opacity: 0; }} 60% {{ transform: scale(1.04); opacity: 1; }} 100% {{ transform: scale(1); }} }}
@keyframes rc-glow {{ 0%, 100% {{ box-shadow: 0 0 0 0 {ACCENT}00; }} 50% {{ box-shadow: 0 0 0 6px {ACCENT}33; }} }}
@keyframes rc-float {{ 0%, 100% {{ transform: translateY(0); }} 50% {{ transform: translateY(-6px); }} }}
@keyframes rc-shimmer {{ from {{ background-position: -200% 0; }} to {{ background-position: 200% 0; }} }}
.rc-card {{
  display: flex; gap: 14px; align-items: flex-start;
  padding: 14px 16px; border-radius: 14px;
  border: 1px solid rgba(124, 92, 255, .22);
  background: linear-gradient(135deg, rgba(124, 92, 255, .08), rgba(34, 195, 166, .05));
  opacity: 0; animation: rc-rise .5s cubic-bezier(.2, .8, .2, 1) forwards;
  transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease;
}}
.rc-card:hover {{ transform: translateY(-2px); border-color: rgba(124, 92, 255, .55); box-shadow: 0 8px 24px rgba(124, 92, 255, .18); }}
.rc-rank {{
  flex: none; width: 34px; height: 34px; border-radius: 50%;
  display: grid; place-items: center; font-weight: 700; color: #fff;
  background: linear-gradient(135deg, {ACCENT}, {ACCENT_2});
}}
.rc-card.rc-top .rc-rank {{ animation: rc-glow 2.4s ease-in-out 1s infinite; }}
.rc-body {{ flex: 1; min-width: 0; }}
.rc-title {{ font-weight: 600; line-height: 1.3; }}
.rc-sku {{ font-family: ui-monospace, monospace; font-size: .78em; opacity: .7; margin-left: 6px; }}
.rc-meter {{ height: 7px; border-radius: 99px; background: rgba(128, 128, 128, .18); margin: 10px 0 8px; overflow: hidden; }}
.rc-fill {{
  height: 100%; width: 0; border-radius: 99px;
  background: linear-gradient(90deg, {ACCENT}, {ACCENT_2});
  animation: rc-grow 1s cubic-bezier(.2, .8, .2, 1) forwards;
}}
.rc-stats {{ font-size: .86em; display: flex; flex-wrap: wrap; gap: 6px 14px; }}
.rc-stat b {{ font-variant-numeric: tabular-nums; }}
.rc-count {{ animation: rc-count 1.1s ease-out forwards; counter-reset: rc-p var(--rc-p); font-weight: 700; }}
.rc-count::after {{ content: counter(rc-p) '%'; }}
.rc-reason {{ font-size: .8em; opacity: .7; margin-top: 6px; }}
.rc-cart-item {{
  display: flex; align-items: center; gap: 10px; padding: 8px 12px; border-radius: 12px;
  background: rgba(128, 128, 128, .08);
}}
.rc-cart-item.rc-new {{ animation: rc-pop .45s cubic-bezier(.2, .8, .2, 1); border: 1px solid {ACCENT}88; }}
.rc-dot {{ width: 9px; height: 9px; border-radius: 50%; flex: none; background: linear-gradient(135deg, {ACCENT}, {ACCENT_2}); }}
.rc-cart-name {{ font-size: .92em; line-height: 1.25; }}
.rc-empty {{
  text-align: center; padding: 28px 18px; border-radius: 16px;
  border: 1px dashed rgba(128, 128, 128, .4);
  animation: rc-rise .5s ease forwards;
}}
.rc-empty .rc-icon {{ font-size: 2.2em; display: inline-block; animation: rc-float 3s ease-in-out infinite; }}
.rc-empty h4 {{ margin: 8px 0 4px; }}
.rc-empty p {{ opacity: .75; font-size: .9em; margin: 0; }}
.rc-skeleton {{
  height: 10px; border-radius: 99px; margin: 14px auto 0; width: 70%;
  background: linear-gradient(90deg, rgba(128,128,128,.10) 25%, rgba(128,128,128,.25) 50%, rgba(128,128,128,.10) 75%);
  background-size: 200% 100%; animation: rc-shimmer 1.8s linear infinite;
}}
@media (prefers-reduced-motion: reduce) {{
  .rc-card, .rc-fill, .rc-count, .rc-cart-item, .rc-empty, .rc-empty .rc-icon, .rc-skeleton, .rc-rank {{ animation: none !important; opacity: 1; }}
  .rc-fill {{ width: var(--w); }}
  .rc-count {{ --rc-p: var(--p); }}
}}
</style>
"""


def recommendation_card(rec: dict, index: int, top_score: float) -> str:
    width = 100 * rec["score"] / top_score if top_score > 0 else 0
    delay = index * 110
    pct = round(rec["confidence"] * 100)
    top = " rc-top" if index == 0 else ""
    return f"""
<div class="rc-card{top}" style="animation-delay:{delay}ms">
  <div class="rc-rank">{rec['rank']}</div>
  <div class="rc-body">
    <div class="rc-title">{escape(rec['name'])}<span class="rc-sku">{escape(rec['product_id'])}</span></div>
    <div class="rc-meter" title="Ranking score relative to the top suggestion">
      <div class="rc-fill" style="--w:{width:.1f}%;animation-delay:{delay + 150}ms"></div>
    </div>
    <div class="rc-stats">
      <span class="rc-stat"><span class="rc-count" style="--p:{pct};animation-delay:{delay + 150}ms"></span> confidence</span>
      <span class="rc-stat">lift <b>{rec['lift']:.2f}×</b></span>
      <span class="rc-stat"><b>{rec['evidence_count']}</b> of {rec['antecedent_count']} baskets</span>
    </div>
    <div class="rc-reason">{escape(rec['reason'])}</div>
  </div>
</div>"""


def cart_item(name: str, sku: str, is_new: bool) -> str:
    new = " rc-new" if is_new else ""
    return (
        f'<div class="rc-cart-item{new}"><span class="rc-dot"></span>'
        f'<span class="rc-cart-name">{escape(name)}<br><span class="rc-sku" style="margin:0">{escape(sku)}</span></span></div>'
    )


def empty_state(icon: str, title: str, message: str, shimmer: bool = False) -> str:
    skeleton = '<div class="rc-skeleton"></div><div class="rc-skeleton" style="width:50%"></div>' if shimmer else ""
    return (
        f'<div class="rc-empty"><span class="rc-icon">{icon}</span>'
        f"<h4>{escape(title)}</h4><p>{escape(message)}</p>{skeleton}</div>"
    )


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def copurchase_graph(cart: list[tuple[str, str]], recs: list[dict], dark: bool) -> tuple[str, int]:
    """SVG of cart → hub → suggestions. Edge width follows confidence; dots travel the
    edges. Returns (html, height) for st.components.v1.html."""
    text = "#e8e6f3" if dark else "#1f1d2b"
    muted = "#a19fb3" if dark else "#6b6880"
    node_bg = "#1d1b2a" if dark else "#ffffff"
    width = 640
    rows = max(len(cart), len(recs), 1)
    height = max(220, 46 * rows + 50)
    hub_x, hub_y = 238, height / 2
    cart_x, rec_x = 4, 390

    def ys(n: int) -> list[float]:
        gap = min(52.0, (height - 40) / max(n, 1))
        start = height / 2 - gap * (n - 1) / 2
        return [start + i * gap for i in range(n)]

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="100%" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg" '
        f'font-family="-apple-system, Segoe UI, Roboto, sans-serif">',
        "<defs>"
        f'<linearGradient id="g" x1="0" x2="1"><stop offset="0" stop-color="{ACCENT}"/><stop offset="1" stop-color="{ACCENT_2}"/></linearGradient>'
        f'<radialGradient id="hub"><stop offset="0" stop-color="{ACCENT}"/><stop offset="1" stop-color="{ACCENT_2}"/></radialGradient>'
        "</defs>",
    ]

    for (name, sku), y in zip(cart, ys(len(cart))):
        path = f"M{cart_x + 170},{y} C{hub_x - 60},{y} {hub_x - 80},{hub_y} {hub_x - 26},{hub_y}"
        parts.append(f'<path d="{path}" fill="none" stroke="{muted}" stroke-opacity=".45" stroke-width="1.5"/>')
        parts.append(
            f'<rect x="{cart_x}" y="{y - 16}" width="170" height="32" rx="10" fill="{node_bg}" stroke="{muted}" stroke-opacity=".5"/>'
            f'<text x="{cart_x + 12}" y="{y + 4}" font-size="13.5" fill="{text}">{escape(_truncate(name, 21))}</text>'
        )

    max_conf = max((r["confidence"] for r in recs), default=1) or 1
    for i, (rec, y) in enumerate(zip(recs, ys(len(recs)))):
        begin = 0.25 + i * 0.12
        stroke = 2 + 7 * rec["confidence"] / max_conf
        path = f"M{hub_x + 26},{hub_y} C{hub_x + 90},{hub_y} {rec_x - 90},{y} {rec_x},{y}"
        parts.append(
            f'<path id="e{i}" d="{path}" fill="none" stroke="url(#g)" stroke-linecap="round" stroke-width="{stroke:.1f}" '
            f'stroke-opacity=".75" stroke-dasharray="400" stroke-dashoffset="400">'
            f'<animate attributeName="stroke-dashoffset" from="400" to="0" dur=".9s" begin="{begin:.2f}s" fill="freeze"/></path>'
        )
        parts.append(
            f'<circle r="3.5" fill="{ACCENT_2}" opacity="0">'
            f'<animate attributeName="opacity" from="0" to="1" begin="{begin + 0.9:.2f}s" dur=".2s" fill="freeze"/>'
            f'<animateMotion dur="{2.4 - 0.9 * rec["confidence"] / max_conf:.2f}s" begin="{begin + 0.9:.2f}s" repeatCount="indefinite">'
            f'<mpath href="#e{i}"/></animateMotion></circle>'
        )
        label = f"{escape(_truncate(rec['name'], 20))}"
        stats = f"{rec['confidence']:.0%} · lift {rec['lift']:.2f}"
        parts.append(
            f'<g opacity="0"><animate attributeName="opacity" from="0" to="1" begin="{begin + 0.6:.2f}s" dur=".4s" fill="freeze"/>'
            f'<circle cx="{rec_x}" cy="{y}" r="7" fill="url(#g)"/>'
            f'<text x="{rec_x + 16}" y="{y - 2}" font-size="14.5" font-weight="600" fill="{text}">{rec["rank"]}. {label}</text>'
            f'<text x="{rec_x + 16}" y="{y + 13}" font-size="12.5" fill="{muted}">{stats}</text></g>'
        )

    parts.append(
        f'<circle cx="{hub_x}" cy="{hub_y}" r="26" fill="url(#hub)">'
        f'<animate attributeName="r" values="24;27;24" dur="2.6s" repeatCount="indefinite"/></circle>'
        f'<text x="{hub_x}" y="{hub_y + 5}" text-anchor="middle" font-size="16" fill="#fff">🛒</text>'
        f'<text x="{hub_x}" y="{hub_y + 46}" text-anchor="middle" font-size="12.5" fill="{muted}">your cart</text>'
        "</svg>"
    )
    page = f'<div style="margin:0;height:{round(height * GRAPH_SCALE)}px">{"".join(parts)}</div>'
    return page, round(height * GRAPH_SCALE) + 8
