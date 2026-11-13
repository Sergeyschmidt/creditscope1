"""Optik: CSS, Kopfbereich und Kennzahlen-Karten für CreditScope."""
import streamlit as st

RATING_COLORS = {"A": "#1e8e5a", "BBB": "#5fa043", "BB": "#c98a1b", "B": "#dc7a33", "CCC": "#cf4a3a", "D-Gefahr": "#9b1c1c"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root{--ink:#0e1b2c;--muted:#5b6b7f;--line:#e3e8ee;--card:#ffffff;--bg:#f3f5f8;--accent:#0d5c63;--good:#1e8e5a;--warn:#c98a1b;--bad:#d64545;}
html, body, .stApp, [class*="st-"] {font-family:'DM Sans',system-ui,sans-serif;}
.stApp{background:var(--bg);color:var(--ink);}
.block-container{padding-top:1.4rem;padding-bottom:3rem;max-width:1280px;}
h1,h2,h3,h4{letter-spacing:-.015em;font-weight:700;color:var(--ink);}
h4{font-size:1.05rem;margin-top:.6rem;}
section[data-testid="stSidebar"]{background:#ffffff;border-right:1px solid var(--line);}
section[data-testid="stSidebar"] h2{font-size:.78rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);font-weight:600;margin-top:.5rem;}

/* Kopfbereich */
.hero{background:linear-gradient(120deg,#0e1b2c 0%,#12384a 55%,#0d5c63 100%);border-radius:20px;padding:28px 34px;margin-bottom:18px;
  display:flex;justify-content:space-between;align-items:center;gap:24px;flex-wrap:wrap;color:#fff;box-shadow:0 8px 24px rgba(14,27,44,.14);}
.hero .eyebrow{font-size:.74rem;letter-spacing:.14em;text-transform:uppercase;color:#8fd3d9;font-weight:600;}
.hero h1{color:#fff;font-size:2.5rem;margin:.1rem 0 .3rem;line-height:1.05;}
.hero p{color:#c9d6e2;margin:0;max-width:60ch;font-size:1rem;}
.hero .pills{display:flex;gap:8px;flex-wrap:wrap;}
.pill{display:inline-block;padding:5px 12px;border-radius:999px;font-size:.78rem;font-weight:500;background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.22);color:#e8f1f7;}
.pill.warn{background:rgba(255,196,87,.16);border-color:rgba(255,196,87,.5);color:#ffe2a8;}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{gap:4px;border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{height:44px;padding:0 16px;font-weight:600;color:var(--muted);}
.stTabs [aria-selected="true"]{color:var(--accent);}
.stTabs [data-baseweb="tab-highlight"]{background:var(--accent);height:3px;border-radius:3px;}

/* Standard-Kennzahlen als Karten */
div[data-testid="stMetric"]{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 16px;box-shadow:0 1px 2px rgba(14,27,44,.05);}
div[data-testid="stMetricLabel"] p{color:var(--muted);font-size:.82rem;font-weight:500;}
div[data-testid="stMetricValue"]{font-family:'IBM Plex Mono',monospace;font-weight:500;}

/* Karten */
div[data-testid="stVerticalBlockBorderWrapper"]{border-radius:16px;}
div[data-testid="stPlotlyChart"]{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:6px 6px 0;box-shadow:0 1px 2px rgba(14,27,44,.05);}
div[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;overflow:hidden;}

/* Eingaben und Buttons */
.stNumberInput input,.stTextInput input,.stTextArea textarea{border-radius:10px;font-family:'IBM Plex Mono',monospace;}
.stButton>button,.stDownloadButton>button{border-radius:12px;border:1px solid var(--line);background:#fff;color:var(--ink);font-weight:500;padding:.55rem .8rem;transition:all .15s;}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--accent);color:var(--accent);box-shadow:0 2px 8px rgba(13,92,99,.12);}
.stButton>button[kind="primary"]{background:var(--accent);border-color:var(--accent);color:#fff;}
.stButton>button[kind="primary"]:hover{color:#fff;filter:brightness(1.08);}

/* Kennzahlen-Karten oben */
.kpis{display:grid;grid-template-columns:1.6fr 1fr 1fr;gap:14px;margin-bottom:12px;}
@media(max-width:900px){.kpis{grid-template-columns:1fr;}}
.kcard{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px 20px;box-shadow:0 1px 2px rgba(14,27,44,.05);}
.kcard .lab{font-size:.78rem;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.06em;}
.kcard .big{font-family:'IBM Plex Mono',monospace;font-size:2.6rem;font-weight:500;line-height:1.15;margin-top:4px;}
.kcard .sub{color:var(--muted);font-size:.88rem;}
.scale{height:10px;border-radius:6px;background:linear-gradient(90deg,#1e8e5a 0%,#c98a1b 45%,#d64545 80%,#9b1c1c 100%);position:relative;margin:16px 0 6px;}
.scale i{position:absolute;top:-5px;width:5px;height:20px;border-radius:3px;background:var(--ink);transform:translateX(-2px);box-shadow:0 0 0 2px #fff;}
.ticks{display:flex;justify-content:space-between;font-size:.7rem;color:var(--muted);font-family:'IBM Plex Mono',monospace;}
.badge{display:inline-flex;align-items:center;justify-content:center;min-width:64px;height:64px;padding:0 14px;border-radius:18px;color:#fff;font-size:1.7rem;font-weight:700;margin-top:6px;}
.zone{display:inline-block;margin-top:6px;padding:3px 12px;border-radius:999px;font-size:.8rem;font-weight:600;}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(subtitle: str, pills: list, warn_pill: str | None = None):
    chips = "".join(f'<span class="pill">{p}</span>' for p in pills)
    if warn_pill:
        chips += f'<span class="pill warn">{warn_pill}</span>'
    st.markdown(
        f'<div class="hero"><div><div class="eyebrow">Kreditrisiko-Analyse</div><h1>CreditScope</h1>'
        f'<p>{subtitle}</p></div><div class="pills">{chips}</div></div>', unsafe_allow_html=True)


def _pos(p: float) -> float:
    stops = [(0, 0), (0.05, 25), (0.10, 50), (0.20, 75), (0.40, 100)]
    for (a, pa), (b, pb) in zip(stops, stops[1:]):
        if p <= b:
            return pa + (p - a) / (b - a) * (pb - pa)
    return 100.0


def kpi_cards(pdv: float, rating: str, z: float, zone: str):
    zc = {"sicher": ("#e3f4ea", "#1e8e5a"), "Grauzone": ("#fbf0d9", "#a8700f"), "Gefahrenzone": ("#fbe4e1", "#c23a2b")}[zone]
    risk = "geringes Risiko" if pdv < 0.015 else "erhöhtes Risiko" if pdv < 0.10 else "hohes Risiko"
    col = RATING_COLORS.get(rating, "#555")
    st.markdown(
        f'''<div class="kpis">
<div class="kcard"><div class="lab">Ausfallwahrscheinlichkeit</div>
<div class="big" style="color:{col}">{pdv:.1%}</div><div class="sub">Notlage im Folgejahr</div>
<div class="scale"><i style="left:{_pos(pdv):.1f}%"></i></div>
<div class="ticks"><span>0 %</span><span>5 %</span><span>10 %</span><span>20 %</span><span>40 %+</span></div></div>
<div class="kcard"><div class="lab">Rating (intern)</div>
<div class="badge" style="background:{col}">{rating}</div><div class="sub" style="margin-top:8px">{risk}</div></div>
<div class="kcard"><div class="lab">Altman Z (Vergleich)</div>
<div class="big">{z:.2f}</div><span class="zone" style="background:{zc[0]};color:{zc[1]}">{zone}</span></div>
</div>''', unsafe_allow_html=True)
