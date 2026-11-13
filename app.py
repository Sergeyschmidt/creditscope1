"""CreditScope - Kreditrisiko-Scoring mit erklärbarem ML, Szenarien, Portfolio-Sicht und Claude-Assistent."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import viz  # noqa: E402
from analysis import (SCENARIOS, Blend, apply_shocks, feature_curve, heatmap_grid, implied_cost_of_debt,  # noqa: E402
                      percentile_better, predict_many, rating_breaks, revenue_curve, run_scenarios,
                      score_portfolio)
from features import (FEATURE_BETTER, FEATURE_FORMULA, FEATURE_INFO, FEATURES, RAW_COLUMNS,  # noqa: E402
                      altman_z, compute_features, pd_to_rating)
from llm import extract_financials, get_client, get_model, write_memo  # noqa: E402

st.set_page_config(page_title="CreditScope", page_icon="📉", layout="wide")


@st.cache_resource
def load():
    bundle = joblib.load(ROOT / "models" / "gb_model.joblib")
    lrb = joblib.load(ROOT / "models" / "lr_model.joblib")
    bg = pd.read_csv(ROOT / "models" / "background.csv")
    metrics = json.loads((ROOT / "models" / "metrics.json").read_text())
    preds = pd.read_csv(ROOT / "models" / "test_preds.csv")
    panel = pd.read_csv(ROOT / "data" / "panel.csv")
    pop = compute_features(panel)
    return bundle["model"], lrb, shap.TreeExplainer(bundle["model"], bg), metrics, preds, panel, pop


gb, lrb, explainer, metrics, preds, panel, pop = load()
LABELS_F = [FEATURE_INFO[f][0] for f in FEATURES]
LABEL2KEY = {FEATURE_INFO[f][0]: f for f in FEATURES}

PRESETS = {
    "Solides Unternehmen": dict(revenue=5000, ebit=650, ebitda=900, net_income=450, interest_expense=60,
                                total_assets=6000, total_liabilities=3000, total_debt=1200, cash=700,
                                current_assets=2200, current_liabilities=1300, retained_earnings=1500,
                                market_cap=7000, operating_cashflow=800),
    "Angeschlagenes Unternehmen": dict(revenue=2000, ebit=40, ebitda=140, net_income=-60, interest_expense=95,
                                       total_assets=3000, total_liabilities=2700, total_debt=1900, cash=90,
                                       current_assets=700, current_liabilities=1100, retained_earnings=-250,
                                       market_cap=400, operating_cashflow=60),
}
LABELS = {
    "revenue": "Umsatz", "ebit": "EBIT", "ebitda": "EBITDA", "net_income": "Jahresüberschuss",
    "interest_expense": "Zinsaufwand", "total_assets": "Bilanzsumme", "total_liabilities": "Verbindlichkeiten gesamt",
    "total_debt": "Finanzschulden", "cash": "Zahlungsmittel", "current_assets": "Umlaufvermögen",
    "current_liabilities": "Kurzfr. Verbindlichkeiten", "retained_earnings": "Gewinnrücklagen",
    "market_cap": "Marktkapitalisierung", "operating_cashflow": "Operativer Cashflow",
}


# ---------- Zustand und Callbacks ----------
def apply_preset():
    for c in RAW_COLUMNS:
        st.session_state[f"in-{c}"] = float(PRESETS[st.session_state["preset"]][c])


def pick_tile(label):
    st.session_state["detail_feat"] = label


def pick_chart():
    try:
        pts = st.session_state["drivers_chart"]["selection"]["points"]
    except Exception:  # noqa: BLE001
        pts = []
    if pts and pts[0].get("y") in LABEL2KEY:
        st.session_state["detail_feat"] = pts[0]["y"]


if "in-revenue" not in st.session_state:
    st.session_state["preset"] = list(PRESETS)[0]
    apply_preset()
if "detail_feat" not in st.session_state:
    st.session_state["detail_feat"] = FEATURE_INFO["interest_coverage"][0]
pending = st.session_state.pop("pending_fill", None)  # Werte aus der KI-Auslesung übernehmen
if pending:
    for c in RAW_COLUMNS:
        if pending.get(c) is not None:
            st.session_state[f"in-{c}"] = float(pending[c])

# ---------- Seitenleiste: Annahmen ----------
with st.sidebar:
    st.header("Modell")
    MODE_LABEL = st.radio("Scoring-Modell", ["Ensemble (Ø beider Modelle)", "Gradient Boosting", "Logistische Regression"],
                          help="Gradient Boosting rechnet stufenförmig, die logistische Regression glatt. "
                               "Das Ensemble reagiert in Szenarien am aussagekräftigsten.")
    MODE = {"Ensemble (Ø beider Modelle)": "ens", "Gradient Boosting": "gb", "Logistische Regression": "lr"}[MODE_LABEL]
    st.header("Annahmen")
    lgd = st.slider("Verlustquote bei Ausfall (LGD)", 10, 90, 45, 5, help="Anteil des Kredits, der im Ausfall verloren geht") / 100
    rf = st.slider("Risikoloser Zins (%)", 0.0, 6.0, 3.0, 0.25) / 100
    cm = st.slider("Deckungsbeitragsquote (Stresstest)", 10, 70, 35, 5, help="Wie stark Umsatzänderungen auf das EBIT durchschlagen") / 100
    tax = st.slider("Steuersatz (Stresstest)", 0, 40, 25, 5) / 100
    st.caption("Diese Annahmen wirken auf Szenarien, Portfolio und Fremdkapitalkosten.")


model = Blend(gb, lrb, MODE)


def score(raw: pd.DataFrame):
    X = compute_features(raw)
    return X, model.predict_proba(X)[:, 1]


st.title("📉 CreditScope")
st.caption("Kreditrisiko-Scoring mit erklärbarem Machine Learning: Wahrscheinlichkeit einer Notlage im Folgejahr")
if metrics["data_source"] == "synthetic":
    st.warning("Das Modell wurde mit **synthetischen Demo-Daten** trainiert. Für belastbare Ergebnisse "
               "echte Daten mit `src/data_fetch.py` laden und neu trainieren.")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Analyse", "Szenarien & Stress", "Portfolio", "Modellqualität", "KI-Assistent (Claude)", "Methodik"])

# =====================================================================================
with tab1:
    left, right = st.columns([1, 1.5])
    with left:
        st.selectbox("Startwerte", list(PRESETS), key="preset", on_change=apply_preset)
        vals = {}
        st.markdown("**Jahresabschluss (Mio.)**")
        cols = st.columns(2)
        for i, c in enumerate(RAW_COLUMNS):
            vals[c] = cols[i % 2].number_input(LABELS[c], key=f"in-{c}", step=10.0)
    raw = pd.DataFrame([vals])
    X, p = score(raw)
    pdv = float(p[0])
    rating = pd_to_rating(pdv)
    z = float(altman_z(raw).iloc[0])
    zone = "sicher" if z > 2.99 else "Grauzone" if z > 1.81 else "Gefahrenzone"
    sv = explainer(X)
    lr_c = lrb["lr"].coef_[0] * (X[FEATURES].values - lrb["scaler"].mean_) / lrb["scaler"].scale_
    drv = {"gb": sv.values, "lr": lr_c, "ens": (sv.values + lr_c) / 2}[MODE]

    with right:
        m1, m2, m3 = st.columns(3)
        m1.metric("Ausfallwahrscheinlichkeit", f"{pdv:.1%}")
        m2.metric("Rating (intern)", rating)
        m3.metric("Altman Z (Vergleich)", f"{z:.2f}", zone, delta_color="off")
        st.plotly_chart(viz.drivers_bar(LABELS_F, drv[0]), key="drivers_chart", on_select=pick_chart,
                        selection_mode="points", width="stretch")

    st.markdown("#### Kennzahlen: Zahl anklicken, um Details und Grafiken zu sehen")
    tc = st.columns(3)
    for i, f in enumerate(FEATURES):
        lab = FEATURE_INFO[f][0]
        tc[i % 3].button(f"{lab}: {X[f].iloc[0]:.2f}", key=f"tile-{f}", on_click=pick_tile, args=(lab,),
                         type="primary" if st.session_state["detail_feat"] == lab else "secondary", width="stretch")

    st.selectbox("Gewählte Kennzahl", LABELS_F, key="detail_feat")
    fk = LABEL2KEY[st.session_state["detail_feat"]]
    v = float(X[fk].iloc[0])
    popf = pop[fk]
    better = FEATURE_BETTER[fk]
    pct = percentile_better(popf, v, better)
    contrib = float(drv[0][FEATURES.index(fk)])
    med = float(popf.median())
    X_med = X.copy()
    X_med[fk] = med
    pd_med = float(model.predict_proba(X_med)[:, 1][0])

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Wert", f"{v:.2f}")
    d2.metric("Besser als Vergleichsfirmen", f"{pct:.0%}")
    d3.metric("Beitrag zum Risiko", f"{contrib:+.2f}", "erhöht Risiko" if contrib > 0 else "senkt Risiko", delta_color="inverse")
    d4.metric("PD bei Median-Wert", f"{pd_med:.1%}", f"{(pd_med - pdv) * 100:+.1f} pp", delta_color="inverse")
    level = "stark" if pct >= 0.66 else "schwach" if pct <= 0.33 else "durchschnittlich"
    st.caption(f"**Formel:** {FEATURE_FORMULA[fk]}. {FEATURE_INFO[fk][1]}. "
               f"Das Unternehmen ist bei dieser Kennzahl **{level}** (besser als {pct:.0%} der Stichprobe). "
               f"Läge der Wert beim Stichprobenmedian ({med:.2f}), wäre die PD {pd_med:.1%} statt {pdv:.1%}.")
    g1, g2 = st.columns(2)
    lo, hi = float(popf.quantile(0.01)), float(popf.quantile(0.99))
    g1.plotly_chart(viz.hist_marker(popf.clip(lo, hi), min(max(v, lo), hi), f"Verteilung: {LABELS_F[FEATURES.index(fk)]}"),
                    width="stretch", key="g-hist")
    cv = feature_curve(model, X, fk, float(popf.quantile(0.02)), float(popf.quantile(0.98)))
    g2.plotly_chart(viz.line_marker(cv["x"], cv["pd"] * 100, v, pdv * 100,
                                    "Wie reagiert die PD auf diese Kennzahl? (alles andere konstant)",
                                    LABELS_F[FEATURES.index(fk)]), width="stretch", key="g-curve")

# =====================================================================================
with tab2:
    st.subheader("Was passiert bei Umsatzrückgang oder Zinsanstieg?")
    st.caption("Basis ist das Unternehmen aus der Analyse. Annahmen (Deckungsbeitrag, Steuer) stehen in der Seitenleiste.")
    c1, c2 = st.columns([1, 2.2])
    with c1:
        rev = st.slider("Umsatzänderung (%)", -50, 20, -10) / 100
        rate = st.slider("Zinsanstieg (Prozentpunkte)", 0.0, 5.0, 1.0, 0.5)
    base_raw = {k: float(vals[k]) for k in RAW_COLUMNS}
    shocked = apply_shocks(base_raw, rev, rate, cm, tax)
    ps, Xs = predict_many(model, [shocked])
    ps = float(ps[0])
    with c2:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("PD im Szenario", f"{ps:.1%}", f"{(ps - pdv) * 100:+.1f} pp", delta_color="inverse")
        k2.metric("Rating", pd_to_rating(ps), f"vorher {rating}", delta_color="off")
        k3.metric("Zinsdeckung", f"{Xs['interest_coverage'].iloc[0]:.2f}", f"{Xs['interest_coverage'].iloc[0] - X['interest_coverage'].iloc[0]:+.2f}")
        k4.metric("Net Debt/EBITDA", f"{Xs['net_debt_to_ebitda'].iloc[0]:.2f}", f"{Xs['net_debt_to_ebitda'].iloc[0] - X['net_debt_to_ebitda'].iloc[0]:+.2f}", delta_color="inverse")

    curve = revenue_curve(model, base_raw, rate, cm, tax)
    base_pd_curve = float(curve.iloc[(curve["shock"]).abs().argmin()]["pd"])
    cc1, cc2 = st.columns(2)
    cc1.plotly_chart(viz.revenue_curve_fig(curve, rev, ps), width="stretch", key="s-curve")
    rv, rt, pg = heatmap_grid(model, base_raw, cm, tax)
    cc2.plotly_chart(viz.heatmap_fig(rv, rt, pg), width="stretch", key="s-heat")

    brk = rating_breaks(curve, base_pd_curve)
    if brk:
        txt = " | ".join(f"Rating **{lab}** ab Umsatz {s:+.0%}" if s is not None else f"Rating **{lab}**: im Bereich bis −50 % nicht erreicht"
                         for lab, s, _ in brk[:3])
        st.info(f"Kippunkte bei Zinsanstieg +{rate:.1f} pp: {txt}")

    st.markdown("#### Vordefinierte Szenarien")
    sc = run_scenarios(model, base_raw, SCENARIOS, cm, tax)
    sc1, sc2 = st.columns([1.1, 1])
    sc1.dataframe(sc.assign(PD=(sc["PD"] * 100).round(1).astype(str) + " %"), hide_index=True, width="stretch")
    sc2.plotly_chart(viz.scenario_bar(sc), width="stretch", key="s-bar")

    st.markdown("#### Folgen für die Fremdkapitalkosten")
    kd0, kd1 = implied_cost_of_debt(pdv, lgd, rf), implied_cost_of_debt(ps, lgd, rf)
    f1, f2, f3 = st.columns(3)
    f1.metric("Risikoaufschlag (Basis)", f"{pdv * lgd * 100:.2f} %")
    f2.metric("Fremdkapitalkosten (Basis)", f"{kd0 * 100:.2f} %")
    f3.metric("Fremdkapitalkosten (Szenario)", f"{kd1 * 100:.2f} %", f"{(kd1 - kd0) * 100:+.2f} pp", delta_color="inverse")
    st.caption("Vereinfachte Näherung: Risikoaufschlag ≈ PD × LGD, Fremdkapitalkosten = risikoloser Zins + Aufschlag. "
               "Das Ergebnis lässt sich als Fremdkapitalkostensatz in einer WACC-Rechnung weiterverwenden.")

# =====================================================================================
with tab3:
    st.subheader("Kreditportfolio")
    mode = st.radio("Datenquelle", ["Demo-Portfolio (30 Firmen)", "CSV hochladen"], horizontal=True)
    df_in = None
    if mode.startswith("Demo"):
        df_in = panel.sample(30, random_state=3).reset_index(drop=True)
        df_in["exposure"] = np.random.default_rng(3).uniform(5, 50, 30).round(1)
    else:
        st.write("Spalten: `company`, " + ", ".join(f"`{c}`" for c in RAW_COLUMNS) + ", optional `exposure` (Mio.)")
        up = st.file_uploader("CSV", type="csv", key="pf_up")
        if up:
            d = pd.read_csv(up)
            miss = [c for c in ["company"] + RAW_COLUMNS if c not in d.columns]
            if miss:
                st.error(f"Fehlende Spalten: {miss}")
            else:
                df_in = d
    if df_in is not None:
        pf = score_portfolio(model, df_in, lgd)
        tot = pf["exposure"].sum()
        wpd = float((pf["PD"] * pf["exposure"]).sum() / tot)
        el = float(pf["EL"].sum())
        risky = float(pf.loc[pf["Rating"].isin(["CCC", "D-Gefahr"]), "exposure"].sum() / tot)
        a, b, c, d, e = st.columns(5)
        a.metric("Gesamtexposure", f"{tot:,.0f} Mio.")
        b.metric("Ø PD (gewichtet)", f"{wpd:.1%}")
        c.metric("Expected Loss", f"{el:,.1f} Mio.")
        d.metric("EL-Quote", f"{el / tot:.2%}")
        e.metric("Anteil CCC/D", f"{risky:.0%}")
        p1, p2 = st.columns(2)
        p1.plotly_chart(viz.rating_dist(pf), width="stretch", key="p-rating")
        p2.plotly_chart(viz.pd_hist(pf["PD"].values), width="stretch", key="p-hist")
        p3, p4 = st.columns(2)
        p3.plotly_chart(viz.el_bar(pf), width="stretch", key="p-el")
        top5 = pf.sort_values("exposure", ascending=False).head(5)["exposure"].sum() / tot
        p4.metric("Konzentration: größte 5 Kredite", f"{top5:.0%} des Exposures")
        p4.caption("Expected Loss = PD × LGD × Exposure. Ein hoher Anteil weniger Kredite deutet auf Klumpenrisiko hin.")
        show = pf[["company", "exposure", "PD", "Rating", "EL"]].sort_values("EL", ascending=False)
        st.dataframe(show.assign(PD=(show["PD"] * 100).round(2)).rename(columns={"PD": "PD (%)", "exposure": "Exposure", "EL": "Expected Loss"}),
                     hide_index=True, width="stretch")
        st.download_button("Ergebnis als CSV", pf.to_csv(index=False).encode(), "creditscope_portfolio.csv")

# =====================================================================================
with tab4:
    st.subheader("Wie gut ist das Modell?")
    from sklearn.metrics import brier_score_loss, roc_auc_score
    y = preds["y"].values
    p_ens = ((preds["p_gb"] + preds["p_lr"]) / 2).values
    p_sel = {"gb": preds["p_gb"].values, "lr": preds["p_lr"].values, "ens": p_ens}[MODE]
    q1, q2, q3, q4, q5 = st.columns(5)
    q1.metric("AUC Gradient Boosting", f"{roc_auc_score(y, preds['p_gb']):.3f}")
    q2.metric("AUC Log. Regression", f"{roc_auc_score(y, preds['p_lr']):.3f}")
    q3.metric("AUC Ensemble", f"{roc_auc_score(y, p_ens):.3f}")
    q4.metric(f"Brier-Score ({MODE_LABEL.split(' ')[0]})", f"{brier_score_loss(y, p_sel):.3f}")
    q5.metric("Notlagen-Rate (Daten)", f"{metrics['base_rate']:.1%}")
    r1, r2 = st.columns(2)
    r1.plotly_chart(viz.roc_fig(y, preds["p_gb"].values, preds["p_lr"].values, p_ens), width="stretch", key="m-roc")
    r2.plotly_chart(viz.calib_fig(y, p_sel), width="stretch", key="m-cal")
    r3, r4 = st.columns(2)
    r3.plotly_chart(viz.importance_fig(LABELS_F, gb.feature_importances_), width="stretch", key="m-imp")
    r4.plotly_chart(viz.class_hist(y, p_sel), width="stretch", key="m-cls")
    st.markdown("#### Schwellenwert: ab welcher PD schlagen wir Alarm?")
    thr = st.slider("Alarm ab PD (%)", 1, 50, 10) / 100
    pr = p_sel >= thr
    tp, fp = int(((pr) & (y == 1)).sum()), int(((pr) & (y == 0)).sum())
    fn, tn = int(((~pr) & (y == 1)).sum()), int(((~pr) & (y == 0)).sum())
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Erkannte Notlagen", f"{tp} von {tp + fn}", f"Trefferquote {tp / max(tp + fn, 1):.0%}", delta_color="off")
    t2.metric("Fehlalarme", f"{fp}", f"Präzision {tp / max(tp + fp, 1):.0%}", delta_color="off")
    t3.metric("Übersehene Notlagen", f"{fn}")
    t4.metric("Korrekt als gesund", f"{tn}")
    st.caption("Niedriger Schwellenwert: weniger Übersehenes, mehr Fehlalarme. Die Wahl hängt davon ab, was ein Ausfall im Vergleich zu einer unnötigen Prüfung kostet.")

# =====================================================================================
with tab5:
    client = get_client()
    if client is None:
        st.info("Kein API-Schlüssel gefunden. Lokal: Umgebungsvariable `ANTHROPIC_API_KEY` setzen. "
                "Auf Streamlit Cloud: App-Einstellungen → Secrets → `ANTHROPIC_API_KEY = \"...\"`.")

    st.subheader("1) Zahlen aus einem Geschäftsbericht auslesen")
    st.caption("Text aus dem Abschlussteil (GuV, Bilanz, Cashflow) einfügen oder eine PDF hochladen. "
               "Claude trägt die Werte in die Analyse ein. Bitte immer gegenprüfen.")
    pdf = st.file_uploader("PDF (optional)", type="pdf")
    txt = st.text_area("Berichtstext", height=160, key="report_text")
    if st.button("Zahlen auslesen", disabled=client is None):
        text = txt
        if pdf is not None:
            from pypdf import PdfReader
            text += "\n".join((pg.extract_text() or "") for pg in PdfReader(pdf).pages)
        if len(text.strip()) < 200:
            st.error("Zu wenig Text. Bitte den Abschlussteil des Berichts einfügen.")
        else:
            try:
                with st.spinner("Claude liest den Bericht ..."):
                    res = extract_financials(text, client, get_model())
                st.session_state["extract_result"] = res
                st.session_state["pending_fill"] = {c: res.get(c) for c in RAW_COLUMNS}
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"Auslesen fehlgeschlagen: {e}")
    res = st.session_state.get("extract_result")
    if res:
        missing = [LABELS[c] for c in RAW_COLUMNS if res.get(c) is None]
        st.success("Werte wurden in die Analyse übernommen.")
        if missing:
            st.warning("Nicht gefunden (bitte selbst ergänzen): " + ", ".join(missing))
        st.write("**Herkunft der Werte:** " + str(res.get("evidence", "")))

    st.divider()
    st.subheader("2) Kreditmemo aus dem aktuellen Ergebnis")
    st.caption("Claude formuliert nur auf Basis der Modellzahlen, ohne eigene Fakten. Szenario-Ergebnisse fließen mit ein.")
    if st.button("Kreditmemo erstellen", disabled=client is None):
        drivers = sorted(zip(LABELS_F, drv[0]), key=lambda t: abs(t[1]), reverse=True)
        ctx = dict(pd=pdv, rating=rating, z=z, zone=zone,
                   features={FEATURE_INFO[f][0]: round(float(X[f].iloc[0]), 3) for f in FEATURES},
                   drivers=[(n, float(v_)) for n, v_ in drivers],
                   limits=f"Datenbasis: {metrics['data_source']}, Proxy-Zielgröße statt echter Ausfälle, "
                          "wenige Jahre pro Firma, keine Branchenvariablen.")
        sc_txt = "; ".join(f"{r['Szenario']}: PD {r['PD']:.1%} ({r['Rating']})" for _, r in sc.iterrows())
        ctx["limits"] += f" Stresstest-Ergebnisse zur Einordnung: {sc_txt}."
        try:
            with st.spinner("Claude schreibt das Memo ..."):
                st.session_state["memo"] = write_memo(ctx, client, get_model())
        except Exception as e:  # noqa: BLE001
            st.error(f"Memo fehlgeschlagen: {e}")
    if st.session_state.get("memo"):
        st.markdown(st.session_state["memo"])
        st.download_button("Memo als Textdatei", st.session_state["memo"].encode(), "kreditmemo.md")

# =====================================================================================
with tab6:
    st.subheader("Methodik und Grenzen")
    st.markdown(f"""
**Fragestellung:** Wie wahrscheinlich gerät ein Unternehmen im Folgejahr in Zahlungsnot?

**Zielgröße (Proxy):** Zinsdeckung unter 1 oder negatives Eigenkapital im Folgejahr. Features stammen aus Jahr *t*, das Label aus Jahr *t+1*, damit keine Zukunftsinformation einfließt.

**Daten:** {metrics['n_train'] + metrics['n_test']} Beobachtungen ({'synthetisch' if metrics['data_source'] == 'synthetic' else 'reale Unternehmensdaten'}).
Aufteilung nach Unternehmen (Train {metrics['n_train']} / Test {metrics['n_test']}), damit keine Firma in beiden Teilen steckt.

**Modelle:** Gradient Boosting (Erklärung mit SHAP), logistische Regression (glatt und transparent) und das Ensemble aus beiden (Standard). Die Treiber-Grafik zeigt den Durchschnitt der Beiträge beider Modelle.

**Kennzahlen:** {', '.join(LABELS_F)}.

**Stresstest:** Umsatzänderung wirkt über die Deckungsbeitragsquote auf EBIT, EBITDA und Cashflow. Der Zinsanstieg trifft alle Finanzschulden. Ergebnisänderungen nach Steuern verändern Rücklagen und Eigenkapital. Das ist eine bewusste Vereinfachung.

**Expected Loss:** PD × LGD × Exposure. Die LGD ist eine Annahme der Seitenleiste, kein geschätzter Wert.

**Grenzen:** Proxy statt echter Kreditausfälle, wenige Jahre pro Firma, keine Branchen- oder Makrovariablen, Altman Z nur als Vergleich. Die Ergebnisse sind kein Ersatz für eine Bonitätsprüfung.
""")
