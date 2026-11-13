"""Analysefunktionen: Stresstests, Sensitivitäten, Portfolio-Kennzahlen.

Alle Stresstests sind bewusst einfach und transparent (Annahmen stehen in der App):
- Umsatzänderung wirkt über eine Deckungsbeitragsquote auf EBIT/EBITDA/Cashflow.
- Zinsanstieg trifft die Finanzschulden vollständig (variabel verzinst, Worst Case).
- Ergebnisänderung nach Steuern verändert Gewinnrücklagen; Verluste werden über
  zusätzliche Verbindlichkeiten finanziert, Gewinne erhöhen Zahlungsmittel.
"""
import numpy as np
import pandas as pd

from features import FEATURES, compute_features, pd_to_rating

RATING_THRESHOLDS = [(0.005, "A"), (0.015, "BBB"), (0.04, "BB"), (0.10, "B"), (0.25, "CCC")]

SCENARIOS = {
    "Basis": (0.0, 0.0),
    "Leichte Rezession": (-0.10, 1.0),
    "Schwere Rezession": (-0.25, 2.0),
    "Zinsschock": (0.0, 3.0),
    "Umsatzeinbruch": (-0.20, 0.0),
    "Aufschwung": (0.10, 0.0),
}


def apply_shocks(raw: dict, revenue_pct=0.0, rate_pp=0.0, cm=0.35, tax=0.25) -> dict:
    r = dict(raw)
    d_rev = raw["revenue"] * revenue_pct
    d_ebit = d_rev * cm
    d_int = raw["total_debt"] * rate_pp / 100.0
    r["revenue"] = raw["revenue"] + d_rev
    r["ebit"] = raw["ebit"] + d_ebit
    r["ebitda"] = raw["ebitda"] + d_ebit
    r["operating_cashflow"] = raw["operating_cashflow"] + d_ebit - d_int
    r["interest_expense"] = raw["interest_expense"] + d_int
    d_ni = (d_ebit - d_int) * (1 - tax)
    r["net_income"] = raw["net_income"] + d_ni
    r["retained_earnings"] = raw["retained_earnings"] + d_ni
    if d_ni < 0:
        r["total_liabilities"] = raw["total_liabilities"] - d_ni
    else:
        r["total_assets"] = raw["total_assets"] + d_ni
        r["cash"] = raw["cash"] + d_ni
    return r


def predict_many(model, raws: list):
    """PD und Features für mehrere Roh-Datensätze in einem Aufruf."""
    df = pd.DataFrame(raws)
    X = compute_features(df)
    return model.predict_proba(X)[:, 1], X


def run_scenarios(model, raw, scenarios=SCENARIOS, cm=0.35, tax=0.25) -> pd.DataFrame:
    names = list(scenarios)
    raws = [apply_shocks(raw, *scenarios[n], cm=cm, tax=tax) for n in names]
    p, X = predict_many(model, raws)
    return pd.DataFrame({
        "Szenario": names,
        "Umsatz": [f"{scenarios[n][0]:+.0%}" for n in names],
        "Zins": [f"+{scenarios[n][1]:.1f} pp" for n in names],
        "PD": p,
        "Rating": [pd_to_rating(x) for x in p],
        "Zinsdeckung": X["interest_coverage"].round(2).values,
        "Net Debt/EBITDA": X["net_debt_to_ebitda"].round(2).values,
    })


def revenue_curve(model, raw, rate_pp=0.0, cm=0.35, tax=0.25, lo=-0.5, hi=0.2, n=71) -> pd.DataFrame:
    shocks = np.linspace(lo, hi, n)
    p, _ = predict_many(model, [apply_shocks(raw, s, rate_pp, cm, tax) for s in shocks])
    return pd.DataFrame({"shock": shocks, "pd": p})


def rating_breaks(curve: pd.DataFrame, base_pd: float):
    """Ab welchem Umsatzrückgang kippt das Rating in die nächste Stufe?"""
    out = []
    neg = curve[curve["shock"] <= 0].sort_values("shock", ascending=False)
    for thr, label in RATING_THRESHOLDS:
        if base_pd >= thr:
            continue
        hit = neg[neg["pd"] >= thr]
        out.append((label_after(label), float(hit["shock"].iloc[0]) if len(hit) else None, thr))
    return out


def label_after(label: str) -> str:
    order = ["A", "BBB", "BB", "B", "CCC", "D-Gefahr"]
    return order[order.index(label) + 1]


def heatmap_grid(model, raw, cm=0.35, tax=0.25):
    rev = np.round(np.linspace(-0.40, 0.10, 11), 2)
    rate = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    raws = [apply_shocks(raw, r, k, cm, tax) for k in rate for r in rev]
    p, _ = predict_many(model, raws)
    return rev, rate, p.reshape(len(rate), len(rev))


def feature_curve(model, X_row: pd.DataFrame, feature: str, lo: float, hi: float, n=60) -> pd.DataFrame:
    grid = np.linspace(lo, hi, n)
    rows = pd.concat([X_row[FEATURES]] * n, ignore_index=True)
    rows[feature] = grid
    return pd.DataFrame({"x": grid, "pd": model.predict_proba(rows)[:, 1]})


def percentile_better(pop: pd.Series, value: float, better: str) -> float:
    """Anteil der Stichprobe, der schlechter ist als der Wert (0..1)."""
    return float((pop > value).mean() if better == "low" else (pop < value).mean())


def implied_cost_of_debt(pd_value: float, lgd: float, rf: float) -> float:
    """Vereinfachtes 'Credit Triangle': Spread ≈ PD x LGD. Grobe Näherung, keine Marktbewertung."""
    return rf + pd_value * lgd


def score_portfolio(model, df: pd.DataFrame, lgd: float) -> pd.DataFrame:
    X = compute_features(df)
    p = model.predict_proba(X)[:, 1]
    out = df.copy()
    out["PD"] = p
    out["Rating"] = [pd_to_rating(x) for x in p]
    if "exposure" not in out:
        out["exposure"] = 100.0
    out["EL"] = out["PD"] * lgd * out["exposure"]
    return out


class Blend:
    """Einheitliche Schnittstelle für Gradient Boosting, Logistische Regression oder deren Mittelwert.

    Gradient Boosting rechnet stufenförmig und reagiert in 'gesunden' Bereichen kaum auf kleine
    Änderungen. Die logistische Regression ist glatt. Der Mittelwert ist für Szenarien am
    aussagekräftigsten.
    """

    def __init__(self, gb, lr_bundle, mode="ens", train_rate=None, target_rate=None):
        self.gb, self.sc, self.lr, self.mode = gb, lr_bundle["scaler"], lr_bundle["lr"], mode
        self.train_rate, self.target_rate = train_rate, target_rate

    def predict_proba(self, X):
        X = X[FEATURES]
        p_gb = self.gb.predict_proba(X)[:, 1]
        p_lr = self.lr.predict_proba(self.sc.transform(X))[:, 1]
        p = {"gb": p_gb, "lr": p_lr, "ens": (p_gb + p_lr) / 2}[self.mode]
        if self.train_rate and self.target_rate:
            p = recalibrate(p, self.train_rate, self.target_rate)
        return np.column_stack([1 - p, p])


def recalibrate(p, train_rate, target_rate):
    """Prior-Korrektur: rechnet PDs von der Notlagenrate der Trainingsdaten auf eine realistische Rate um.

    Die Trainingsdaten enthalten absichtlich viele angeschlagene Firmen (Notlagenrate ca. 20 %).
    Ohne Korrektur wären alle PDs zu hoch. Die Rangfolge der Firmen (AUC) ändert sich dadurch nicht.
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    a = target_rate / train_rate
    b = (1 - target_rate) / (1 - train_rate)
    return p * a / (p * a + (1 - p) * b)
