"""Kennzahlen (Features) für CreditScope.

Alle Features werden aus Rohdaten der Jahresabschlüsse berechnet. Die Modelle
und die App nutzen ausschließlich FEATURES (Reihenfolge ist fix).
"""
import numpy as np
import pandas as pd

# Rohdaten-Spalten (in Mio. der Berichtswährung), die jede Datenquelle liefern muss
RAW_COLUMNS = [
    "revenue", "ebit", "ebitda", "net_income", "interest_expense",
    "total_assets", "total_liabilities", "total_debt", "cash",
    "current_assets", "current_liabilities", "retained_earnings",
    "market_cap", "operating_cashflow",
]

# Features mit Anzeigenamen und Kurzerklärung (wird in der App angezeigt)
FEATURE_INFO = {
    "debt_to_equity": ("Verschuldungsgrad (Debt/Equity)", "Fremdkapital im Verhältnis zum Eigenkapital"),
    "net_debt_to_ebitda": ("Net Debt / EBITDA", "Wie viele Jahre EBITDA nötig wären, um Nettoschulden zu tilgen"),
    "interest_coverage": ("Zinsdeckung (EBIT/Zinsen)", "Wie oft der Gewinn die Zinsen deckt"),
    "ebit_margin": ("EBIT-Marge", "Operativer Gewinn pro Umsatz"),
    "roa": ("Return on Assets", "Jahresüberschuss im Verhältnis zur Bilanzsumme"),
    "current_ratio": ("Current Ratio", "Kurzfristige Liquidität (Umlaufvermögen / kurzfr. Verbindlichkeiten)"),
    "cfo_to_debt": ("Operativer CF / Schulden", "Cashflow-Deckung der Finanzschulden"),
    "retained_earnings_to_assets": ("Gewinnrücklagen / Bilanzsumme", "Anteil kumulierter Gewinne, Indikator für Reife"),
    "log_assets": ("Größe (log Bilanzsumme)", "Größere Firmen fallen seltener aus"),
}
FEATURES = list(FEATURE_INFO)


def _safe_div(a, b):
    a = pd.Series(a, dtype="float64")
    b = pd.Series(b, dtype="float64").replace(0, np.nan)
    return a / b


def compute_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Berechnet FEATURES aus RAW_COLUMNS. Extreme Werte werden gekappt."""
    f = pd.DataFrame(index=raw.index)
    equity = raw["total_assets"] - raw["total_liabilities"]
    net_debt = raw["total_debt"] - raw["cash"]

    # Negatives Eigenkapital -> Verschuldungsgrad sehr hoch (gekappt)
    f["debt_to_equity"] = np.where(equity > 0, _safe_div(raw["total_debt"], equity), 10.0)
    f["net_debt_to_ebitda"] = np.where(raw["ebitda"] > 0, _safe_div(net_debt, raw["ebitda"]), 15.0)
    f["interest_coverage"] = np.where(raw["interest_expense"] > 0,
                                      _safe_div(raw["ebit"], raw["interest_expense"]), 50.0)
    f["ebit_margin"] = _safe_div(raw["ebit"], raw["revenue"])
    f["roa"] = _safe_div(raw["net_income"], raw["total_assets"])
    f["current_ratio"] = _safe_div(raw["current_assets"], raw["current_liabilities"])
    f["cfo_to_debt"] = np.where(raw["total_debt"] > 0,
                                _safe_div(raw["operating_cashflow"], raw["total_debt"]), 5.0)
    f["retained_earnings_to_assets"] = _safe_div(raw["retained_earnings"], raw["total_assets"])
    f["log_assets"] = np.log(raw["total_assets"].clip(lower=1))

    bounds = {
        "debt_to_equity": (0, 10), "net_debt_to_ebitda": (-5, 15),
        "interest_coverage": (-10, 50), "ebit_margin": (-1, 1), "roa": (-0.5, 0.5),
        "current_ratio": (0, 6), "cfo_to_debt": (-1, 5),
        "retained_earnings_to_assets": (-2, 1), "log_assets": (0, 20),
    }
    for col, (lo, hi) in bounds.items():
        f[col] = f[col].clip(lo, hi)
    return f[FEATURES].fillna(f[FEATURES].median())


def altman_z(raw: pd.DataFrame) -> pd.Series:
    """Altman Z-Score (Originalversion für börsennotierte Industrieunternehmen).

    Wird NUR als Vergleichswert in der App angezeigt, nicht als Zielgröße.
    Zonen: > 2.99 sicher, 1.81-2.99 Grauzone, < 1.81 Gefahrenzone.
    """
    ta = raw["total_assets"].replace(0, np.nan)
    tl = raw["total_liabilities"].replace(0, np.nan)
    wc = raw["current_assets"] - raw["current_liabilities"]
    return (1.2 * wc / ta + 1.4 * raw["retained_earnings"] / ta + 3.3 * raw["ebit"] / ta
            + 0.6 * raw["market_cap"] / tl + 1.0 * raw["revenue"] / ta)


def pd_to_rating(pd_value: float) -> str:
    """Einfache Ratingskala aus der Ausfallwahrscheinlichkeit (1 Jahr)."""
    bounds = [(0.005, "A"), (0.015, "BBB"), (0.04, "BB"), (0.10, "B"), (0.25, "CCC")]
    for limit, label in bounds:
        if pd_value < limit:
            return label
    return "D-Gefahr"


# Formeln und Richtung (niedrig/hoch ist besser) für die Detailansicht in der App
FEATURE_FORMULA = {
    "debt_to_equity": "Finanzschulden / (Bilanzsumme − Verbindlichkeiten gesamt)",
    "net_debt_to_ebitda": "(Finanzschulden − Zahlungsmittel) / EBITDA",
    "interest_coverage": "EBIT / Zinsaufwand",
    "ebit_margin": "EBIT / Umsatz",
    "roa": "Jahresüberschuss / Bilanzsumme",
    "current_ratio": "Umlaufvermögen / kurzfristige Verbindlichkeiten",
    "cfo_to_debt": "Operativer Cashflow / Finanzschulden",
    "retained_earnings_to_assets": "Gewinnrücklagen / Bilanzsumme",
    "log_assets": "ln(Bilanzsumme)",
}
FEATURE_BETTER = {  # "low" = kleinerer Wert ist besser, "high" = größerer Wert ist besser
    "debt_to_equity": "low", "net_debt_to_ebitda": "low", "interest_coverage": "high",
    "ebit_margin": "high", "roa": "high", "current_ratio": "high", "cfo_to_debt": "high",
    "retained_earnings_to_assets": "high", "log_assets": "high",
}
