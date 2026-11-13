"""Live-Daten: holt den letzten Jahresabschluss eines börsennotierten Unternehmens über yfinance.

parse_statements() arbeitet nur auf den Tabellen und ist ohne Internet testbar.
Werte werden in Millionen der Berichtswährung zurückgegeben.
"""
import numpy as np
import pandas as pd

ROWS = {  # Zielspalte -> mögliche yfinance-Zeilennamen
    "revenue": ["Total Revenue"],
    "ebit": ["EBIT", "Operating Income"],
    "ebitda": ["EBITDA", "Normalized EBITDA"],
    "net_income": ["Net Income", "Net Income Common Stockholders"],
    "interest_expense": ["Interest Expense", "Interest Expense Non Operating"],
    "total_assets": ["Total Assets"],
    "total_liabilities": ["Total Liabilities Net Minority Interest"],
    "total_debt": ["Total Debt"],
    "cash": ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"],
    "current_assets": ["Current Assets"],
    "current_liabilities": ["Current Liabilities"],
    "retained_earnings": ["Retained Earnings"],
    "operating_cashflow": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
}
REQUIRED = ["revenue", "ebit", "total_assets", "total_liabilities"]


def _first(series: pd.Series, names):
    for n in names:
        if n in series.index and pd.notna(series[n]):
            return float(series[n])
    return np.nan


def parse_statements(fin: pd.DataFrame, bs: pd.DataFrame, cf: pd.DataFrame):
    """Nimmt das jüngste Geschäftsjahr, in dem die Pflichtfelder vorhanden sind."""
    frames = [d for d in (fin, bs, cf) if d is not None and not d.empty]
    if not frames:
        raise ValueError("Keine Jahresabschlussdaten gefunden. Ticker prüfen (z. B. SAP.DE, AAPL).")
    stm = pd.concat(frames)
    stm = stm[~stm.index.duplicated(keep="first")]
    for col in sorted(stm.columns, reverse=True):
        s = stm[col]
        vals = {k: _first(s, names) for k, names in ROWS.items()}
        if all(pd.notna(vals[k]) for k in REQUIRED):
            if pd.isna(vals["ebitda"]):  # Näherung: EBIT + Abschreibungen
                vals["ebitda"] = vals["ebit"] + _first(s, ["Reconciled Depreciation"]) if pd.notna(_first(s, ["Reconciled Depreciation"])) else np.nan
            if pd.notna(vals["interest_expense"]):
                vals["interest_expense"] = abs(vals["interest_expense"])
            raw = {k: (None if pd.isna(v) else round(v / 1e6, 2)) for k, v in vals.items()}
            return raw, pd.Timestamp(col).year
    raise ValueError("Pflichtangaben (Umsatz, EBIT, Bilanzsumme, Verbindlichkeiten) fehlen in den Daten.")


def fetch_latest(ticker: str) -> dict:
    import yfinance as yf

    t = yf.Ticker(ticker)
    raw, year = parse_statements(t.financials, t.balance_sheet, t.cashflow)
    try:
        info = t.info or {}
    except Exception:  # noqa: BLE001
        info = {}
    mcap = info.get("marketCap")
    raw["market_cap"] = round(mcap / 1e6, 2) if mcap else None
    return {
        "raw": raw, "year": year, "ticker": ticker,
        "name": info.get("shortName") or ticker,
        "currency": info.get("financialCurrency") or info.get("currency") or "?",
        "missing": [k for k, v in raw.items() if v is None],
    }
