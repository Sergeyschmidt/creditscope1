"""Holt ECHTE Jahresabschlussdaten über yfinance und baut den Trainingsdatensatz.

Aufruf (lokal, mit Internet):
    python src/data_fetch.py                     # Standard-Tickerliste
    python src/data_fetch.py AAPL SAP.DE VOW3.DE # eigene Ticker

Ergebnis: data/panel.csv (gleiches Schema wie der Demo-Datensatz).

Zielgröße (Proxy, bitte in der Arbeit methodisch begründen):
    distress_next_year = 1, wenn im FOLGEJAHR
        - die Zinsdeckung (EBIT / Zinsaufwand) < 1.0 ist  ODER
        - das Eigenkapital negativ ist.
Features stammen aus Jahr t, das Label aus Jahr t+1 (kein Look-Ahead).
Hinweis: yfinance liefert meist nur 4 Jahresabschlüsse pro Firma -> 3 Beobachtungen.
Je mehr Ticker, desto besser (Empfehlung: 300+ aus S&P 500 / STOXX 600 / Russell 2000).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

OUT = Path(__file__).resolve().parents[1] / "data" / "panel.csv"

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "F", "GM", "BA", "T", "VZ", "KO", "PEP", "XOM", "CVX", "WMT",
    "SAP.DE", "VOW3.DE", "BMW.DE", "SIE.DE", "BAS.DE", "LHA.DE", "TUI1.DE", "ZAL.DE",
    "CCL", "AAL", "UAL", "M", "KSS", "GPS", "RIG", "NCLH",
]

ROWS = {  # Zielspalte -> mögliche yfinance-Zeilennamen
    "revenue": ["Total Revenue"],
    "ebit": ["EBIT", "Operating Income"],
    "ebitda": ["EBITDA", "Normalized EBITDA"],
    "net_income": ["Net Income"],
    "interest_expense": ["Interest Expense", "Interest Expense Non Operating"],
    "total_assets": ["Total Assets"],
    "total_liabilities": ["Total Liabilities Net Minority Interest"],
    "total_debt": ["Total Debt"],
    "cash": ["Cash And Cash Equivalents"],
    "current_assets": ["Current Assets"],
    "current_liabilities": ["Current Liabilities"],
    "retained_earnings": ["Retained Earnings"],
    "operating_cashflow": ["Operating Cash Flow"],
}


def _get(df: pd.DataFrame, names):
    for n in names:
        if n in df.index:
            return df.loc[n]
    return pd.Series(dtype="float64")


def company_frame(ticker: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    stmts = pd.concat([t.financials, t.balance_sheet, t.cashflow])
    stmts = stmts[~stmts.index.duplicated(keep="first")]
    if stmts.empty:
        return pd.DataFrame()
    d = pd.DataFrame({col: _get(stmts, names) for col, names in ROWS.items()}) / 1e6
    d = d.sort_index()                           # älteste Periode zuerst
    d["interest_expense"] = d["interest_expense"].abs()
    mcap = (t.info or {}).get("marketCap")       # aktuell, nur Näherung für alle Jahre
    d["market_cap"] = (mcap / 1e6) if mcap else np.nan
    d.insert(0, "company", ticker)
    d.insert(1, "year", pd.to_datetime(d.index).year)

    # Label aus dem Folgejahr
    cover_next = (d["ebit"] / d["interest_expense"].replace(0, np.nan)).shift(-1)
    equity_next = (d["total_assets"] - d["total_liabilities"]).shift(-1)
    distress = ((cover_next < 1.0) | (equity_next < 0)).astype(float)
    distress[cover_next.isna() & equity_next.isna()] = np.nan
    d["distress_next_year"] = distress
    return d.dropna(subset=["distress_next_year", "total_assets", "ebit", "revenue"]).reset_index(drop=True)


def main(tickers):
    frames = []
    for tk in tickers:
        try:
            f = company_frame(tk)
            print(f"{tk:10s} {len(f)} Beobachtungen")
            frames.append(f)
        except Exception as e:  # einzelne Ticker dürfen fehlschlagen
            print(f"{tk:10s} FEHLER: {e}")
    panel = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(exist_ok=True)
    panel.to_csv(OUT, index=False)
    print(f"\n{len(panel)} Zeilen -> {OUT} | Notlage-Rate: {panel['distress_next_year'].mean():.1%}")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_TICKERS)
