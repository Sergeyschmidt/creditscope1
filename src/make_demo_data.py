"""Erzeugt einen SYNTHETISCHEN Demo-Datensatz (damit die App ohne API-Zugang läuft).

ACHTUNG: Diese Daten sind künstlich. Für die Abgabe sollte data_fetch.py mit echten
Unternehmensdaten laufen (siehe README). Das Schema ist identisch, die App und das
Training funktionieren mit beiden Datensätzen gleich.

Aufruf:  python src/make_demo_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "panel.csv"


def make(n=4000, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    assets = np.exp(rng.normal(8.0, 1.4, n))                 # Mio.
    revenue = assets * rng.uniform(0.3, 1.6, n)
    ebit_margin = rng.normal(0.07, 0.09, n)
    ebit = revenue * ebit_margin
    ebitda = ebit + revenue * rng.uniform(0.02, 0.07, n)
    debt = assets * rng.beta(2, 4, n) * 1.1
    cash = assets * rng.uniform(0.01, 0.15, n)
    other_liab = assets * rng.uniform(0.05, 0.35, n)
    liabilities = debt + other_liab
    interest = debt * rng.uniform(0.02, 0.08, n)
    net_income = (ebit - interest) * 0.75
    current_assets = assets * rng.uniform(0.15, 0.55, n)
    current_liab = (other_liab * rng.uniform(0.5, 1.0, n)) + debt * rng.uniform(0.05, 0.3, n)
    retained = assets * rng.normal(0.12, 0.2, n)
    equity = assets - liabilities
    market_cap = np.maximum(equity, assets * 0.05) * np.exp(rng.normal(0.3, 0.5, n))
    cfo = ebitda * rng.uniform(0.5, 1.0, n) - interest

    df = pd.DataFrame(dict(
        revenue=revenue, ebit=ebit, ebitda=ebitda, net_income=net_income,
        interest_expense=interest, total_assets=assets, total_liabilities=liabilities,
        total_debt=debt, cash=cash, current_assets=current_assets,
        current_liabilities=current_liab, retained_earnings=retained,
        market_cap=market_cap, operating_cashflow=cfo))

    # Label: Ausfall/Notlage im Folgejahr. Latente Risiko-Funktion + Zufall.
    from features import compute_features
    f = compute_features(df)
    z = (-4.2
         + 0.35 * f["debt_to_equity"].clip(0, 6)
         + 0.18 * f["net_debt_to_ebitda"].clip(-2, 12)
         - 0.22 * np.log1p(f["interest_coverage"].clip(-0.9, 40))
         - 3.0 * f["ebit_margin"]
         - 4.0 * f["roa"]
         - 0.25 * f["current_ratio"].clip(0, 4)
         - 1.2 * f["cfo_to_debt"].clip(-0.5, 2)
         - 0.8 * f["retained_earnings_to_assets"]
         - 0.08 * (f["log_assets"] - 8)
         + rng.normal(0, 0.6, n))
    p = 1 / (1 + np.exp(-z))
    df["distress_next_year"] = (rng.uniform(size=n) < p).astype(int)
    df.insert(0, "company", [f"FIRMA_{i:04d}" for i in range(n)])
    df.insert(1, "year", rng.integers(2019, 2025, n))
    return df.round(4)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    OUT.parent.mkdir(exist_ok=True)
    d = make()
    d.to_csv(OUT, index=False)
    print(f"{len(d)} Zeilen -> {OUT} | Ausfallrate: {d['distress_next_year'].mean():.1%}")
