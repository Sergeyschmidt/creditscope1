"""Holt echte Jahresabschlüsse für die Referenzfirmen und schreibt data/reference_firms.csv.

Aufruf (lokal, braucht Internet):  python src/fetch_reference.py
Für Pleitefirmen liefert Yahoo oft nichts mehr: dort Werte aus dem Geschäftsbericht von Hand ergänzen.
"""
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_live import fetch_latest  # noqa: E402

FIRMS = [
    ("MSFT", "A (sehr solide)", "keine Notlage"), ("KO", "A (sehr solide)", "keine Notlage"),
    ("JNJ", "A (sehr solide)", "keine Notlage"), ("T", "BBB (mittel)", "keine Notlage"),
    ("F", "BB (mittel)", "keine Notlage"), ("BA", "B (angeschlagen)", "Verlust, kein Ausfall"),
    ("UAL", "B (angeschlagen)", "keine Notlage"), ("AMC", "CCC (kritisch)", "Notlage"),
    ("CVNA", "B (angeschlagen)", "keine Notlage"),
]


def main():
    rows = []
    for tk, cls, outcome in FIRMS:
        try:
            r = fetch_latest(tk)
            rows.append({"name": r["name"], "ticker": tk, "year": r["year"], "risk_class": cls, **r["raw"], "outcome": outcome})
            print("ok", tk)
        except Exception as e:  # noqa: BLE001
            print("Fehler", tk, e)
    out = Path(__file__).resolve().parents[1] / "data" / "reference_firms.csv"
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main()
