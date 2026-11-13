# CreditScope – Kreditrisiko-Scoring mit erklärbarem ML

Schätzt die Wahrscheinlichkeit, dass ein Unternehmen im Folgejahr in Zahlungsnot gerät
(Zinsdeckung < 1 oder negatives Eigenkapital), und erklärt jedes Ergebnis mit SHAP.

## Schnellstart
```bash
pip install -r requirements.txt
python src/make_demo_data.py   # synthetische Demo-Daten (nur zum Testen!)
python src/train.py            # trainiert und speichert Modelle in models/
streamlit run app.py
```

## Mit echten Daten (für die Abgabe)
```bash
python src/data_fetch.py                  # oder: python src/data_fetch.py AAPL SAP.DE ...
python src/train.py
```
Tickerliste in `src/data_fetch.py` erweitern (Ziel: 300+ Firmen). Ergebnis nach dem Training prüfen
(AUC, Notlagen-Rate), bevor ihr die Zahlen in die Arbeit übernehmt.

## Hosting (Streamlit Community Cloud, kostenlos)
1. Ordner als GitHub-Repo hochladen (inkl. `models/` und `data/`, damit die App sofort startet).
2. share.streamlit.io -> "New app" -> Repo, Branch, Datei `app.py`.
3. Vor dem Termin einmal aufrufen (App schläft nach Inaktivität).

## Funktionen der App
| Tab | Inhalt |
|---|---|
| Analyse | PD, Rating, Treiber-Grafik (anklickbar), Kennzahl-Kacheln: Klick zeigt Formel, Verteilung der Vergleichsfirmen, PD-Kurve und Median-Szenario |
| Szenarien & Stress | Umsatz- und Zinsschock per Regler, Kippunkte des Ratings, PD-Landkarte, vordefinierte Szenarien, Fremdkapitalkosten (Anschluss an WACC) |
| Portfolio | Demo-Portfolio oder CSV: Expected Loss (PD × LGD × Exposure), Rating-Verteilung, Konzentration, Top-Risiken |
| Modellqualität | ROC, Kalibrierung, Wichtigkeit, Schwellenwert-Simulation (Trefferquote vs. Fehlalarme) |
| KI-Assistent | Claude liest Zahlen aus Berichten, schreibt Kreditmemo |
| Methodik | Zielgröße, Daten, Annahmen, Grenzen |

Scoring-Modell in der Seitenleiste wählbar: Ensemble (Standard), Gradient Boosting oder Logistische Regression.

## Claude API (KI-Assistent)
Tab "KI-Assistent": liest Zahlen aus Geschäftsbericht-Text/PDF und schreibt ein Kreditmemo.
1. API-Schlüssel auf console.anthropic.com erzeugen (eigenes Guthaben nötig, getrennt vom Claude-Abo).
2. Lokal: `.streamlit/secrets.toml` nach Vorlage `secrets.toml.example` anlegen.
3. Streamlit Cloud: App -> Settings -> Secrets -> `ANTHROPIC_API_KEY = "..."`.
Der Schlüssel darf nie im Repository landen (`.gitignore` schützt die lokale Datei).
Hinweis: Die statische HTML-Version in `web/` nutzt die API nicht, weil ein Schlüssel im Browser öffentlich wäre.

## Struktur
| Datei | Zweck |
|---|---|
| `src/analysis.py` | Stresstest, Szenarien, Portfolio, Ensemble |
| `src/viz.py` | Plotly-Grafiken |
| `src/features.py` | Kennzahlen, Altman Z, Rating-Skala |
| `src/data_fetch.py` | echte Daten über yfinance, Label aus Jahr t+1 |
| `src/make_demo_data.py` | synthetischer Testdatensatz |
| `src/train.py` | Gradient Boosting + Logistische Regression, Export |
| `src/llm.py` | Claude-API: Auslesen und Kreditmemo |
| `app.py` | Streamlit-Dashboard |

## Projektplan und Aufgaben
| Phase | Inhalt | Wer |
|---|---|---|
| 1 Daten | Tickerliste erweitern, `data_fetch.py` laufen lassen, Daten prüfen | Person A |
| 2 Modell | Features/Hyperparameter testen, Kalibrierung, Vergleich mit Altman Z | Person B |
| 3 App | Dashboard verbessern (Branchenvergleich, Zeitreihe), Deployment | Person C |
| 4 Doku | Methodik, Proxy-Begründung, Grenzen, Präsentation | Person D |

## Ausbaustufen nach der Abgabe
LLM-generiertes Kreditmemo aus Kennzahlen und Geschäftsbericht, Branchen-/Makrovariablen,
Zeitreihe der PD je Firma, Covenant-Monitoring, Verknüpfung mit WACC (PD -> Fremdkapitalkosten).
