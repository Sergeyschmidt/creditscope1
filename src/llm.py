"""Claude-API-Anbindung für CreditScope.

Zwei Funktionen:
  extract_financials(text)  -> liest Kennzahlen aus Geschäftsbericht-Text (strukturierte Ausgabe)
  write_memo(...)           -> schreibt ein Kreditmemo aus den Modellergebnissen

Der API-Schlüssel kommt aus st.secrets["ANTHROPIC_API_KEY"] oder der Umgebungsvariable
ANTHROPIC_API_KEY. Niemals im Code oder im Repository speichern.
"""
import os

DEFAULT_MODEL = "claude-sonnet-5-5"

RAW_FIELDS = {
    "revenue": "Umsatz", "ebit": "EBIT (operatives Ergebnis)", "ebitda": "EBITDA",
    "net_income": "Jahresüberschuss", "interest_expense": "Zinsaufwand (positiv)",
    "total_assets": "Bilanzsumme", "total_liabilities": "Verbindlichkeiten gesamt (ohne Eigenkapital)",
    "total_debt": "Finanzschulden (verzinsliche Verbindlichkeiten)", "cash": "Zahlungsmittel",
    "current_assets": "Umlaufvermögen", "current_liabilities": "Kurzfristige Verbindlichkeiten",
    "retained_earnings": "Gewinnrücklagen", "market_cap": "Marktkapitalisierung (falls genannt, sonst null)",
    "operating_cashflow": "Operativer Cashflow",
}

EXTRACT_TOOL = {
    "name": "record_financials",
    "description": "Speichert die aus dem Text gelesenen Finanzkennzahlen in Millionen der Berichtswährung.",
    "input_schema": {
        "type": "object",
        "properties": {
            **{k: {"type": ["number", "null"], "description": v} for k, v in RAW_FIELDS.items()},
            "currency": {"type": ["string", "null"], "description": "Währung, z. B. EUR"},
            "fiscal_year": {"type": ["integer", "null"]},
            "evidence": {"type": "string",
                         "description": "Kurz: aus welchen Textstellen/Tabellen die Werte stammen und was geschätzt oder abgeleitet wurde"},
        },
        "required": list(RAW_FIELDS) + ["evidence"],
    },
}


def get_api_key():
    try:
        import streamlit as st
        if "ANTHROPIC_API_KEY" in st.secrets:
            return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass
    return os.environ.get("ANTHROPIC_API_KEY")


def get_client():
    key = get_api_key()
    if not key:
        return None
    import anthropic
    return anthropic.Anthropic(api_key=key)


def get_model():
    try:
        import streamlit as st
        if "CLAUDE_MODEL" in st.secrets:
            return st.secrets["CLAUDE_MODEL"]
    except Exception:
        pass
    return os.environ.get("CLAUDE_MODEL", DEFAULT_MODEL)


def extract_financials(text: str, client, model: str = DEFAULT_MODEL) -> dict:
    """Liest Kennzahlen aus Berichtstext. Fehlende Werte kommen als None zurück."""
    msg = client.messages.create(
        model=model, max_tokens=1500,
        tools=[EXTRACT_TOOL], tool_choice={"type": "tool", "name": "record_financials"},
        messages=[{"role": "user", "content":
                   "Lies die folgenden Auszüge aus einem Geschäftsbericht und trage die Kennzahlen des "
                   "letzten Geschäftsjahres in Millionen der Berichtswährung ein. Rechne Tausend/Milliarden um. "
                   "Erfinde nichts: Was nicht im Text steht oder sich nicht eindeutig ableiten lässt, bleibt null. "
                   "Nenne in 'evidence', was abgeleitet wurde.\n\n<bericht>\n" + text[:60000] + "\n</bericht>"}],
    )
    for block in msg.content:
        if getattr(block, "type", None) == "tool_use":
            return dict(block.input)
    raise ValueError("Die KI hat keine strukturierten Werte geliefert.")


def write_memo(ctx: dict, client, model: str = DEFAULT_MODEL) -> str:
    """Schreibt ein kurzes Kreditmemo. ctx enthält nur Zahlen aus dem Modell, nichts Erfundenes."""
    lines = "\n".join(f"- {k}: {v}" for k, v in ctx["features"].items())
    drivers = "\n".join(f"- {k}: {v:+.2f}" for k, v in ctx["drivers"])
    prompt = f"""Du bist Kreditanalyst. Schreibe auf Deutsch ein kurzes Kreditmemo (max. 250 Wörter) zu einem Unternehmen.
Nutze AUSSCHLIESSLICH die folgenden Zahlen. Erfinde keine Branche, Namen, Ereignisse oder weitere Zahlen.

Modellergebnis (Notlage im Folgejahr): {ctx['pd']:.1%}, internes Rating: {ctx['rating']}
Altman Z-Score (Vergleich): {ctx['z']:.2f} ({ctx['zone']})

Kennzahlen:
{lines}

Beiträge zum Risiko (Log-Odds, positiv = erhöht das Risiko):
{drivers}

Gliederung mit kurzen Überschriften: 1. Einschätzung, 2. Wichtigste Treiber, 3. Auffälligkeiten und Risiken, 4. Empfehlung für die weitere Prüfung.
Weise am Ende in einem Satz auf die Grenzen des Modells hin: {ctx['limits']}"""
    msg = client.messages.create(model=model, max_tokens=900,
                                 messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
