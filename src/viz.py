"""Plotly-Grafiken für CreditScope."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score, roc_curve

GOOD, BAD, ACCENT, NEUTRAL = "#2e7d4f", "#b3372c", "#1f6fb2", "#8a94a3"
RATING_COLORS = {"A": "#2e7d4f", "BBB": "#6aa84f", "BB": "#d4a72c", "B": "#e07b39", "CCC": "#c4472f", "D-Gefahr": "#8e1b1b"}
RATING_ORDER = ["A", "BBB", "BB", "B", "CCC", "D-Gefahr"]
THRESHOLDS = [(0.005, "A/BBB"), (0.015, "BBB/BB"), (0.04, "BB/B"), (0.10, "B/CCC"), (0.25, "CCC/D")]


def _fig(h=320, title=None):
    f = go.Figure()
    f.update_layout(height=h, margin=dict(l=10, r=10, t=40 if title else 15, b=10), title=title,
                    legend=dict(orientation="h", y=-0.2), hoverlabel=dict(font_size=13))
    return f


def drivers_bar(labels, values, title="Beitrag zum Risiko (Log-Odds), anklicken für Details"):
    order = np.argsort(np.abs(values))
    lab = [labels[i] for i in order]
    val = [float(values[i]) for i in order]
    f = _fig(340, title)
    f.add_bar(x=val, y=lab, orientation="h", marker_color=[BAD if v > 0 else GOOD for v in val],
              hovertemplate="%{y}: %{x:+.2f}<extra></extra>")
    f.update_xaxes(title="← senkt Risiko | erhöht Risiko →", zeroline=True)
    f.update_layout(clickmode="event+select")
    return f


def hist_marker(pop, value, title, name="Unternehmen"):
    f = _fig(300, title)
    f.add_histogram(x=pop, nbinsx=40, marker_color=NEUTRAL, opacity=0.8, name="Stichprobe",
                    hovertemplate="%{x}: %{y} Firmen<extra></extra>")
    f.add_vline(x=value, line_dash="dash", line_color=ACCENT, line_width=3,
                annotation_text=name, annotation_position="top")
    f.update_layout(showlegend=False, yaxis_title="Anzahl", bargap=0.03)
    return f


def line_marker(x, y_pct, x0, y0_pct, title, xlabel):
    f = _fig(300, title)
    f.add_scatter(x=x, y=y_pct, mode="lines", line=dict(color=ACCENT, width=3),
                  hovertemplate=xlabel + ": %{x:.2f}<br>PD: %{y:.1f} %<extra></extra>")
    f.add_scatter(x=[x0], y=[y0_pct], mode="markers", marker=dict(size=12, color=BAD, line=dict(width=2, color="white")),
                  hovertemplate="aktuell: %{y:.1f} %<extra></extra>")
    f.update_layout(showlegend=False, xaxis_title=xlabel, yaxis_title="PD (%)")
    return f


def revenue_curve_fig(curve, current_shock, current_pd):
    f = _fig(340, "Ausfallwahrscheinlichkeit bei Umsatzänderung")
    f.add_scatter(x=curve["shock"] * 100, y=curve["pd"] * 100, mode="lines", line=dict(color=ACCENT, width=3),
                  hovertemplate="Umsatz %{x:+.0f} %<br>PD %{y:.1f} %<extra></extra>")
    for thr, label in THRESHOLDS:
        if thr * 100 <= max(curve["pd"].max() * 100 * 1.1, 5):
            f.add_hline(y=thr * 100, line_dash="dot", line_color=NEUTRAL, line_width=1,
                        annotation_text=label, annotation_position="right")
    f.add_scatter(x=[current_shock * 100], y=[current_pd * 100], mode="markers",
                  marker=dict(size=13, color=BAD, line=dict(width=2, color="white")), name="Gewähltes Szenario")
    f.update_layout(showlegend=False, xaxis_title="Umsatzänderung (%)", yaxis_title="PD (%)")
    return f


def heatmap_fig(rev, rate, p):
    f = _fig(340, "PD-Landkarte: Umsatz × Zinsanstieg")
    f.add_heatmap(x=[f"{r:+.0%}" for r in rev], y=[f"+{k:.0f} pp" for k in rate], z=p * 100,
                  colorscale=[[0, "#2e7d4f"], [0.15, "#d4a72c"], [0.5, "#e07b39"], [1, "#8e1b1b"]],
                  zmin=0, zmax=max(40, float(p.max() * 100)),
                  hovertemplate="Umsatz %{x}, Zins %{y}<br>PD %{z:.1f} %<extra></extra>",
                  colorbar=dict(title="PD %"))
    f.update_layout(xaxis_title="Umsatzänderung", yaxis_title="Zinsanstieg")
    return f


def scenario_bar(df):
    f = _fig(300, "PD je Szenario")
    f.add_bar(x=df["Szenario"], y=df["PD"] * 100, marker_color=[RATING_COLORS[r] for r in df["Rating"]],
              text=[f"{r}" for r in df["Rating"]], textposition="outside",
              hovertemplate="%{x}: %{y:.1f} %<extra></extra>")
    f.update_layout(yaxis_title="PD (%)", showlegend=False)
    return f


def roc_fig(y, p_gb, p_lr, p_ens=None):
    f = _fig(340, "ROC-Kurve (Testdaten)")
    for name, p, col in [("Gradient Boosting", p_gb, ACCENT), ("Logistische Regression", p_lr, "#d4a72c")] + ([("Ensemble", p_ens, "#2e7d4f")] if p_ens is not None else []):
        fpr, tpr, _ = roc_curve(y, p)
        f.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC {roc_auc_score(y, p):.3f})", line=dict(color=col, width=3))
    f.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=NEUTRAL, dash="dash"), name="Zufall")
    f.update_layout(xaxis_title="Falsch-Positiv-Rate", yaxis_title="Trefferquote", legend=dict(y=-0.25))
    return f


def calib_fig(y, p, bins=8):
    d = pd.DataFrame({"y": y, "p": p})
    d["bin"] = pd.qcut(d["p"], bins, duplicates="drop")
    g = d.groupby("bin", observed=True).agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
    m = max(g["pred"].max(), g["obs"].max()) * 100 * 1.1
    f = _fig(340, "Kalibrierung: vorhergesagt vs. tatsächlich")
    f.add_scatter(x=[0, m], y=[0, m], mode="lines", line=dict(color=NEUTRAL, dash="dash"), name="perfekt")
    f.add_scatter(x=g["pred"] * 100, y=g["obs"] * 100, mode="lines+markers", name="Modell",
                  marker=dict(size=9, color=ACCENT), line=dict(color=ACCENT, width=3),
                  customdata=g["n"], hovertemplate="Vorhersage %{x:.1f} %<br>Tatsächlich %{y:.1f} %<br>n=%{customdata}<extra></extra>")
    f.update_layout(xaxis_title="Ø vorhergesagte PD (%)", yaxis_title="Tatsächliche Notlagen-Rate (%)")
    return f


def importance_fig(names, values):
    order = np.argsort(values)
    f = _fig(320, "Globale Wichtigkeit der Kennzahlen")
    f.add_bar(x=[float(values[i]) for i in order], y=[names[i] for i in order], orientation="h", marker_color=ACCENT)
    f.update_layout(xaxis_title="Anteil am Modell")
    return f


def class_hist(y, p):
    f = _fig(320, "Verteilung der PD: gesunde vs. notleidende Firmen")
    f.add_histogram(x=p[y == 0] * 100, name="keine Notlage", marker_color=GOOD, opacity=0.7, nbinsx=40)
    f.add_histogram(x=p[y == 1] * 100, name="Notlage", marker_color=BAD, opacity=0.7, nbinsx=40)
    f.update_layout(barmode="overlay", xaxis_title="Vorhergesagte PD (%)", yaxis_title="Anzahl")
    return f


def rating_dist(df):
    g = df.groupby("Rating")["exposure"].sum().reindex(RATING_ORDER).fillna(0)
    f = _fig(300, "Exposure nach Rating")
    f.add_bar(x=g.index, y=g.values, marker_color=[RATING_COLORS[r] for r in g.index],
              hovertemplate="%{x}: %{y:.0f} Mio.<extra></extra>")
    f.update_layout(yaxis_title="Exposure (Mio.)", showlegend=False)
    return f


def el_bar(df, n=10):
    t = df.sort_values("EL", ascending=False).head(n).iloc[::-1]
    f = _fig(320, f"Größte Expected-Loss-Treiber (Top {n})")
    f.add_bar(x=t["EL"], y=t["company"], orientation="h", marker_color=[RATING_COLORS[r] for r in t["Rating"]],
              hovertemplate="%{y}: %{x:.2f} Mio.<extra></extra>")
    f.update_layout(xaxis_title="Expected Loss (Mio.)")
    return f


def pd_hist(p):
    f = _fig(300, "Verteilung der PD im Portfolio")
    f.add_histogram(x=p * 100, nbinsx=30, marker_color=ACCENT)
    f.update_layout(xaxis_title="PD (%)", yaxis_title="Anzahl", showlegend=False)
    return f
