"""Trainiert zwei Modelle auf data/panel.csv und speichert sie in models/.

1) Gradient Boosting (Hauptmodell der Streamlit-App, mit SHAP-Erklärung)
2) Logistische Regression (transparentes Vergleichsmodell, auch als JSON für die Web-Version)

Aufruf:  python src/train.py
Split: nach Firma (GroupShuffleSplit), damit dieselbe Firma nicht in Train UND Test steckt.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES, FEATURE_INFO, compute_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "panel.csv"
MODELS = ROOT / "models"


def main():
    MODELS.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    X = compute_features(df)
    y = df["distress_next_year"].astype(int)
    groups = df["company"]

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=7).split(X, y, groups))
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]

    gb = GradientBoostingClassifier(n_estimators=250, max_depth=3, learning_rate=0.05,
                                    subsample=0.8, random_state=7).fit(Xtr, ytr)
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(Xtr), ytr)

    p_gb = gb.predict_proba(Xte)[:, 1]
    p_lr = lr.predict_proba(sc.transform(Xte))[:, 1]
    metrics = {
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "base_rate": float(y.mean()),
        "gradient_boosting": {"auc": float(roc_auc_score(yte, p_gb)), "brier": float(brier_score_loss(yte, p_gb))},
        "logistic_regression": {"auc": float(roc_auc_score(yte, p_lr)), "brier": float(brier_score_loss(yte, p_lr))},
        "data_source": "synthetic" if str(df["company"].iloc[0]).startswith("FIRMA_") else "real",
    }
    print(json.dumps(metrics, indent=2))

    pd.DataFrame({"y": yte.values, "p_gb": p_gb, "p_lr": p_lr}).to_csv(MODELS / "test_preds.csv", index=False)
    joblib.dump({"scaler": sc, "lr": lr}, MODELS / "lr_model.joblib")
    joblib.dump({"model": gb, "features": FEATURES}, MODELS / "gb_model.joblib")
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    # Hintergrunddaten für SHAP und Perzentil-Vergleiche in der App
    Xtr.sample(min(100, len(Xtr)), random_state=1).to_csv(MODELS / "background.csv", index=False)
    X.describe(percentiles=[.05, .25, .5, .75, .95]).T.to_csv(MODELS / "feature_stats.csv")

    # Export der logistischen Regression für die Browser-Version
    export = {
        "features": FEATURES,
        "labels": {k: v[0] for k, v in FEATURE_INFO.items()},
        "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
        "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
        "metrics": metrics["logistic_regression"], "base_rate": metrics["base_rate"],
        "data_source": metrics["data_source"],
    }
    (MODELS / "logreg_export.json").write_text(json.dumps(export, indent=2))
    print("Gespeichert in", MODELS)


if __name__ == "__main__":
    main()
