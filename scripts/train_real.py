#!/usr/bin/env python3
"""
Entrena el predictor de resultados (H/D/A) con partidos REALES de LaLiga y
reporta métricas HONESTAS contra los dos baselines que importan:

  1. Mayoría ("siempre gana el local") — el suelo.
  2. Cuotas de cierre de los bookies (Pinnacle/B365, margen quitado) — el techo
     práctico: el mercado agrega información que un modelo de stats públicas
     no tiene (alineaciones, lesiones, dinero informado).

Predecir fútbol es difícil: el estado del arte público con features de
rendimiento ronda 50-55% de accuracy y el mercado ~55%. Si el modelo no llega
al mercado, se publica tal cual — el valor está en CUÁNTO te acercas con
información pública.

Split temporal estricto: train ≤ 2022-23 · val 2023-24 · test 2024-25/26.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, log_loss

from fetch_real_data import FEATURE_COLS

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR, MODELS_DIR = ROOT / "data", ROOT / "models"

LABELS = {"H": 0, "D": 1, "A": 2}
VAL_SEASONS = {"2324"}
TEST_SEASONS = {"2425", "2526"}


def main():
    df = pd.read_parquet(DATA_DIR / "features.parquet")
    df["y"] = df["result"].map(LABELS)

    test_mask = df["season"].isin(TEST_SEASONS)
    val_mask = df["season"].isin(VAL_SEASONS)
    train_mask = ~(test_mask | val_mask)

    Xtr, ytr = df[train_mask][FEATURE_COLS], df[train_mask]["y"]
    Xva, yva = df[val_mask][FEATURE_COLS], df[val_mask]["y"]
    Xte, yte = df[test_mask][FEATURE_COLS], df[test_mask]["y"]
    print(f"train={len(Xtr)} val={len(Xva)} test={len(Xte)}")

    # ── Modelos ──────────────────────────────────────────────────────────────
    lgbm = lgb.LGBMClassifier(
        objective="multiclass", num_class=3, n_estimators=600, learning_rate=0.03,
        num_leaves=31, min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=42, verbosity=-1,
    )
    lgbm.fit(Xtr, ytr, eval_set=[(Xva, yva)],
             callbacks=[lgb.early_stopping(50, verbose=False)])

    xgbc = xgb.XGBClassifier(
        objective="multi:softprob", num_class=3, n_estimators=600, learning_rate=0.03,
        max_depth=5, min_child_weight=8, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=42, early_stopping_rounds=50, eval_metric="mlogloss",
    )
    xgbc.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

    lgbm_proba, xgb_proba = lgbm.predict_proba(Xte), xgbc.predict_proba(Xte)
    proba = (lgbm_proba + xgb_proba) / 2
    pred = proba.argmax(axis=1)

    acc = accuracy_score(yte, pred)
    lgbm_acc = accuracy_score(yte, lgbm_proba.argmax(axis=1))
    xgb_acc = accuracy_score(yte, xgb_proba.argmax(axis=1))
    f1m = f1_score(yte, pred, average="macro")
    ll = log_loss(yte, proba, labels=[0, 1, 2])

    # ── Baselines ────────────────────────────────────────────────────────────
    # 1) mayoría: siempre local
    base_majority = (yte == 0).mean()
    # 2) mercado (prob. implícitas sin margen)
    mkt = df[test_mask][["mkt_ph", "mkt_pd", "mkt_pa"]].to_numpy(dtype=float)
    has_mkt = ~np.isnan(mkt).any(axis=1)
    mkt_acc = accuracy_score(yte[has_mkt], mkt[has_mkt].argmax(axis=1))
    mkt_ll = log_loss(yte[has_mkt], mkt[has_mkt], labels=[0, 1, 2])
    # modelo sobre el MISMO subconjunto con cuotas (comparación justa)
    model_acc_mkt = accuracy_score(yte[has_mkt], pred[has_mkt])
    model_ll_mkt = log_loss(yte[has_mkt], proba[has_mkt], labels=[0, 1, 2])

    print(f"\n{'':22s}{'accuracy':>10s}{'log-loss':>10s}")
    print(f"{'Mayoría (local)':22s}{base_majority:>10.3f}{'—':>10s}")
    print(f"{'Modelo (ensemble)':22s}{model_acc_mkt:>10.3f}{model_ll_mkt:>10.3f}")
    print(f"{'Mercado (cierre)':22s}{mkt_acc:>10.3f}{mkt_ll:>10.3f}")

    # ── Guardar (misma interfaz que consume el router) ───────────────────────
    joblib.dump(lgbm, MODELS_DIR / "lgbm_outcome.pkl")
    joblib.dump(xgbc, MODELS_DIR / "xgb_outcome.pkl")

    imp = dict(sorted(
        zip(FEATURE_COLS, (lgbm.feature_importances_ / lgbm.feature_importances_.sum()).round(4).tolist()),
        key=lambda t: -t[1]))

    metrics = {
        "dataset": "LaLiga real — football-data.co.uk",
        "seasons": sorted(df["season"].unique().tolist()),
        "n_matches_total": int(len(df)),
        "n_train": int(len(Xtr)), "n_val": int(len(Xva)), "n_test": int(len(Xte)),
        "test_seasons": sorted(TEST_SEASONS & set(df["season"])),
        "ensemble_accuracy": round(float(acc), 4),
        "lgbm_accuracy": round(float(lgbm_acc), 4),
        "xgb_accuracy": round(float(xgb_acc), 4),
        "ensemble_f1_macro": round(float(f1m), 4),
        "ensemble_log_loss": round(float(ll), 4),
        "class_distribution": {k: round(float((df["result"] == k).mean()), 4) for k in ("H", "D", "A")},
        "baseline_majority_accuracy": round(float(base_majority), 4),
        "market_accuracy": round(float(mkt_acc), 4),
        "market_log_loss": round(float(mkt_ll), 4),
        "model_vs_market_accuracy_gap": round(float(model_acc_mkt - mkt_acc), 4),
        "model_vs_market_logloss_gap": round(float(model_ll_mkt - mkt_ll), 4),
        "honest_note": (
            "Métricas sobre partidos reales con split temporal. El mercado de cierre "
            "es el techo práctico; este modelo usa solo stats públicas de rendimiento."
        ),
        "feature_cols": FEATURE_COLS,
        "feature_importance": imp,
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("\nArtefactos guardados en models/.")


if __name__ == "__main__":
    main()
