#!/usr/bin/env python3
"""
Sports Performance Engine - Entrenamiento de modelos.
Entrena LightGBM + XGBoost ensemble para predicción de resultados.
"""
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import xgboost as xgb
import optuna
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.calibration import CalibratedClassifierCV
import shap

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "home_xg_for_avg", "home_xg_against_avg", "home_goals_avg",
    "home_conceded_avg", "home_form_pts", "home_shots_avg", "home_pressures_avg",
    "away_xg_for_avg", "away_xg_against_avg", "away_goals_avg",
    "away_conceded_avg", "away_form_pts", "away_shots_avg", "away_pressures_avg",
    "xg_diff_rolling", "form_diff", "shots_diff_avg",
]


def train_lgbm(X_train, y_train, X_val, y_val):
    """Entrena LightGBM con Optuna."""
    def objective(trial):
        params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": 5,
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        pred = model.predict_proba(X_val)
        return log_loss(y_val, pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30, show_progress_bar=False)
    best_params = study.best_params
    best_params.update({"objective": "multiclass", "num_class": 3,
                        "metric": "multi_logloss", "verbosity": -1})
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train, y_train)
    return model, study.best_value


def train_xgb(X_train, y_train, X_val, y_val):
    """Entrena XGBoost con Optuna."""
    def objective(trial):
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "verbosity": 0,
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
        }
        model = xgb.XGBClassifier(**params, use_label_encoder=False)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        pred = model.predict_proba(X_val)
        return log_loss(y_val, pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=25, show_progress_bar=False)
    best_params = study.best_params
    best_params.update({"objective": "multi:softprob", "num_class": 3, "verbosity": 0})
    model = xgb.XGBClassifier(**best_params, use_label_encoder=False)
    model.fit(X_train, y_train, verbose=False)
    return model, study.best_value


def main():
    print("=== Sports Performance Engine — Entrenamiento ===")

    # Cargar features
    features_path = DATA_DIR / "features.parquet"
    if not features_path.exists():
        print("Ejecutando data pipeline primero...")
        import subprocess, sys
        subprocess.run([sys.executable, str(Path(__file__).parent / "generate_data.py")])

    df = pd.read_parquet(features_path)
    print(f"Dataset: {len(df)} partidos")

    # Filtrar filas con NaN en features
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    df_clean = df[feature_cols + ["result"]].dropna()
    print(f"Tras eliminar NaN: {len(df_clean)} partidos")

    X = df_clean[feature_cols].values
    y = df_clean["result"].values

    # Split temporal (70/15/15)
    n = len(X)
    t1, t2 = int(n * 0.70), int(n * 0.85)
    X_train, y_train = X[:t1], y[:t1]
    X_val, y_val = X[t1:t2], y[t1:t2]
    X_test, y_test = X[t2:], y[t2:]

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"Distribución: Gana local={np.mean(y==0):.2%} Empate={np.mean(y==1):.2%} Gana visitante={np.mean(y==2):.2%}")

    # Entrenar LightGBM
    print("\n[1/2] Entrenando LightGBM (Optuna 30 trials)...")
    lgbm_model, lgbm_loss = train_lgbm(X_train, y_train, X_val, y_val)
    lgbm_pred = lgbm_model.predict(X_test)
    lgbm_acc = accuracy_score(y_test, lgbm_pred)
    print(f"  LightGBM → Accuracy: {lgbm_acc:.4f} | Log-Loss: {lgbm_loss:.4f}")

    # Entrenar XGBoost
    print("\n[2/2] Entrenando XGBoost (Optuna 25 trials)...")
    xgb_model, xgb_loss = train_xgb(X_train, y_train, X_val, y_val)
    xgb_pred = xgb_model.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    print(f"  XGBoost → Accuracy: {xgb_acc:.4f} | Log-Loss: {xgb_loss:.4f}")

    # Ensemble (media de probabilidades)
    lgbm_proba = lgbm_model.predict_proba(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)
    ensemble_proba = (lgbm_proba + xgb_proba) / 2
    ensemble_pred = np.argmax(ensemble_proba, axis=1)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    print(f"\n  Ensemble → Accuracy: {ensemble_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, ensemble_pred, target_names=["Local", "Empate", "Visitante"]))

    # Guardar modelos
    joblib.dump(lgbm_model, MODELS_DIR / "lgbm_outcome.pkl")
    joblib.dump(xgb_model, MODELS_DIR / "xgb_outcome.pkl")
    print(f"\n✓ Modelos guardados en {MODELS_DIR}")

    # SHAP feature importance (usando LightGBM)
    print("\nCalculando SHAP importances...")
    explainer = shap.TreeExplainer(lgbm_model)
    shap_values = explainer.shap_values(X_test[:100])
    mean_abs_shap = np.mean(np.abs(np.array(shap_values)).mean(axis=0), axis=0)
    feature_importance = sorted(zip(feature_cols, mean_abs_shap), key=lambda x: x[1], reverse=True)

    # Guardar métricas y metadata
    metrics = {
        "lgbm_accuracy": float(lgbm_acc),
        "xgb_accuracy": float(xgb_acc),
        "ensemble_accuracy": float(ensemble_acc),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_cols": feature_cols,
        "feature_importance": {k: float(v) for k, v in feature_importance},
        "class_distribution": {
            "home_win": float(np.mean(y == 0)),
            "draw": float(np.mean(y == 1)),
            "away_win": float(np.mean(y == 2)),
        },
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Guardar teams con sus stats
    team_stats_path = DATA_DIR / "team_stats.json"
    if team_stats_path.exists():
        with open(team_stats_path) as f:
            team_stats = json.load(f)
        # Calcular resumen por equipo
        team_summary = {}
        for team, stats in team_stats.items():
            def safe_tail_mean(lst, n=5):
                return float(np.mean(lst[-n:])) if lst else 0.0
            team_summary[team] = {
                "xg_for_avg": safe_tail_mean(stats.get("xg_for", [])),
                "xg_against_avg": safe_tail_mean(stats.get("xg_against", [])),
                "goals_avg": safe_tail_mean(stats.get("goals", [])),
                "conceded_avg": safe_tail_mean(stats.get("conceded", [])),
                "form_pts": sum(stats.get("points", [])[-5:]),
                "pressures_avg": safe_tail_mean(stats.get("pressures", [])),
            }
        with open(MODELS_DIR / "team_summary.json", "w") as f:
            json.dump(team_summary, f, indent=2)

    print(f"\n✓ Entrenamiento completado. Accuracy ensemble: {ensemble_acc:.4f}")
    print(f"✓ Modelos en: {MODELS_DIR}")


if __name__ == "__main__":
    main()
