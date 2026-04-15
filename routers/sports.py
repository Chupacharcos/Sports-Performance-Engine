"""
Sports Performance Engine — FastAPI Router
"""
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List

router = APIRouter(prefix="/ml", tags=["sports"])

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Cargar modelos al arrancar
_lgbm = None
_xgb = None
_metrics = None
_team_summary = None
_feature_cols = None
_matches_df = None


def load_models():
    global _lgbm, _xgb, _metrics, _team_summary, _feature_cols, _matches_df
    if not (MODELS_DIR / "lgbm_outcome.pkl").exists():
        return False
    try:
        _lgbm = joblib.load(MODELS_DIR / "lgbm_outcome.pkl")
        _xgb = joblib.load(MODELS_DIR / "xgb_outcome.pkl")
        with open(MODELS_DIR / "metrics.json") as f:
            _metrics = json.load(f)
        _feature_cols = _metrics.get("feature_cols", [])
        if (MODELS_DIR / "team_summary.json").exists():
            with open(MODELS_DIR / "team_summary.json") as f:
                _team_summary = json.load(f)
        if (DATA_DIR / "features.parquet").exists():
            _matches_df = pd.read_parquet(DATA_DIR / "features.parquet")
        return True
    except Exception as e:
        print(f"Error cargando modelos: {e}")
        return False


class MatchPredictRequest(BaseModel):
    home_team: str
    away_team: str
    home_form_pts: Optional[float] = None
    away_form_pts: Optional[float] = None


@router.on_event("startup")
async def startup():
    load_models()


@router.get("/sports/health")
def health():
    models_loaded = _lgbm is not None
    return {
        "status": "ok" if models_loaded else "no_models",
        "models_loaded": models_loaded,
        "service": "Sports Performance Engine",
    }


@router.get("/sports/stats")
def get_stats():
    if not _metrics:
        if not load_models():
            raise HTTPException(503, "Modelos no disponibles. Ejecuta train_all.py")
    return {
        "model_performance": {
            "lgbm_accuracy": round(_metrics.get("lgbm_accuracy", 0), 4),
            "xgb_accuracy": round(_metrics.get("xgb_accuracy", 0), 4),
            "ensemble_accuracy": round(_metrics.get("ensemble_accuracy", 0), 4),
            "n_train": _metrics.get("n_train", 0),
        },
        "class_distribution": _metrics.get("class_distribution", {}),
        "feature_importance": dict(list(_metrics.get("feature_importance", {}).items())[:8]),
    }


@router.get("/sports/teams")
def get_teams():
    if _team_summary is None:
        if not load_models():
            raise HTTPException(503, "Modelos no disponibles")
    teams = []
    for name, stats in (_team_summary or {}).items():
        teams.append({
            "team": name,
            "xg_for_avg": round(stats.get("xg_for_avg", 0), 2),
            "xg_against_avg": round(stats.get("xg_against_avg", 0), 2),
            "goals_avg": round(stats.get("goals_avg", 0), 2),
            "conceded_avg": round(stats.get("conceded_avg", 0), 2),
            "form_pts_last5": int(stats.get("form_pts", 0)),
            "pressures_avg": round(stats.get("pressures_avg", 0), 1),
        })
    teams.sort(key=lambda x: -x["form_pts_last5"])
    return {"teams": teams}


@router.post("/sports/predict")
def predict_match(req: MatchPredictRequest):
    if _lgbm is None or _xgb is None:
        if not load_models():
            raise HTTPException(503, "Modelos no disponibles. Ejecuta train_all.py")

    team_data = _team_summary or {}

    def get_team_features(team, is_home=True):
        ts = team_data.get(team, {})
        return {
            "xg_for_avg": ts.get("xg_for_avg", 1.2 if is_home else 0.9),
            "xg_against_avg": ts.get("xg_against_avg", 0.9 if is_home else 1.1),
            "goals_avg": ts.get("goals_avg", 1.4 if is_home else 1.1),
            "conceded_avg": ts.get("conceded_avg", 1.0 if is_home else 1.2),
            "form_pts": ts.get("form_pts", 7.5) / 15,
            "shots_avg": 11.0 if is_home else 9.0,
            "pressures_avg": ts.get("pressures_avg", 35.0 if is_home else 28.0),
        }

    home_f = get_team_features(req.home_team, True)
    away_f = get_team_features(req.away_team, False)

    form_home = req.home_form_pts / 15 if req.home_form_pts is not None else home_f["form_pts"]
    form_away = req.away_form_pts / 15 if req.away_form_pts is not None else away_f["form_pts"]

    features = [
        home_f["xg_for_avg"], home_f["xg_against_avg"], home_f["goals_avg"],
        home_f["conceded_avg"], form_home, home_f["shots_avg"], home_f["pressures_avg"],
        away_f["xg_for_avg"], away_f["xg_against_avg"], away_f["goals_avg"],
        away_f["conceded_avg"], form_away, away_f["shots_avg"], away_f["pressures_avg"],
        home_f["xg_for_avg"] - away_f["xg_for_avg"],
        form_home - form_away,
        home_f["shots_avg"] - away_f["shots_avg"],
    ]

    X = np.array([features])
    lgbm_proba = _lgbm.predict_proba(X)[0]
    xgb_proba = _xgb.predict_proba(X)[0]
    ensemble_proba = (lgbm_proba + xgb_proba) / 2

    labels = ["Victoria local", "Empate", "Victoria visitante"]
    prediction_idx = int(np.argmax(ensemble_proba))

    return {
        "home_team": req.home_team,
        "away_team": req.away_team,
        "prediction": labels[prediction_idx],
        "confidence": round(float(ensemble_proba[prediction_idx]), 3),
        "probabilities": {
            "home_win": round(float(ensemble_proba[0]), 3),
            "draw": round(float(ensemble_proba[1]), 3),
            "away_win": round(float(ensemble_proba[2]), 3),
        },
        "home_stats": {k: round(v, 2) for k, v in home_f.items()},
        "away_stats": {k: round(v, 2) for k, v in away_f.items()},
        "expected_goals": {
            "home_xg": round(home_f["xg_for_avg"] * 0.9, 2),
            "away_xg": round(away_f["xg_for_avg"] * 0.85, 2),
        },
        "disclaimer": "Predicción estadística. No es asesoramiento de apuestas.",
    }


@router.get("/sports/matches/recent")
def get_recent_matches(limit: int = 20, competition: Optional[str] = None):
    if _matches_df is None:
        if not load_models():
            raise HTTPException(503, "Datos no disponibles")
    df = _matches_df.copy()
    if competition and "competition" in df.columns:
        df = df[df["competition"].str.contains(competition, case=False, na=False)]
    df = df.sort_values("match_date", ascending=False).head(limit)

    matches = []
    for _, row in df.iterrows():
        matches.append({
            "match_id": int(row.get("match_id", 0)),
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_score": int(row.get("home_score", 0)),
            "away_score": int(row.get("away_score", 0)),
            "home_xg": round(float(row.get("home_xg", 0)), 2),
            "away_xg": round(float(row.get("away_xg", 0)), 2),
            "competition": row.get("competition", ""),
            "match_date": str(row.get("match_date", "")),
        })
    return {"matches": matches, "total": len(matches)}


@router.get("/sports/match/{match_id}")
def get_match_detail(match_id: int):
    if _matches_df is None:
        if not load_models():
            raise HTTPException(503, "Datos no disponibles")
    row = _matches_df[_matches_df["match_id"] == match_id]
    if len(row) == 0:
        raise HTTPException(404, f"Partido {match_id} no encontrado")
    r = row.iloc[0].to_dict()
    # Hacer predicción retroactiva
    pred_req = MatchPredictRequest(
        home_team=r["home_team"],
        away_team=r["away_team"],
        home_form_pts=float(r.get("home_form_pts", 7.5)),
        away_form_pts=float(r.get("away_form_pts", 6.0)),
    )
    try:
        prediction = predict_match(pred_req)
    except Exception:
        prediction = None

    return {
        "match_id": match_id,
        "home_team": r["home_team"],
        "away_team": r["away_team"],
        "result": {
            "home_score": int(r.get("home_score", 0)),
            "away_score": int(r.get("away_score", 0)),
            "outcome": ["Victoria local", "Empate", "Victoria visitante"][int(r.get("result", 0))],
        },
        "metrics": {
            "home_xg": round(float(r.get("home_xg", 0)), 3),
            "away_xg": round(float(r.get("away_xg", 0)), 3),
            "home_shots": int(r.get("home_shots_match", 0)),
            "away_shots": int(r.get("away_shots_match", 0)),
            "home_possession": round(float(r.get("home_possession", 50)), 1),
            "home_pressures": int(r.get("home_pressures", 0)),
            "away_pressures": int(r.get("away_pressures", 0)),
        },
        "pre_match_prediction": prediction,
        "competition": r.get("competition", ""),
        "match_date": str(r.get("match_date", "")),
    }


@router.get("/sports/competitions")
def get_competitions():
    if _matches_df is None:
        if not load_models():
            return {"competitions": ["LaLiga 2022/23", "Champions League 2022/23"]}
    comps = _matches_df["competition"].dropna().unique().tolist() if "competition" in _matches_df.columns else []
    return {"competitions": sorted(comps)}
