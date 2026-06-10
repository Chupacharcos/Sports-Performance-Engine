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
            "ensemble_log_loss": _metrics.get("ensemble_log_loss"),
            "n_train": _metrics.get("n_train", 0),
            "n_test": _metrics.get("n_test", 0),
        },
        # Baselines honestos: el mercado de cierre es el techo práctico
        "baselines": {
            "majority_accuracy": _metrics.get("baseline_majority_accuracy"),
            "market_accuracy": _metrics.get("market_accuracy"),
            "market_log_loss": _metrics.get("market_log_loss"),
            "model_vs_market_gap": _metrics.get("model_vs_market_accuracy_gap"),
        },
        "dataset": _metrics.get("dataset"),
        "n_matches_total": _metrics.get("n_matches_total"),
        "test_seasons": _metrics.get("test_seasons"),
        "honest_note": _metrics.get("honest_note"),
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
            "elo": round(stats.get("elo", 1500), 0),
            "goals_avg": round(stats.get("gf_avg5", 0), 2),
            "conceded_avg": round(stats.get("ga_avg5", 0), 2),
            "sot_avg": round(stats.get("sot_avg5", 0), 2),
            "form_pts_last5": int(stats.get("form5", 0)),
        })
    teams.sort(key=lambda x: -x["elo"])
    return {"teams": teams}


@router.post("/sports/predict")
def predict_match(req: MatchPredictRequest):
    if _lgbm is None or _xgb is None:
        if not load_models():
            raise HTTPException(503, "Modelos no disponibles. Ejecuta train_all.py")

    team_data = _team_summary or {}

    def team_stats(team, is_home):
        ts = team_data.get(team, {})
        return {
            "elo": ts.get("elo", 1500.0),
            "gf_avg5": ts.get("gf_avg5", 1.3 if is_home else 1.0),
            "ga_avg5": ts.get("ga_avg5", 1.1 if is_home else 1.3),
            "sot_avg5": ts.get("sot_avg5", 4.5 if is_home else 3.8),
            "sota_avg5": ts.get("sota_avg5", 3.8 if is_home else 4.5),
            "form5": ts.get("form5", 6.0),
            "gf_side": ts.get("gf_home_avg" if is_home else "gf_away_avg", 1.2 if is_home else 0.9),
        }

    h, a = team_stats(req.home_team, True), team_stats(req.away_team, False)
    form_home = req.home_form_pts if req.home_form_pts is not None else h["form5"]
    form_away = req.away_form_pts if req.away_form_pts is not None else a["form5"]

    # Mismo orden que FEATURE_COLS del entrenamiento (fetch_real_data.FEATURE_COLS)
    features = [
        h["elo"], a["elo"], h["elo"] - a["elo"],
        h["gf_avg5"], h["ga_avg5"], h["sot_avg5"], h["sota_avg5"], form_home,
        a["gf_avg5"], a["ga_avg5"], a["sot_avg5"], a["sota_avg5"], form_away,
        h["gf_side"], a["gf_side"],
        form_home - form_away,
        h["sot_avg5"] - a["sot_avg5"],
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
        "home_stats": {"elo": round(h["elo"], 0), "goles_favor_5": round(h["gf_avg5"], 2),
                       "goles_contra_5": round(h["ga_avg5"], 2), "tiros_puerta_5": round(h["sot_avg5"], 2),
                       "forma_5": round(form_home, 1)},
        "away_stats": {"elo": round(a["elo"], 0), "goles_favor_5": round(a["gf_avg5"], 2),
                       "goles_contra_5": round(a["ga_avg5"], 2), "tiros_puerta_5": round(a["sot_avg5"], 2),
                       "forma_5": round(form_away, 1)},
        "disclaimer": "Predicción estadística sobre datos reales de LaLiga. No es asesoramiento de apuestas.",
    }


@router.get("/sports/matches/recent")
def get_recent_matches(limit: int = 20, competition: Optional[str] = None):
    if _matches_df is None:
        if not load_models():
            raise HTTPException(503, "Datos no disponibles")
    df = _matches_df.copy()
    if competition and "season" in df.columns:
        df = df[df["season"].astype(str).str.contains(competition, case=False, na=False)]
    df = df.sort_values("match_date", ascending=False).head(limit)

    matches = []
    for _, row in df.iterrows():
        matches.append({
            "match_id": int(row.get("match_id", 0)),
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_score": int(row.get("home_score", 0)),
            "away_score": int(row.get("away_score", 0)),
            "home_elo": round(float(row.get("home_elo", 1500))),
            "away_elo": round(float(row.get("away_elo", 1500))),
            "mkt_ph": round(float(row.get("mkt_ph", 0)), 3) if pd.notna(row.get("mkt_ph")) else None,
            "mkt_pa": round(float(row.get("mkt_pa", 0)), 3) if pd.notna(row.get("mkt_pa")) else None,
            "season": str(row.get("season", "")),
            "match_date": str(row.get("match_date", ""))[:10],
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
    # Predicción retroactiva con la forma real pre-partido del dataset
    pred_req = MatchPredictRequest(
        home_team=r["home_team"],
        away_team=r["away_team"],
        home_form_pts=float(r.get("home_form5", 6.0)),
        away_form_pts=float(r.get("away_form5", 6.0)),
    )
    try:
        prediction = predict_match(pred_req)
    except Exception:
        prediction = None

    outcome_map = {"H": "Victoria local", "D": "Empate", "A": "Victoria visitante"}
    return {
        "match_id": match_id,
        "home_team": r["home_team"],
        "away_team": r["away_team"],
        "result": {
            "home_score": int(r.get("home_score", 0)),
            "away_score": int(r.get("away_score", 0)),
            "outcome": outcome_map.get(str(r.get("result", "")), "—"),
        },
        "metrics": {
            "home_elo": round(float(r.get("home_elo", 1500))),
            "away_elo": round(float(r.get("away_elo", 1500))),
            "home_sot_avg5": round(float(r.get("home_sot_avg5", 0)), 2),
            "away_sot_avg5": round(float(r.get("away_sot_avg5", 0)), 2),
            "market_prob_home": round(float(r.get("mkt_ph", 0)), 3) if pd.notna(r.get("mkt_ph")) else None,
            "market_prob_draw": round(float(r.get("mkt_pd", 0)), 3) if pd.notna(r.get("mkt_pd")) else None,
            "market_prob_away": round(float(r.get("mkt_pa", 0)), 3) if pd.notna(r.get("mkt_pa")) else None,
        },
        "pre_match_prediction": prediction,
        "season": str(r.get("season", "")),
        "match_date": str(r.get("match_date", ""))[:10],
    }


@router.get("/sports/competitions")
def get_competitions():
    if _matches_df is None:
        if not load_models():
            return {"competitions": []}
    seasons = _matches_df["season"].dropna().unique().tolist() if "season" in _matches_df.columns else []
    return {"competitions": sorted(f"LaLiga {s[:2]}/{s[2:]}" for s in seasons)}


# ── Live data desde API-Football v3 ──────────────────────────────────────────

@router.get("/sports/live/upcoming")
def live_upcoming(competition: str = "laliga", n: int = 8):
    """Próximos N partidos en tiempo real desde API-Football."""
    try:
        from scripts.api_football import get_upcoming_fixtures, LEAGUE_LALIGA, LEAGUE_CHAMPIONS
        league = LEAGUE_CHAMPIONS if "champions" in competition.lower() else LEAGUE_LALIGA
        fixtures = get_upcoming_fixtures(league=league, n=n)
        out = []
        for f in fixtures:
            teams = f.get("teams", {})
            fix = f.get("fixture", {})
            out.append({
                "match_id": fix.get("id"),
                "home_team": teams.get("home", {}).get("name"),
                "away_team": teams.get("away", {}).get("name"),
                "home_logo": teams.get("home", {}).get("logo"),
                "away_logo": teams.get("away", {}).get("logo"),
                "match_date": fix.get("date"),
                "venue": fix.get("venue", {}).get("name"),
                "status": fix.get("status", {}).get("short"),
            })
        return {"competition": competition, "upcoming": out, "source": "API-Football v3"}
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(500, f"Error consultando API-Football: {e}")


@router.get("/sports/live/standings")
def live_standings(competition: str = "laliga"):
    """Tabla actual de la liga desde API-Football."""
    try:
        from scripts.api_football import get_standings, LEAGUE_LALIGA, LEAGUE_CHAMPIONS
        league = LEAGUE_CHAMPIONS if "champions" in competition.lower() else LEAGUE_LALIGA
        standings = get_standings(league=league)
        out = []
        for row in standings:
            team = row.get("team", {})
            out.append({
                "rank": row.get("rank"),
                "team": team.get("name"),
                "logo": team.get("logo"),
                "points": row.get("points"),
                "played": row.get("all", {}).get("played"),
                "wins": row.get("all", {}).get("win"),
                "draws": row.get("all", {}).get("draw"),
                "losses": row.get("all", {}).get("lose"),
                "goals_for": row.get("all", {}).get("goals", {}).get("for"),
                "goals_against": row.get("all", {}).get("goals", {}).get("against"),
            })
        return {"competition": competition, "standings": out, "source": "API-Football v3"}
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(500, f"Error consultando API-Football: {e}")
