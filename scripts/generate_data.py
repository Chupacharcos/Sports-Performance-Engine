#!/usr/bin/env python3
"""
Sports Performance Engine - Data Pipeline
Descarga datos de StatsBomb Open Data (LaLiga + Champions League)
y genera features para entrenamiento de modelos de rendimiento.
"""
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

def load_statsbomb_data():
    """Descarga partidos y eventos de StatsBomb Open Data."""
    try:
        from statsbombpy import sb
    except ImportError:
        print("statsbombpy no instalado. Generando datos sintéticos...")
        return None, None

    competitions = sb.competitions()
    # LaLiga (id=11) + Champions League (id=16)
    target_comps = [
        (11, 90),  # LaLiga 2022/23
        (11, 281), # LaLiga 2021/22
        (16, 90),  # Champions 2022/23
        (16, 4),   # Champions 2021/22
    ]
    all_matches = []
    all_events = []
    for comp_id, season_id in target_comps:
        try:
            matches = sb.matches(competition_id=comp_id, season_id=season_id)
            if matches is None or len(matches) == 0:
                continue
            print(f"Comp {comp_id} Season {season_id}: {len(matches)} partidos")
            all_matches.append(matches)
            # Tomar hasta 50 partidos por competición para no exceder RAM
            match_ids = matches["match_id"].tolist()[:50]
            for mid in match_ids:
                try:
                    ev = sb.events(match_id=mid)
                    if ev is not None and len(ev) > 0:
                        ev["match_id"] = mid
                        all_events.append(ev[["match_id","type","period","minute",
                                              "team","player","location","shot_statsbomb_xg",
                                              "pass_recipient","carry_end_location",
                                              "under_pressure","duel_type"
                                              ] + [c for c in ev.columns
                                                   if c.startswith("shot_") or
                                                   c.startswith("pass_") or
                                                   c.startswith("dribble_") or
                                                   c.startswith("carry_") or
                                                   c.startswith("pressure_")
                                                   if c in ev.columns]])
                except Exception:
                    pass
        except Exception as e:
            print(f"Error comp {comp_id} season {season_id}: {e}")

    if all_matches:
        matches_df = pd.concat(all_matches, ignore_index=True)
        matches_df.to_parquet(DATA_DIR / "matches_raw.parquet")
        print(f"Total partidos: {len(matches_df)}")
        if all_events:
            events_df = pd.concat(all_events, ignore_index=True)
            events_df.to_parquet(DATA_DIR / "events_raw.parquet")
            print(f"Total eventos: {len(events_df)}")
        return matches_df, pd.concat(all_events, ignore_index=True) if all_events else None
    return None, None


def compute_team_stats(matches_df, events_df):
    """Genera features de equipo por partido."""
    records = []
    for _, match in matches_df.iterrows():
        home_team = match["home_team"]
        away_team = match["away_team"]
        home_score = match["home_score"]
        away_score = match["away_score"]
        match_id = match["match_id"]

        # Resultado: 0=local gana, 1=empate, 2=visitante gana
        if home_score > away_score:
            result = 0
        elif home_score == away_score:
            result = 1
        else:
            result = 2

        row = {
            "match_id": match_id,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score,
            "result": result,
            "competition": match.get("competition", {}).get("competition_name", "Unknown") if isinstance(match.get("competition"), dict) else "Unknown",
            "season": match.get("season", {}).get("season_name", "Unknown") if isinstance(match.get("season"), dict) else "Unknown",
            "match_date": str(match.get("match_date", "")),
        }

        # Agregar xG si disponemos de eventos
        if events_df is not None and match_id in events_df["match_id"].values:
            ev = events_df[events_df["match_id"] == match_id]
            shots = ev[ev["type"] == "Shot"] if "type" in ev.columns else pd.DataFrame()
            if len(shots) > 0 and "shot_statsbomb_xg" in shots.columns and "team" in shots.columns:
                row["home_xg"] = shots[shots["team"] == home_team]["shot_statsbomb_xg"].sum()
                row["away_xg"] = shots[shots["team"] == away_team]["shot_statsbomb_xg"].sum()
                row["home_shots"] = len(shots[shots["team"] == home_team])
                row["away_shots"] = len(shots[shots["team"] == away_team])
            else:
                row["home_xg"] = np.nan
                row["away_xg"] = np.nan
                row["home_shots"] = np.nan
                row["away_shots"] = np.nan

            # Presiones
            pressures = ev[ev["type"] == "Pressure"] if "type" in ev.columns else pd.DataFrame()
            if len(pressures) > 0 and "team" in pressures.columns:
                row["home_pressures"] = len(pressures[pressures["team"] == home_team])
                row["away_pressures"] = len(pressures[pressures["team"] == away_team])
            else:
                row["home_pressures"] = np.nan
                row["away_pressures"] = np.nan
        else:
            row.update({
                "home_xg": np.nan, "away_xg": np.nan,
                "home_shots": np.nan, "away_shots": np.nan,
                "home_pressures": np.nan, "away_pressures": np.nan,
            })

        records.append(row)

    return pd.DataFrame(records)


def generate_synthetic_data(n_matches=2000):
    """Genera datos sintéticos realistas de fútbol basados en distribuciones reales."""
    print("Generando dataset sintético de fútbol...")
    rng = np.random.RandomState(42)

    teams = [
        "Real Madrid", "Barcelona", "Atlético de Madrid", "Sevilla",
        "Real Betis", "Real Sociedad", "Valencia", "Athletic Club",
        "Villarreal", "Osasuna", "Getafe", "Rayo Vallecano",
        "Celta de Vigo", "Girona", "Las Palmas", "Alavés",
        "Cádiz", "Almería", "Granada", "Mallorca",
        # Champions
        "Manchester City", "Bayern Munich", "PSG", "Inter Milan",
        "Arsenal", "Napoli", "Benfica", "Borussia Dortmund",
    ]
    team_strengths = {
        "Real Madrid": 0.85, "Barcelona": 0.82, "Atlético de Madrid": 0.78,
        "Manchester City": 0.90, "Bayern Munich": 0.87, "PSG": 0.84,
        "Sevilla": 0.70, "Real Betis": 0.68, "Real Sociedad": 0.67,
        "Valencia": 0.63, "Athletic Club": 0.65, "Villarreal": 0.66,
        "Arsenal": 0.80, "Napoli": 0.78, "Inter Milan": 0.81,
        "Borussia Dortmund": 0.77, "Benfica": 0.73,
        "Osasuna": 0.57, "Getafe": 0.55, "Rayo Vallecano": 0.54,
        "Celta de Vigo": 0.58, "Girona": 0.62, "Las Palmas": 0.50,
        "Alavés": 0.49, "Cádiz": 0.47, "Almería": 0.46,
        "Granada": 0.45, "Mallorca": 0.52,
    }

    records = []
    competitions = ["LaLiga 2022/23", "LaLiga 2023/24", "Champions League 2022/23", "Champions League 2023/24"]
    for i in range(n_matches):
        home_team = rng.choice(teams)
        away_team = rng.choice([t for t in teams if t != home_team])
        competition = rng.choice(competitions)

        hs = team_strengths.get(home_team, 0.6)
        aws = team_strengths.get(away_team, 0.6)
        home_advantage = 0.07

        # xG simulado
        home_xg = rng.gamma(shape=max(0.5, (hs - aws + home_advantage + 0.5) * 2), scale=0.7)
        away_xg = rng.gamma(shape=max(0.5, (aws - hs + 0.5) * 2), scale=0.65)
        home_xg = np.clip(home_xg, 0, 4.5)
        away_xg = np.clip(away_xg, 0, 4.0)

        # Goles desde xG (distribución Poisson)
        home_goals = rng.poisson(lam=home_xg * 0.85 + 0.1)
        away_goals = rng.poisson(lam=away_xg * 0.85 + 0.05)

        # Shots (correlacionados con xG)
        home_shots = int(np.clip(rng.normal(home_xg * 4 + 4, 2), 2, 25))
        away_shots = int(np.clip(rng.normal(away_xg * 4 + 3.5, 2), 1, 22))
        home_shots_target = int(np.clip(rng.normal(home_xg * 2.5 + 2, 1.5), 0, home_shots))
        away_shots_target = int(np.clip(rng.normal(away_xg * 2.5 + 1.8, 1.5), 0, away_shots))

        # Presiones (correlacionadas con estilo del equipo)
        home_pressures = int(np.clip(rng.normal(hs * 50 + 10, 8), 5, 80))
        away_pressures = int(np.clip(rng.normal(aws * 45 + 8, 8), 3, 75))

        # Posesión
        total_passes = rng.poisson(lam=350 + hs * 200)
        away_passes = rng.poisson(lam=280 + aws * 180)
        home_possession = round(total_passes / (total_passes + away_passes) * 100, 1)

        # Pases progresivos
        home_prog_passes = int(np.clip(rng.normal(hs * 30 + 20, 8), 5, 80))
        away_prog_passes = int(np.clip(rng.normal(aws * 25 + 15, 7), 3, 70))

        # Faltas, tarjetas
        home_fouls = rng.poisson(lam=11)
        away_fouls = rng.poisson(lam=12)
        home_yellows = rng.poisson(lam=1.8)
        away_yellows = rng.poisson(lam=2.0)

        # Resultado
        if home_goals > away_goals:
            result = 0
        elif home_goals == away_goals:
            result = 1
        else:
            result = 2

        # Rolling form (simulated for current match context)
        home_form_pts = rng.binomial(n=15, p=hs * 0.7) / 15 * 15
        away_form_pts = rng.binomial(n=15, p=aws * 0.6) / 15 * 15

        match_date = pd.Timestamp("2022-08-01") + pd.Timedelta(days=int(rng.uniform(0, 700)))

        records.append({
            "match_id": i + 10000,
            "competition": competition,
            "match_date": str(match_date.date()),
            "home_team": home_team,
            "away_team": away_team,
            "home_score": int(home_goals),
            "away_score": int(away_goals),
            "result": result,
            "home_xg": round(home_xg, 3),
            "away_xg": round(away_xg, 3),
            "home_shots": home_shots,
            "away_shots": away_shots,
            "home_shots_target": home_shots_target,
            "away_shots_target": away_shots_target,
            "home_pressures": home_pressures,
            "away_pressures": away_pressures,
            "home_possession": home_possession,
            "home_prog_passes": home_prog_passes,
            "away_prog_passes": away_prog_passes,
            "home_fouls": home_fouls,
            "away_fouls": away_fouls,
            "home_yellows": home_yellows,
            "away_yellows": away_yellows,
            "home_form_pts": round(home_form_pts, 1),
            "away_form_pts": round(away_form_pts, 1),
            "home_strength": hs,
            "away_strength": aws,
        })

    df = pd.DataFrame(records)
    df.to_parquet(DATA_DIR / "matches_synthetic.parquet")
    print(f"Dataset sintético: {len(df)} partidos guardado.")
    return df


def engineer_features(df):
    """Genera features de rolling para el modelo de predicción."""
    df = df.sort_values("match_date").reset_index(drop=True)
    df["home_xg"] = df["home_xg"].fillna(df["home_shots"] / 10 if "home_shots" in df.columns else 1.2)
    df["away_xg"] = df["away_xg"].fillna(df["away_shots"] / 10 if "away_shots" in df.columns else 0.9)

    # Calcular rolling stats por equipo (últimos 5 partidos)
    team_stats = {}
    records = []
    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        def get_rolling(team, n=5):
            stats = team_stats.get(team, {"xg_for": [], "xg_against": [],
                                         "shots_for": [], "goals": [], "conceded": [],
                                         "points": [], "pressures": []})
            def safe_mean(lst):
                return np.mean(lst[-n:]) if lst else 1.0
            return {
                "xg_for_avg": safe_mean(stats["xg_for"]),
                "xg_against_avg": safe_mean(stats["xg_against"]),
                "shots_avg": safe_mean(stats["shots_for"]),
                "goals_avg": safe_mean(stats["goals"]),
                "conceded_avg": safe_mean(stats["conceded"]),
                "form_pts": sum(stats["points"][-n:]) / n if stats["points"] else 1.0,
                "pressures_avg": safe_mean(stats["pressures"]),
            }

        hs = get_rolling(home)
        aws = get_rolling(away)
        feat = {
            "match_id": row["match_id"],
            "home_team": home,
            "away_team": away,
            "result": row["result"],
            "home_score": row["home_score"],
            "away_score": row["away_score"],
            "competition": row.get("competition", ""),
            "match_date": row["match_date"],
            "home_xg": row.get("home_xg", 1.2),
            "away_xg": row.get("away_xg", 0.9),
            # Features de rolling
            "home_xg_for_avg": hs["xg_for_avg"],
            "home_xg_against_avg": hs["xg_against_avg"],
            "home_goals_avg": hs["goals_avg"],
            "home_conceded_avg": hs["conceded_avg"],
            "home_form_pts": hs["form_pts"],
            "home_shots_avg": hs["shots_avg"],
            "home_pressures_avg": hs["pressures_avg"],
            "away_xg_for_avg": aws["xg_for_avg"],
            "away_xg_against_avg": aws["xg_against_avg"],
            "away_goals_avg": aws["goals_avg"],
            "away_conceded_avg": aws["conceded_avg"],
            "away_form_pts": aws["form_pts"],
            "away_shots_avg": aws["shots_avg"],
            "away_pressures_avg": aws["pressures_avg"],
            # Diferencias
            "xg_diff_rolling": hs["xg_for_avg"] - aws["xg_for_avg"],
            "form_diff": hs["form_pts"] - aws["form_pts"],
            "shots_diff_avg": hs["shots_avg"] - aws["shots_avg"],
            # Features directas del partido (para análisis post-match)
            "home_shots_match": row.get("home_shots", 10),
            "away_shots_match": row.get("away_shots", 8),
            "home_possession": row.get("home_possession", 50),
            "home_pressures": row.get("home_pressures", 35),
            "away_pressures": row.get("away_pressures", 30),
            "home_yellows": row.get("home_yellows", 2),
            "away_yellows": row.get("away_yellows", 2),
        }
        records.append(feat)

        # Actualizar stats del equipo local
        if home not in team_stats:
            team_stats[home] = {"xg_for": [], "xg_against": [], "shots_for": [],
                                "goals": [], "conceded": [], "points": [], "pressures": []}
        team_stats[home]["xg_for"].append(row.get("home_xg", 1.2))
        team_stats[home]["xg_against"].append(row.get("away_xg", 0.9))
        team_stats[home]["goals"].append(row["home_score"])
        team_stats[home]["conceded"].append(row["away_score"])
        team_stats[home]["shots_for"].append(row.get("home_shots", 10))
        team_stats[home]["pressures"].append(row.get("home_pressures", 35))
        pts = 3 if row["result"] == 0 else (1 if row["result"] == 1 else 0)
        team_stats[home]["points"].append(pts)

        # Actualizar stats del equipo visitante
        if away not in team_stats:
            team_stats[away] = {"xg_for": [], "xg_against": [], "shots_for": [],
                                "goals": [], "conceded": [], "points": [], "pressures": []}
        team_stats[away]["xg_for"].append(row.get("away_xg", 0.9))
        team_stats[away]["xg_against"].append(row.get("home_xg", 1.2))
        team_stats[away]["goals"].append(row["away_score"])
        team_stats[away]["conceded"].append(row["home_score"])
        team_stats[away]["shots_for"].append(row.get("away_shots", 8))
        team_stats[away]["pressures"].append(row.get("away_pressures", 30))
        pts_away = 3 if row["result"] == 2 else (1 if row["result"] == 1 else 0)
        team_stats[away]["points"].append(pts_away)

    features_df = pd.DataFrame(records)
    features_df.to_parquet(DATA_DIR / "features.parquet")
    print(f"Features generadas: {len(features_df)} filas, {len(features_df.columns)} columnas")

    # Guardar también team_stats como JSON
    team_stats_serializable = {
        team: {k: [float(x) for x in v] for k, v in stats.items()}
        for team, stats in team_stats.items()
    }
    with open(DATA_DIR / "team_stats.json", "w") as f:
        json.dump(team_stats_serializable, f)

    return features_df, team_stats


if __name__ == "__main__":
    print("=== Sports Performance Engine — Data Pipeline ===")
    matches_df = None
    events_df = None

    # Intentar StatsBomb real
    try:
        matches_df, events_df = load_statsbomb_data()
    except Exception as e:
        print(f"StatsBomb falló: {e}")

    if matches_df is None or len(matches_df) < 50:
        print("Usando datos sintéticos...")
        matches_df = generate_synthetic_data(n_matches=3000)

    features_df, team_stats = engineer_features(matches_df)
    print(f"\nResumen features: {features_df.describe()}")
    print("\n✓ Data pipeline completado.")
