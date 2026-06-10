#!/usr/bin/env python3
"""
Descarga partidos REALES de LaLiga desde football-data.co.uk (CSV públicos,
sin API key) y construye el dataset de features para el modelo.

Por cada partido se guardan también las cuotas de cierre (Pinnacle/B365/media
de mercado), que sirven como BASELINE honesto a comparar: si el modelo no se
acerca al log-loss de los bookies, se dice tal cual.

Features con higiene temporal estricta: todo lo que entra en la fila de un
partido se calcula SOLO con partidos anteriores (rolling y Elo pre-partido).

Salidas:
  data/matches_real.parquet   partidos crudos + cuotas
  data/features.parquet       features por partido + resultado + prob. implícitas
  models/team_summary.json    stats rolling actuales por equipo (para /predict)
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Temporadas 2010-11 .. 2025-26 (football-data usa "1011".."2526")
SEASONS = [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(2010, 2026)]
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/SP1.csv"

ELO_START, ELO_K, ELO_HOME_ADV = 1500.0, 20.0, 60.0
ROLL_N = 5  # ventana de forma

FEATURE_COLS = [
    "home_elo", "away_elo", "elo_diff",
    "home_gf_avg5", "home_ga_avg5", "home_sot_avg5", "home_sota_avg5", "home_form5",
    "away_gf_avg5", "away_ga_avg5", "away_sot_avg5", "away_sota_avg5", "away_form5",
    "home_gf_home_avg", "away_gf_away_avg",
    "form_diff", "sot_diff",
]


def download_all() -> pd.DataFrame:
    frames = []
    for season in SEASONS:
        url = BASE_URL.format(season=season)
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text), on_bad_lines="skip", encoding_errors="replace")
            if "HomeTeam" not in df.columns or len(df) < 100:
                print(f"  {season}: formato inesperado, skip")
                continue
            df["season"] = season
            frames.append(df)
            print(f"  {season}: {len(df)} partidos")
        except Exception as e:
            print(f"  {season}: ERROR {e}")
        time.sleep(0.4)  # cortesía con el servidor
    raw = pd.concat(frames, ignore_index=True)
    return raw


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
    keep = {
        "season": "season", "Date": "match_date",
        "HomeTeam": "home_team", "AwayTeam": "away_team",
        "FTHG": "home_score", "FTAG": "away_score", "FTR": "result",
        "HS": "home_shots", "AS": "away_shots",
        "HST": "home_sot", "AST": "away_sot",
        # Cuotas de cierre: Pinnacle (PS*) es la referencia "sharp"; si faltan, B365; si no, media de mercado
        "PSH": "ps_h", "PSD": "ps_d", "PSA": "ps_a",
        "B365H": "b365_h", "B365D": "b365_d", "B365A": "b365_a",
        "AvgH": "avg_h", "AvgD": "avg_d", "AvgA": "avg_a",
    }
    cols = {k: v for k, v in keep.items() if k in df.columns}
    df = df[list(cols)].rename(columns=cols)
    df = df.dropna(subset=["home_team", "away_team", "home_score", "away_score", "result"])
    df = df.sort_values("match_date").reset_index(drop=True)
    df["match_id"] = df.index

    # Probabilidades implícitas del mercado (quitar el margen normalizando)
    for src in ("ps", "b365", "avg"):
        h, d, a = f"{src}_h", f"{src}_d", f"{src}_a"
        if h in df.columns:
            inv = 1 / df[h] + 1 / df[d] + 1 / df[a]
            df[f"{src}_ph"] = (1 / df[h]) / inv
            df[f"{src}_pd"] = (1 / df[d]) / inv
            df[f"{src}_pa"] = (1 / df[a]) / inv
    # mejor fuente disponible por fila
    df["mkt_ph"] = df.get("ps_ph", pd.Series(np.nan, index=df.index)).fillna(
        df.get("b365_ph", pd.Series(np.nan, index=df.index))).fillna(
        df.get("avg_ph", pd.Series(np.nan, index=df.index)))
    df["mkt_pd"] = df.get("ps_pd", pd.Series(np.nan, index=df.index)).fillna(
        df.get("b365_pd", pd.Series(np.nan, index=df.index))).fillna(
        df.get("avg_pd", pd.Series(np.nan, index=df.index)))
    df["mkt_pa"] = df.get("ps_pa", pd.Series(np.nan, index=df.index)).fillna(
        df.get("b365_pa", pd.Series(np.nan, index=df.index))).fillna(
        df.get("avg_pa", pd.Series(np.nan, index=df.index)))
    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Recorre los partidos en orden temporal manteniendo el estado de cada
    equipo (Elo + historial rolling). La fila del partido usa SOLO estado previo."""
    elo: dict[str, float] = {}
    hist: dict[str, list[dict]] = {}

    def team_state(team: str, is_home: bool) -> dict:
        h = hist.get(team, [])
        last = h[-ROLL_N:]
        gf = np.mean([m["gf"] for m in last]) if last else np.nan
        ga = np.mean([m["ga"] for m in last]) if last else np.nan
        sot = np.mean([m["sot"] for m in last]) if last else np.nan
        sota = np.mean([m["sota"] for m in last]) if last else np.nan
        form = sum(m["pts"] for m in last) if last else np.nan
        side = [m for m in h if m["home"] == is_home][-ROLL_N:]
        gf_side = np.mean([m["gf"] for m in side]) if side else np.nan
        return {"gf": gf, "ga": ga, "sot": sot, "sota": sota, "form": form, "gf_side": gf_side}

    rows = []
    for _, m in df.iterrows():
        ht, at = m["home_team"], m["away_team"]
        he, ae = elo.get(ht, ELO_START), elo.get(at, ELO_START)
        hs, as_ = team_state(ht, True), team_state(at, False)

        rows.append({
            "match_id": m["match_id"], "season": m["season"], "match_date": m["match_date"],
            "home_team": ht, "away_team": at,
            "home_score": m["home_score"], "away_score": m["away_score"], "result": m["result"],
            "mkt_ph": m.get("mkt_ph"), "mkt_pd": m.get("mkt_pd"), "mkt_pa": m.get("mkt_pa"),
            "home_elo": he, "away_elo": ae, "elo_diff": he - ae,
            "home_gf_avg5": hs["gf"], "home_ga_avg5": hs["ga"],
            "home_sot_avg5": hs["sot"], "home_sota_avg5": hs["sota"], "home_form5": hs["form"],
            "away_gf_avg5": as_["gf"], "away_ga_avg5": as_["ga"],
            "away_sot_avg5": as_["sot"], "away_sota_avg5": as_["sota"], "away_form5": as_["form"],
            "home_gf_home_avg": hs["gf_side"], "away_gf_away_avg": as_["gf_side"],
            "form_diff": (hs["form"] - as_["form"]) if not (np.isnan(hs["form"]) or np.isnan(as_["form"])) else np.nan,
            "sot_diff": (hs["sot"] - as_["sot"]) if not (np.isnan(hs["sot"]) or np.isnan(as_["sot"])) else np.nan,
        })

        # ── actualizar estado DESPUÉS de registrar la fila ──
        exp_home = 1 / (1 + 10 ** (-((he - ae + ELO_HOME_ADV) / 400)))
        score = 1.0 if m["result"] == "H" else (0.5 if m["result"] == "D" else 0.0)
        elo[ht] = he + ELO_K * (score - exp_home)
        elo[at] = ae + ELO_K * ((1 - score) - (1 - exp_home))

        pts_h = 3 if m["result"] == "H" else (1 if m["result"] == "D" else 0)
        pts_a = 3 if m["result"] == "A" else (1 if m["result"] == "D" else 0)
        hist.setdefault(ht, []).append({"gf": m["home_score"], "ga": m["away_score"],
                                        "sot": m.get("home_sot", np.nan), "sota": m.get("away_sot", np.nan),
                                        "pts": pts_h, "home": True})
        hist.setdefault(at, []).append({"gf": m["away_score"], "ga": m["home_score"],
                                        "sot": m.get("away_sot", np.nan), "sota": m.get("home_sot", np.nan),
                                        "pts": pts_a, "home": False})

    feats = pd.DataFrame(rows)
    # estado actual por equipo para /predict (equipos de las 2 últimas temporadas)
    recent_teams = set(df[df["season"].isin(df["season"].unique()[-2:])]["home_team"])
    summary = {}
    for t in sorted(recent_teams):
        st_h, st_a = team_state(t, True), team_state(t, False)
        summary[t] = {
            "elo": round(elo.get(t, ELO_START), 1),
            "gf_avg5": round(float(np.nan_to_num(st_h["gf"], nan=1.2)), 2),
            "ga_avg5": round(float(np.nan_to_num(st_h["ga"], nan=1.2)), 2),
            "sot_avg5": round(float(np.nan_to_num(st_h["sot"], nan=4.0)), 2),
            "sota_avg5": round(float(np.nan_to_num(st_h["sota"], nan=4.0)), 2),
            "form5": round(float(np.nan_to_num(st_h["form"], nan=6.0)), 1),
            "gf_home_avg": round(float(np.nan_to_num(st_h["gf_side"], nan=1.3)), 2),
            "gf_away_avg": round(float(np.nan_to_num(st_a["gf_side"], nan=1.0)), 2),
        }
    return feats, summary


def main():
    print("Descargando temporadas de LaLiga (football-data.co.uk)…")
    raw = download_all()
    df = clean(raw)
    print(f"Total partidos reales: {len(df)} ({df['season'].nunique()} temporadas, "
          f"{df['match_date'].min():%Y-%m-%d} → {df['match_date'].max():%Y-%m-%d})")
    df.to_parquet(DATA_DIR / "matches_real.parquet")

    feats, summary = build_features(df)
    # descartar filas sin historial suficiente (primeras jornadas de equipo nuevo)
    n0 = len(feats)
    feats = feats.dropna(subset=FEATURE_COLS)
    print(f"Features: {len(feats)} partidos con historial completo (descartados {n0 - len(feats)})")
    feats.to_parquet(DATA_DIR / "features.parquet")

    with open(MODELS_DIR / "team_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"team_summary.json: {len(summary)} equipos")


if __name__ == "__main__":
    main()
