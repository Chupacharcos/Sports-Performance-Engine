"""
Cliente API-Football v3 — datos en tiempo real de LaLiga / Champions.

Endpoints usados:
  - /fixtures           Próximos partidos + resultados recientes
  - /teams/statistics   Stats de equipo en la liga actual
  - /standings          Tabla de la liga

API key se lee de env API_FOOTBALL_KEY. Cache 1h para no agotar el plan free.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests

API_BASE = "https://v3.football.api-sports.io"
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "api_football"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_S = 3600

LEAGUE_LALIGA = 140
LEAGUE_CHAMPIONS = 2


def _headers() -> dict:
    key = os.getenv("API_FOOTBALL_KEY")
    if not key:
        raise RuntimeError("API_FOOTBALL_KEY no configurada en .env")
    return {"x-apisports-key": key}


def _cached_get(endpoint: str, params: dict) -> dict:
    """GET con cache en disco (1h). Devuelve dict vacío en error de red."""
    cache_key = endpoint.replace("/", "_") + "_" + "_".join(f"{k}={v}" for k, v in sorted(params.items()))
    cache_file = CACHE_DIR / f"{cache_key}.json"

    if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < CACHE_TTL_S:
        try:
            return json.loads(cache_file.read_text())
        except Exception:
            pass

    try:
        r = requests.get(f"{API_BASE}{endpoint}", headers=_headers(), params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        cache_file.write_text(json.dumps(data))
        return data
    except Exception as e:
        # Si hay cache stale, mejor que nada
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text())
            except Exception:
                pass
        return {"errors": [str(e)], "response": []}


def get_upcoming_fixtures(league: int = LEAGUE_LALIGA, season: int | None = None, n: int = 10) -> list[dict]:
    """Partidos en una ventana de 30 días desde hoy.

    El plan free no soporta `next`/`last` — usamos rango from/to explícito y
    filtramos en código. Si la temporada está terminada, devolvemos los últimos
    partidos jugados (los más recientes del rango).
    """
    from datetime import datetime, timedelta
    season = season or _current_season()
    today = datetime.utcnow().date()
    # Ventana: 30 días pasados + 30 futuros para cubrir transición de temporada
    from_d = (today - timedelta(days=30)).isoformat()
    to_d = (today + timedelta(days=30)).isoformat()
    data = _cached_get("/fixtures", {"league": league, "season": season, "from": from_d, "to": to_d})
    fixtures = data.get("response", [])
    # Si todos los partidos del rango son pasados, devuelve los últimos N por fecha
    fixtures.sort(key=lambda f: f.get("fixture", {}).get("date", ""), reverse=False)
    upcoming = [f for f in fixtures if f.get("fixture", {}).get("status", {}).get("short") in ("NS", "TBD")]
    if upcoming:
        return upcoming[:n]
    # Fallback: últimos partidos jugados
    return fixtures[-n:] if fixtures else []


def get_team_statistics(team_id: int, league: int = LEAGUE_LALIGA, season: int | None = None) -> dict:
    """Stats de equipo: goles, posesión, xG, etc."""
    season = season or _current_season()
    data = _cached_get("/teams/statistics", {"team": team_id, "league": league, "season": season})
    return data.get("response", {})


def get_standings(league: int = LEAGUE_LALIGA, season: int | None = None) -> list[dict]:
    """Tabla de la liga."""
    season = season or _current_season()
    data = _cached_get("/standings", {"league": league, "season": season})
    response = data.get("response", [])
    if not response:
        return []
    return response[0].get("league", {}).get("standings", [[]])[0]


def _current_season() -> int:
    """API-Football usa el año de inicio de temporada.

    Plan free solo da acceso 2022-2024. Override por env API_FOOTBALL_SEASON.
    """
    override = os.getenv("API_FOOTBALL_SEASON")
    if override and override.isdigit():
        return int(override)

    from datetime import datetime
    today = datetime.utcnow()
    natural = today.year if today.month >= 7 else today.year - 1
    # Plan free: capar a 2024 (última disponible)
    return min(natural, 2024)


if __name__ == "__main__":
    print(f"Season actual: {_current_season()}")
    print(f"\nPróximos 3 partidos LaLiga:")
    for f in get_upcoming_fixtures(n=3):
        teams = f.get("teams", {})
        date = f.get("fixture", {}).get("date", "")
        print(f"  · {teams.get('home', {}).get('name')} vs {teams.get('away', {}).get('name')} — {date}")
