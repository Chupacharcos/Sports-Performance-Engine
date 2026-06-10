# Sports Performance Engine

Predictor de resultados (1X2) de **LaLiga** entrenado con **6.000+ partidos
reales de 16 temporadas** (football-data.co.uk, datos públicos sin API key) y
evaluado honestamente contra el techo práctico de esta tarea: las cuotas de
cierre de los bookies.

- **Demo:** https://adrianmoreno-dev.com/demo/sports-engine
- **Puerto:** 8001 — `sports-engine.service`

## Resultados honestos (split temporal, test = temporadas 24/25 y 25/26)

|                          | Accuracy | Log-loss |
|--------------------------|---------:|---------:|
| Baseline "siempre local" |   46.7%  |    —     |
| **Modelo (ensemble)**    | **52.6%**| **0.979**|
| Mercado (cuotas cierre)  |   54.9%  |   0.960  |

El mercado de cierre (Pinnacle/B365 con el margen descontado) agrega
información que las estadísticas públicas no tienen (alineaciones, lesiones,
dinero informado): es el techo práctico. El modelo recupera la mayor parte de
ese edge usando solo datos públicos, quedando a 2.2 puntos del mercado y +5.9
sobre el baseline ingenuo. No se reporta ningún número que no esté en
`models/metrics.json`.

## Pipeline

```
football-data.co.uk (16 temporadas SP1.csv)
        │  fetch_real_data.py
        ▼
Features con higiene temporal estricta (solo pasado):
  Elo partido a partido (K=20, ventaja local 60) · goles/encajados rolling 5
  tiros a puerta rolling 5 · forma (pts últimos 5) · medias por localía
        │  train_real.py  (split temporal: train ≤22/23 · val 23/24 · test 24/25+25/26)
        ▼
LightGBM + XGBoost ensemble → models/*.pkl + metrics.json (incluye baseline del mercado)
```

## Stack

| Componente | Tech |
|------------|------|
| Datos históricos | football-data.co.uk (CSV públicos, cuotas de cierre incluidas) |
| Datos en vivo | API-Football v3 (clasificación + próximos partidos) |
| Modelos | LightGBM + XGBoost (multiclase H/D/A) |
| Rating | Elo calculado partido a partido |
| API | FastAPI (puerto 8001) |

## Entrenar

```bash
cd scripts
../venv/bin/python fetch_real_data.py   # descarga 16 temporadas + features + team_summary
../venv/bin/python train_real.py        # entrena + evalúa vs mercado + guarda metrics.json
```

## Endpoints

- `GET /ml/sports/health` · `GET /ml/sports/stats` (métricas + baselines honestos)
- `GET /ml/sports/teams` (Elo + forma actuales) · `POST /ml/sports/predict`
- `GET /ml/sports/matches/recent` · `GET /ml/sports/match/{id}`
- `GET /ml/sports/live/standings` · `GET /ml/sports/live/upcoming` (API-Football)

> Predicción estadística con datos públicos. No es asesoramiento de apuestas.
