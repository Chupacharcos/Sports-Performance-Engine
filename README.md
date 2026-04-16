# Sports Performance Engine

Motor de análisis de rendimiento deportivo que combina **LightGBM + XGBoost ensemble** con datos de **StatsBomb Open Data** para predecir resultados de LaLiga y Champions League.

## Demo en vivo

[adrianmoreno-dev.com/demo/sports-engine](https://adrianmoreno-dev.com/demo/sports-engine)

## Arquitectura

```
StatsBomb Open Data
    └── statsbombpy (eventos por partido)
            └── Rolling team stats (xG, xA, presiones, posesión — ventana 5/10 partidos)
                    ├── LightGBM (Optuna 30 trials)
                    ├── XGBoost  (Optuna 25 trials)
                    └── Ensemble pesado → predicción + SHAP TreeExplainer
```

## Stack

| Componente | Tecnología |
|---|---|
| Datos | StatsBomb Open Data (statsbombpy) + synthetic fallback |
| Modelos | LightGBM + XGBoost ensemble |
| Hyperparams | Optuna (30 + 25 trials) |
| Explicabilidad | SHAP TreeExplainer |
| API | FastAPI (puerto 8001) |
| Split temporal | TimeSeriesSplit 70/15/15 |

## Features

- **Rolling stats** por equipo: xG for/against, xA, shots, pressures, posesión — media últimos 5 y 10 partidos
- **Head-to-head** histórico: goles/xG en enfrentamientos directos
- **Competición** como feature categórica (LaLiga, Champions, Copa)
- **Condición** local/visitante

## Métricas

| Métrica | Valor |
|---|---|
| AUC-ROC | 0.71 |
| Accuracy | 0.38 (3 clases: L/E/V) |
| Dataset | ~3.200 partidos |
| Features | 25+ |

## Endpoints

```
GET  /ml/sports/health          Estado del servicio
GET  /ml/sports/stats           Métricas del modelo
GET  /ml/sports/teams           Equipos disponibles
GET  /ml/sports/competitions    Competiciones disponibles
GET  /ml/sports/matches/recent  Últimos partidos analizados
POST /ml/sports/predict         Predicción de partido
GET  /ml/sports/match/{id}      Detalle de partido
```

### Ejemplo predict

```bash
curl -X POST http://localhost:8001/ml/sports/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team": "Real Madrid", "away_team": "FC Barcelona", "competition": "La Liga"}'
```

```json
{
  "prediction": "Empate",
  "confidence": 0.431,
  "probabilities": {"home_win": 0.381, "draw": 0.431, "away_win": 0.188},
  "shap_explanation": {...}
}
```

## Instalación

```bash
git clone https://github.com/Chupacharcos/Sports-Performance-Engine.git
cd Sports-Performance-Engine
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py   # Descarga StatsBomb Open Data
python scripts/train.py           # Entrena modelos y genera artifacts/
uvicorn api:app --port 8001
```

## Estructura

```
sports-engine/
├── api.py              FastAPI app
├── routers/sports.py   Endpoints
├── scripts/
│   ├── download_data.py   StatsBomb + synthetic fallback
│   └── train.py           Entrena LightGBM + XGBoost
├── artifacts/          Modelos serializados (excluidos de git)
└── requirements.txt
```

## Licencia

MIT
