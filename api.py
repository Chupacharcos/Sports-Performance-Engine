"""Sports Performance Engine — FastAPI App"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.sports import router

app = FastAPI(title="Sports Performance Engine", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router)

@app.get("/health")
def health():
    return {"status": "ok", "service": "sports-engine", "port": 8001}
