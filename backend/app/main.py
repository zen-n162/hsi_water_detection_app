from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes_inference import router as inference_router
from backend.app.api.routes_preview import router as preview_router

PROJECT_ROOT = Path(__file__).resolve().parents[2]

app = FastAPI(title="HSI Water Detection Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

outputs_dir = PROJECT_ROOT / "outputs"
outputs_dir.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")

@app.get("/health")
def health():
    return {"ok": True}

app.include_router(inference_router)
app.include_router(preview_router)
