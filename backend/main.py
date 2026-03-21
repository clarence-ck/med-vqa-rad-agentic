from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import BACKEND_CORS_ORIGINS
from routes.analyze import router as analyze_router
from routes.dataset import router as dataset_router

app = FastAPI(
    title="MedVisionQuery API",
    description="Medical VQA Explanation Agent — 5-agent pipeline for medical image question answering",
    version="1.0.0",
)

# CORS — allow frontend dev server and configured deployed frontend origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router)
app.include_router(dataset_router)


@app.get("/")
async def root():
    return {
        "name": "MedVisionQuery API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "analyze": "POST /api/analyze",
            "baseline": "POST /api/analyze/baseline",
            "workflow": "POST /api/analyze/workflow",
            "samples": "GET /api/dataset/samples",
            "sample": "GET /api/dataset/sample/{id}",
            "sample_analysis": "GET /api/dataset/sample/{id}/analysis",
            "evaluation": "GET /api/dataset/evaluation",
        }
    }
