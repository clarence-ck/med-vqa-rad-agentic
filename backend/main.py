import os
import time
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from config import BACKEND_CORS_ORIGINS
from routes.analyze import router as analyze_router
from routes.dataset import router as dataset_router
from logger import get_logger

_log = get_logger("server")

app = FastAPI(
    title="MedVisionQuery API",
    description="MedVisionQuery — agentic controller loop for medical image question answering",
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


# Paths that are too noisy to log (browser auto-requests, OpenAPI docs)
_SKIP_LOG_PATHS = {"/favicon.ico", "/docs", "/redoc", "/openapi.json"}


@app.middleware("http")
async def _log_requests(request: Request, call_next):
    """Log every HTTP request with method, path, status code, and duration."""
    path = request.url.path
    skip = path in _SKIP_LOG_PATHS
    started = time.perf_counter()
    if not skip:
        _log.info("→ %s %s", request.method, path)
    response = await call_next(request)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    if not skip:
        _log.info(
            "← %s %s  status=%d  duration=%dms",
            request.method,
            path,
            response.status_code,
            elapsed_ms,
        )
    return response


@app.get("/favicon.ico", include_in_schema=False)
@app.get("/favicon.png", include_in_schema=False)
async def favicon():
    """Serve the MEDVQA AGENTIC logo as the favicon."""
    favicon_path = os.path.join(os.path.dirname(__file__), "static", "favicon_transparent.png")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path, media_type="image/png")
    return Response(status_code=204)


@app.on_event("startup")
async def _on_startup():
    _log.info("=" * 60)
    _log.info("MedVisionQuery API  v1.0.0  starting up")
    _log.info("CORS origins: %s", BACKEND_CORS_ORIGINS)
    _log.info("=" * 60)


@app.on_event("shutdown")
async def _on_shutdown():
    _log.info("MedVisionQuery API shutting down")


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
            "evaluation": "GET /api/dataset/evaluation?run=<run_id>",
        }
    }
