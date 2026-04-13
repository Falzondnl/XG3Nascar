"""
XG3 NASCAR Microservice — Entry Point
FastAPI application with lifespan model loading.
Port: 8031

Data: NASCAR Cup Results 100K rows 1949-2025
Markets: Race Winner, Top-3, Top-5, Top-10, Head-to-Head
Model: 3-model stacking ensemble (CatBoost + LightGBM + XGBoost) + BetaCalibrator
ELO: per surface (dirt/paved/road) + overall, updated after each race
"""
from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import DEBUG, PORT, R0_DIR, SERVICE_NAME, SERVICE_VERSION
from ml.predictor import NascarPredictor
from feeds.optic_odds import OpticOddsFeed
from api.routes import health, races, admin
from api.routes.outrights import router as outrights_router
from api.routes.settlement import router as settlement_router

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: load predictor; if no models exist, train first."""
    logger.info("Starting %s v%s (port=%d)", SERVICE_NAME, SERVICE_VERSION, PORT)

    # Initialise Optic Odds feed
    app.state.optic_feed = OpticOddsFeed()
    logger.info("Optic Odds feed initialised (available=%s)", app.state.optic_feed.is_available())

    # Load predictor — auto-train if models missing
    predictor = NascarPredictor()
    ensemble_path = os.path.join(R0_DIR, "ensemble.pkl")

    if not os.path.exists(ensemble_path):
        logger.warning(
            "No trained models found at %s — running training pipeline now ...",
            R0_DIR,
        )
        try:
            from ml.trainer import train as run_training
            metrics = run_training()
            logger.info("Auto-training complete: %s", metrics)
        except Exception as exc:
            logger.error("Auto-training failed: %s — predictor will be unloaded", exc, exc_info=True)

    try:
        predictor.load(R0_DIR)
        logger.info(
            "NascarPredictor loaded: %d drivers tracked",
            predictor.driver_count,
        )
    except FileNotFoundError as exc:
        logger.warning(
            "Model files not found (%s) — predictor in unloaded state. "
            "POST /api/v1/nascar/admin/train to trigger training.",
            exc,
        )
    except Exception as exc:
        logger.error("Predictor load failed: %s", exc, exc_info=True)

    app.state.predictor = predictor

    logger.info("%s startup complete", SERVICE_NAME)
    yield

    logger.info("%s shutting down", SERVICE_NAME)


def create_app() -> FastAPI:
    app = FastAPI(
        title="XG3 NASCAR Microservice",
        description=(
            "NASCAR Cup Series race prediction and market pricing. "
            "100K+ results 1949-2025. ELO per surface, 3-model ensemble, Harville markets."
        ),
        version=SERVICE_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health routes (no prefix)
    app.include_router(health.router, tags=["health"])

    # Domain routes
    app.include_router(
        races.router,
        prefix="/api/v1/nascar/races",
        tags=["races"],
    )
    app.include_router(
        admin.router,
        prefix="/api/v1/nascar/admin",
        tags=["admin"],
    )
    app.include_router(
        outrights_router,
        prefix="/api/v1/nascar/outrights",
        tags=["outrights"],
    )
    app.include_router(settlement_router)

    @app.get("/", include_in_schema=False)
    async def root() -> JSONResponse:
        return JSONResponse({
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "docs": "/docs",
            "health": "/health",
            "api_prefix": "/api/v1/nascar",
        })

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=DEBUG,
        log_level="debug" if DEBUG else "info",
    )
