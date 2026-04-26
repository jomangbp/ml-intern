"""FastAPI application for HF Agent web interface."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from routes.agent import router as agent_router
from routes.auth import router as auth_router
from telegram_bot import telegram_bot_service

# Load .env from project root (parent directory)
load_dotenv(Path(__file__).parent.parent / ".env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting HF Agent backend...")
    await telegram_bot_service.start()
    try:
        yield
    finally:
        await telegram_bot_service.stop()
        logger.info("Shutting down HF Agent backend...")


app = FastAPI(
    title="HF Agent",
    description="ML Engineering Assistant API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(agent_router)
app.include_router(auth_router)


@app.get("/api")
async def api_root():
    """API root endpoint."""
    return {
        "name": "HF Agent API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/api/gateway/health")
async def gateway_health_endpoint():
    """Gateway health endpoint — works with or without Telegram."""
    from gateway.health import gateway_health
    from gateway.identity import identity_manager
    from events.event_store import event_store
    from prompt_cron import prompt_cron_manager
    from telegram_bot import telegram_bot_service
    from session_manager import session_manager

    active_sessions = sum(1 for s in session_manager.sessions.values() if s.is_active)
    crons = await prompt_cron_manager.list()
    active_crons = sum(1 for c in crons if c.get("status") in ("scheduled", "running"))

    return gateway_health(
        telegram_running=telegram_bot_service.running,
        telegram_enabled=telegram_bot_service.enabled,
        active_sessions=active_sessions,
        active_crons=active_crons,
        running_jobs=0,
        event_stats=event_store.stats(),
    )


# Serve static files (frontend build) in production
# MUST be after all API routes — mount("/") catches everything
static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/", StaticFiles(directory=str(static_path), html=True), name="static")
    logger.info(f"Serving static files from {static_path}")
else:
    logger.info("No static directory found, running in API-only mode")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
