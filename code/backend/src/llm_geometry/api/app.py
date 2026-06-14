"""FastAPI application factory.

Registers the JSON API under ``/api``, a consistent error envelope for every typed
error (so the client never sees a fabricated result or a raw traceback — FR-021), and
serves the built Svelte bundle in production when it exists.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..config import REPO_ROOT
from ..errors import LLMGeometryError
from .routes import router


def create_app() -> FastAPI:
    app = FastAPI(title="llm-geometry core machinery", version="0.1.0")

    # Dev convenience: the Vite dev server proxies /api, but allow direct cross-origin
    # calls too.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(LLMGeometryError)
    async def _handle_llm_error(_request: Request, exc: LLMGeometryError) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content=exc.to_envelope())

    app.include_router(router, prefix="/api")

    # Serve the built frontend if present (production). Mounted last so /api wins.
    dist = REPO_ROOT / "code" / "frontend" / "dist"
    if dist.is_dir():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=str(dist), html=True), name="static")

    return app


app = create_app()
