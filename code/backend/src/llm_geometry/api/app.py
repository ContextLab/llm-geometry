"""FastAPI application factory.

Registers the JSON API under ``/api``, a consistent error envelope for every typed
error (so the client never sees a fabricated result or a raw traceback — FR-021), and
serves the built Svelte bundle in production when it exists.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..config import REPO_ROOT
from ..errors import LLMGeometryError
from .routes import router
from .routes_arch import router as arch_router
from .routes_geo import router as geo_router
from .routes_lex import router as lex_router


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

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(_request: Request, exc: RequestValidationError) -> JSONResponse:
        # Malformed/missing typed params must use the SAME envelope as every other
        # failure (contract: one error shape) — not Starlette's {"detail": [...]}.
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(p) for p in first.get("loc", []) if p != "body")
        msg = f"{loc}: {first.get('msg', 'invalid request')}" if loc else "invalid request"
        return JSONResponse(
            status_code=400,
            content={"error": {"type": "InvalidParamError", "message": msg, "detail": {}}},
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(_request: Request, exc: Exception) -> JSONResponse:
        # Any non-typed failure still returns the contract error envelope with the real
        # message (never a fabricated result, never a bare traceback) — FR-021.
        return JSONResponse(
            status_code=500,
            content={"error": {"type": "InternalError", "message": str(exc), "detail": {}}},
        )

    app.include_router(router, prefix="/api")
    app.include_router(geo_router, prefix="/api")
    app.include_router(arch_router, prefix="/api")
    # Feature 006 (additive; the frozen 002 contract is untouched — see
    # specs/006-lexicon-lab-tiny/contracts/api-lex.md).
    app.include_router(lex_router, prefix="/api")

    # Serve the built frontend if present (production). Mounted last so /api wins.
    dist = REPO_ROOT / "code" / "frontend" / "dist"
    if dist.is_dir():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=str(dist), html=True), name="static")

    return app


app = create_app()
