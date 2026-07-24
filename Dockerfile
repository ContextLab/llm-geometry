# Reproducible dual-stack image for the llm-geometry core machinery
# (Constitution V — the released artifacts alone reproduce the environment).
#   build:  docker build -t llm-geometry .
#   run:    docker run -it -p 8000:8000 llm-geometry      # serves API + built UI on :8000
FROM python:3.11-slim

# System deps + Node 20 (for the Svelte frontend build).
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl git build-essential ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /mnt

# --- Backend (CPU PyTorch + transformers + reduction stack) ---
# Copy manifests + sources, then install the package (editable) with test extras.
COPY code/backend/pyproject.toml /mnt/code/backend/
COPY code/backend/src /mnt/code/backend/src
RUN pip install --no-cache-dir --upgrade pip wheel \
    && pip install --no-cache-dir -e "/mnt/code/backend[test]"

# --- Frontend (Svelte + Vite) ---
COPY code/frontend/package.json /mnt/code/frontend/
RUN cd /mnt/code/frontend && npm install
COPY code/frontend /mnt/code/frontend
RUN cd /mnt/code/frontend && npm run build   # -> code/frontend/dist, served by FastAPI

EXPOSE 8000 5173

# Default: serve the JSON API and the built UI from one origin.
CMD ["uvicorn", "llm_geometry.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
