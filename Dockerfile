# syntax=docker/dockerfile:1
# Demo image: API + UI + local sandbox. Docker-in-Docker is not used; the
# sandbox inside this container is the local subprocess backend.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt pyproject.toml ./
COPY backend ./backend
COPY frontend ./frontend
COPY benchmarks ./benchmarks
COPY scripts ./scripts

RUN pip install --no-cache-dir -e . \
    && useradd --create-home --uid 1000 reposentinel \
    && mkdir -p /app/data \
    && chown -R reposentinel:reposentinel /app

USER reposentinel
ENV REPOSENTINEL_SANDBOX_BACKEND=local \
    REPOSENTINEL_VECTOR_STORE=sqlite \
    PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["python", "scripts/serve.py", "--host", "0.0.0.0", "--port", "8000"]
