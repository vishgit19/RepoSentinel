"""Run the RepoSentinel API and UI.

    python scripts/serve.py [--host 127.0.0.1] [--port 8000] [--reload]

Serves the JSON API, the SSE stream and the static frontend from one process,
so a demo needs nothing but this command.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve RepoSentinel.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    arguments = parser.parse_args()

    import uvicorn
    from reposentinel.config import get_settings

    settings = get_settings()
    if not any(settings.provider_availability().values()):
        print(
            "WARNING: no model credentials found. The UI will load, but a run "
            "cannot start until OPENAI_API_KEY is set.",
            file=sys.stderr,
        )

    print(f"RepoSentinel -> http://{arguments.host}:{arguments.port}")
    uvicorn.run(
        "reposentinel.api.app:app",
        host=arguments.host,
        port=arguments.port,
        reload=arguments.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
