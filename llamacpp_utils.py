"""Small, non-invasive health check for a local llama.cpp server.

Expects `llama-server` (the OpenAI-compatible server that ships with
llama.cpp) running locally, e.g.:

    llama-server -hf Qwen/Qwen3-4B-GGUF:Q4_0 --port 8080

which exposes an OpenAI-compatible API at http://localhost:8080/v1.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


def llamacpp_status(base_url: str = "http://localhost:8080/v1", timeout: float = 2.0) -> dict[str, Any]:
    """Ping the llama.cpp server's /v1/models endpoint.

    Returns a dict shaped like the one ollama_utils.ollama_status() returns
    so the rest of the app (status panel, error messages) can treat every
    local provider the same way.
    """
    status: dict[str, Any] = {"running": False, "models": []}
    url = f"{base_url.rstrip('/')}/models"
    try:
        request = Request(url, method="GET")
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        status["running"] = True
        status["models"] = [item.get("id", "") for item in payload.get("data", [])]
    except (OSError, URLError, ValueError):
        pass
    return status
