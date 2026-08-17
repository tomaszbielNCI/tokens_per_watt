# tpw/ollama.py
import json
import time

import requests

BASE = "http://localhost:11434"
_URL = f"{BASE}/api/generate"


def generate(
    model: str,
    prompt: str,
    num_predict: int = 1024,
    num_ctx: int = 4096,
    temperature: float = 0.0,
    seed: int = 0,
    keep_alive: int | str = -1,
    host: str = _URL,
    timeout: int = 600,
) -> dict:
    """Single non-streaming completion. Returns Ollama's raw payload plus wall time."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {
            "num_predict": num_predict,
            "num_ctx": num_ctx,
            "temperature": temperature,
            "seed": seed,
        },
    }
    start = time.monotonic()
    response = requests.post(host, json=payload, timeout=timeout)
    if not response.ok:
        detail = response.json().get("error", response.text)
        raise RuntimeError(f"ollama {response.status_code}: {detail}")
    data = response.json()
    data["wall_s"] = time.monotonic() - start
    return data


def list_models(base: str = BASE) -> list[str]:
    """Model tags currently present on the server."""
    response = requests.get(f"{base}/api/tags", timeout=10)
    response.raise_for_status()
    return [m["name"] for m in response.json().get("models", [])]


def digests(base: str = BASE) -> dict[str, str]:
    """Tag -> digest, recorded so that a tag reassigned upstream cannot
    silently change what was measured."""
    response = requests.get(f"{base}/api/tags", timeout=10)
    response.raise_for_status()
    return {m["name"]: m.get("digest", "") for m in response.json().get("models", [])}


def pull(model: str, base: str = BASE, timeout: int = 3600) -> None:
    """Blocking pull with progress echoed to stdout."""
    with requests.post(
        f"{base}/api/pull", json={"model": model}, stream=True, timeout=timeout
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            status = json.loads(line).get("status", "")
            print(f"  {model}: {status}", end="\r")
    print(f"  {model}: done          ")


def ensure_models(models: list[str], base: str = BASE) -> dict[str, str]:
    """Pull anything missing, then return tag -> digest for the run log."""
    present = set(list_models(base))
    for model in models:
        if model not in present:
            print(f"pulling {model}")
            pull(model, base)
    return digests(base)
