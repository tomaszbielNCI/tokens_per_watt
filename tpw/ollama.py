# tpw/ollama.py
import time

import requests

_URL = "http://localhost:11434/api/generate"


def generate(
    model: str,
    prompt: str,
    num_predict: int = 1024,
    temperature: float = 0.0,
    seed: int = 0,
    host: str = _URL,
    timeout: int = 300,
) -> dict:
    """Single non-streaming completion. Returns Ollama's raw payload plus wall time."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": num_predict,
            "temperature": temperature,
            "seed": seed,
        },
    }
    start = time.monotonic()
    response = requests.post(host, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    data["wall_s"] = time.monotonic() - start
    return data