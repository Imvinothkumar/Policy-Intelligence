"""Small Ollama client used by the local RAG synthesis mode."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator, List, Optional

import requests

from src.config import DEFAULT_OLLAMA_MODEL, OLLAMA_BASE_URL, OLLAMA_MODEL_FALLBACKS


class OllamaUnavailableError(RuntimeError):
    """Raised when Ollama is not running or no model can be selected."""


@dataclass
class OllamaModelSelection:
    selected_model: Optional[str]
    available_models: List[str]
    ollama_available: bool
    message: str


def list_ollama_models(base_url: str = OLLAMA_BASE_URL, timeout: float = 2.0) -> List[str]:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        response.raise_for_status()
    except requests.RequestException:
        return []

    data = response.json()
    models = []
    for item in data.get("models", []):
        name = item.get("name") or item.get("model")
        if name:
            models.append(name)
    return sorted(models)


def select_ollama_model(
    preferred_model: Optional[str] = None,
    base_url: str = OLLAMA_BASE_URL,
) -> OllamaModelSelection:
    available = list_ollama_models(base_url=base_url)
    if not available:
        return OllamaModelSelection(
            selected_model=None,
            available_models=[],
            ollama_available=False,
            message="Ollama is not reachable or has no local models installed.",
        )

    ordered_preferences = []
    if preferred_model:
        ordered_preferences.append(preferred_model)
    ordered_preferences.extend(model for model in OLLAMA_MODEL_FALLBACKS if model not in ordered_preferences)
    if DEFAULT_OLLAMA_MODEL not in ordered_preferences:
        ordered_preferences.insert(0, DEFAULT_OLLAMA_MODEL)

    for model in ordered_preferences:
        if model in available:
            return OllamaModelSelection(
                selected_model=model,
                available_models=available,
                ollama_available=True,
                message=f"Selected Ollama model: {model}",
            )

    return OllamaModelSelection(
        selected_model=available[0],
        available_models=available,
        ollama_available=True,
        message=f"No preferred fallback model found. Selected first available model: {available[0]}",
    )


def generate_with_ollama(
    prompt: str,
    model: Optional[str] = None,
    base_url: str = OLLAMA_BASE_URL,
    timeout: float = 120.0,
) -> str:
    selection = select_ollama_model(preferred_model=model, base_url=base_url)
    if not selection.selected_model:
        raise OllamaUnavailableError(selection.message)

    payload = {
        "model": selection.selected_model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "1h",
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_ctx": 2048,
            "num_predict": 350,
        },
    }
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/api/generate",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise OllamaUnavailableError(f"Ollama generation failed: {exc}") from exc

    data = response.json()
    return str(data.get("response", "")).strip()


def stream_with_ollama(
    prompt: str,
    model: Optional[str] = None,
    base_url: str = OLLAMA_BASE_URL,
    timeout: float = 120.0,
) -> Iterator[str]:
    """Stream response tokens in real-time as they are produced."""
    selection = select_ollama_model(preferred_model=model, base_url=base_url)
    if not selection.selected_model:
        raise OllamaUnavailableError(selection.message)

    payload = {
        "model": selection.selected_model,
        "prompt": prompt,
        "stream": True,
        "keep_alive": "1h",
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_ctx": 2048,
            "num_predict": 350,
        },
    }
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/api/generate",
            json=payload,
            timeout=timeout,
            stream=True,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise OllamaUnavailableError(f"Ollama generation failed: {exc}") from exc

    for line in response.iter_lines(decode_unicode=True):
        if line:
            try:
                chunk_obj = json.loads(line)
                part = chunk_obj.get("response", "")
                if part:
                    yield part
                if chunk_obj.get("done", False):
                    break
            except Exception:
                continue
