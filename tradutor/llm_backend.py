"""Abstrações dos backends LLM suportados pelo pipeline."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import requests

try:
    from google import genai
    from google.genai import types as genai_types
except Exception:  # pragma: no cover - lib opcional
    genai = None
    genai_types = None

from .config import BackendType


@dataclass
class LLMResponse:
    """Encapsula a resposta retornada por um provedor de LLM junto com a latência da requisição."""

    text: str
    latency: float


class LLMBackend:
    """
    Abstração unificada para interagir com os provedores de LLM configurados.
    Gerencia configurações como temperatura, modelo, timeouts e o roteamento da requisição.
    """

    def __init__(
        self,
        backend: BackendType,
        model: str,
        temperature: float,
        logger: logging.Logger,
        base_url: str = "http://localhost:11434",
        request_timeout: int = 120,
        gemini_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: str = "https://api.openai.com/v1",
        repeat_penalty: float | None = None,
        num_predict: int = 768,
        num_ctx: int | None = None,
        keep_alive: str | int | None = "30m",
        api_mode: str = "generate",
        think: bool | None = None,
    ) -> None:
        """Inicializa o estado de LLMBackend."""
        self.backend = backend
        self.model = model
        self.temperature = temperature
        self.base_url = base_url.rstrip("/")
        self.logger = logger
        self.request_timeout = request_timeout
        self.gemini_api_key = gemini_api_key
        self.openai_api_key = openai_api_key
        self.openai_base_url = openai_base_url.rstrip("/")
        self.repeat_penalty = repeat_penalty
        self.num_predict = num_predict
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.api_mode = api_mode
        self.think = think

    def generate(self, prompt: str) -> LLMResponse:
        """Envia o prompt para o backend configurado e retorna a resposta formatada."""
        start = time.perf_counter()
        if self.backend == "ollama":
            text = self._call_ollama(prompt)
        elif self.backend == "gemini":
            text = self._call_gemini(prompt)
        elif self.backend == "openai":
            text = self._call_openai(prompt)
        else:
            raise ValueError(f"Backend não suportado: {self.backend}")
        latency = time.perf_counter() - start
        return LLMResponse(text=text, latency=latency)

    def _call_ollama(self, prompt: str) -> str:
        """Executa ollama."""
        if self.api_mode == "chat":
            return self._call_ollama_chat(prompt)
        if self.api_mode != "generate":
            raise ValueError(f"Modo de API Ollama não suportado: {self.api_mode}")

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }
        if self.repeat_penalty is not None:
            payload["options"]["repeat_penalty"] = self.repeat_penalty
        if self.num_ctx is not None:
            payload["options"]["num_ctx"] = self.num_ctx
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        try:
            resp = requests.post(url, json=payload, timeout=self.request_timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            self.logger.error("Erro ao chamar Ollama: %s", exc)
            raise

        if "response" not in data:
            raise ValueError(f"Resposta inválida do Ollama: {json.dumps(data)[:200]}")
        return data["response"].strip()

    def _call_ollama_chat(self, prompt: str) -> str:
        """Executa uma solicitação pela API de chat do backend local."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }
        if self.repeat_penalty is not None:
            payload["options"]["repeat_penalty"] = self.repeat_penalty
        if self.num_ctx is not None:
            payload["options"]["num_ctx"] = self.num_ctx
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        if self.think is not None:
            payload["think"] = self.think
        try:
            resp = requests.post(url, json=payload, timeout=self.request_timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            self.logger.error("Erro ao chamar Ollama chat: %s", exc)
            raise

        message = data.get("message")
        if not isinstance(message, dict) or "content" not in message:
            raise ValueError(f"Resposta inválida do Ollama chat: {json.dumps(data)[:200]}")
        return (message.get("content") or "").strip()

    def _call_gemini(self, prompt: str) -> str:
        """Executa uma solicitação pelo backend remoto configurado."""
        if genai is None or genai_types is None:
            raise RuntimeError("google-genai não instalado.")
        api_key = self.gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY não configurada.")
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.num_predict,
                ),
            )
        except Exception as exc:
            self.logger.error("Erro ao chamar Gemini: %s", exc)
            raise
        text = (response.text or "").strip()
        if not text:
            raise ValueError("Gemini retornou resposta vazia.")
        return text

    def _call_openai(self, prompt: str) -> str:
        """Executa uma solicitação pela API Responses da OpenAI."""
        api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY não configurada.")

        payload = {
            "model": self.model,
            "input": prompt,
            "max_output_tokens": self.num_predict,
            "temperature": self.temperature,
        }
        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            response = requests.post(
                f"{self.openai_base_url}/responses",
                headers=headers,
                json=payload,
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            self.logger.error("Erro ao chamar OpenAI: %s", exc)
            raise

        text = _extract_openai_output_text(data)
        if not text:
            raise ValueError("OpenAI retornou resposta sem texto.")
        return text


def _extract_openai_output_text(data: dict) -> str:
    """Extrai texto da resposta direta ou da estrutura de itens."""
    direct_text = data.get("output_text")
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text.strip()

    parts: list[str] = []
    for item in data.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict) or content.get("type") != "output_text":
                continue
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return "\n".join(parts)
