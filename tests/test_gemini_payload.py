"""Testes do contrato com o SDK Google Gen AI."""

import logging
from types import SimpleNamespace

import pytest

import tradutor.llm_backend as llm_backend


def test_gemini_uses_current_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cria o cliente atual e envia limites de geração no objeto de configuração."""
    captured: dict = {}

    class FakeGenerateContentConfig:
        """Registra as opções fornecidas ao SDK."""

        def __init__(self, **kwargs) -> None:
            """Guarda as opções para as asserções do teste."""
            captured["config"] = kwargs

    class FakeModels:
        """Simula o recurso de modelos do cliente."""

        def generate_content(self, **kwargs):
            """Captura a chamada e devolve texto traduzido."""
            captured["generate"] = kwargs
            return SimpleNamespace(text="Tradução pelo Gemini.")

    class FakeClient:
        """Simula o cliente Google Gen AI."""

        def __init__(self, *, api_key: str) -> None:
            """Registra a chave sem realizar conexão externa."""
            captured["api_key"] = api_key
            self.models = FakeModels()

    monkeypatch.setattr(llm_backend, "genai", SimpleNamespace(Client=FakeClient))
    monkeypatch.setattr(
        llm_backend,
        "genai_types",
        SimpleNamespace(GenerateContentConfig=FakeGenerateContentConfig),
    )
    backend = llm_backend.LLMBackend(
        backend="gemini",
        model="modelo-gemini-de-teste",
        temperature=0.3,
        logger=logging.getLogger("test"),
        gemini_api_key="chave-de-teste",
        num_predict=456,
    )

    response = backend.generate("Traduza.")

    assert response.text == "Tradução pelo Gemini."
    assert captured["api_key"] == "chave-de-teste"
    assert captured["config"] == {"temperature": 0.3, "max_output_tokens": 456}
    assert captured["generate"]["model"] == "modelo-gemini-de-teste"
    assert captured["generate"]["contents"] == "Traduza."
