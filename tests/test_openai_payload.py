"""Testes do contrato HTTP do backend OpenAI."""

import logging

import pytest

import tradutor.llm_backend as llm_backend


class FakeResponse:
    """Resposta HTTP mínima usada para inspecionar a requisição."""

    def __init__(self, payload: dict) -> None:
        """Armazena o JSON devolvido pelo duplo de teste."""
        self.payload = payload

    def raise_for_status(self) -> None:
        """Simula uma resposta HTTP bem-sucedida."""

    def json(self) -> dict:
        """Retorna o corpo JSON configurado."""
        return self.payload


def test_openai_uses_responses_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Envia modelo, prompt e limite de saída para a Responses API."""
    captured: dict = {}

    def fake_post(url: str, **kwargs) -> FakeResponse:
        """Captura os argumentos e devolve saída estruturada da API."""
        captured.update({"url": url, **kwargs})
        return FakeResponse(
            {"output": [{"content": [{"type": "output_text", "text": "Tradução pronta."}]}]}
        )

    monkeypatch.setattr(llm_backend.requests, "post", fake_post)
    backend = llm_backend.LLMBackend(
        backend="openai",
        model="modelo-de-teste",
        temperature=0.2,
        logger=logging.getLogger("test"),
        openai_api_key="segredo-de-teste",
        num_predict=321,
    )

    response = backend.generate("Traduza este trecho.")

    assert response.text == "Tradução pronta."
    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert captured["headers"] == {"Authorization": "Bearer segredo-de-teste"}
    assert captured["json"] == {
        "model": "modelo-de-teste",
        "input": "Traduza este trecho.",
        "max_output_tokens": 321,
        "temperature": 0.2,
    }


def test_openai_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Falha antes da chamada HTTP quando a variável de ambiente não existe."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backend = llm_backend.LLMBackend(
        backend="openai",
        model="modelo-de-teste",
        temperature=0.2,
        logger=logging.getLogger("test"),
    )

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        backend.generate("teste")
