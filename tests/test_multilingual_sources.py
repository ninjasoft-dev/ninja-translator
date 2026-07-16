"""Testes do catálogo, detecção e guardrails de idiomas de origem."""

import pytest

from tradutor.config import AppConfig
from tradutor.glossary_utils import select_terms_for_chunk
from tradutor.language_guardrails import (
    detect_residual_source_language,
    residual_issue_type,
)
from tradutor.languages import detect_source_language, normalize_source_language
from tradutor.main import build_parser
from tradutor.quality_checks import run_translation_quality_checks
from tradutor.section_splitter import split_into_sections
from tradutor.translate import enforce_canonical_terms, source_heading_to_pt


@pytest.mark.parametrize(
    ("value", "expected"),
    [("English", "en"), ("japonês", "ja"), ("KR", "ko"), ("zh-TW", "zh")],
)
def test_normalize_source_language_aliases(value: str, expected: str) -> None:
    """Aceita nomes legíveis e aliases comuns de configuração."""
    assert normalize_source_language(value) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("彼女は静かに窓を開けた。", "ja"),
        ("그녀는 조용히 창문을 열었다.", "ko"),
        ("她轻轻地打开了窗户。", "zh"),
        ("Ella abrió la ventana, pero no estaba sola.", "es"),
    ],
)
def test_detect_source_language(text: str, expected: str) -> None:
    """Reconhece alfabetos distintos e um idioma latino frequente."""
    assert detect_source_language(text) == expected


def test_configured_language_has_priority_over_detection() -> None:
    """Respeita a opção explícita quando um trecho curto é ambíguo."""
    assert detect_source_language("第1章", "ja") == "ja"


@pytest.mark.parametrize(
    ("language", "text"),
    [("ja", "Ela respondeu: まだ終わっていない。"), ("ko", "Ela disse 아직 끝나지 않았다.")],
)
def test_detect_residual_source_script(language: str, text: str) -> None:
    """Sinaliza escrita da origem que permaneceu na saída em português."""
    detected, reason = detect_residual_source_language(text, language)

    assert detected
    assert reason.startswith(f"residual_source_language:{language}:")
    assert residual_issue_type(language) == "residual_source_language"


def test_english_issue_type_remains_compatible() -> None:
    """Preserva o identificador histórico usado por relatórios em inglês."""
    assert residual_issue_type("en") == "residual_english"


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        ("第3章 静かな朝", "# Capítulo 3: 静かな朝"),
        ("제2장 새로운 길", "# Capítulo 2: 새로운 길"),
        ("序章", "# Prólogo"),
        ("後記", "# Pós-escrito"),
    ],
)
def test_source_heading_to_pt_multilingual(heading: str, expected: str) -> None:
    """Converte marcadores estruturais frequentes sem depender da LLM."""
    assert source_heading_to_pt(heading) == expected


def test_split_japanese_sections_keeps_unspaced_body() -> None:
    """Não confunde um parágrafo japonês sem espaços com entrada curta de sumário."""
    text = (
        "第1章 始まり\n\n彼女は静かに窓を開け、外の雨を見つめていた。\n\n"
        "第2章 旅立ち\n\n朝になると二人は町を出た。"
    )

    sections = split_into_sections(text)

    assert [section["title"] for section in sections] == ["第1章 始まり", "第2章 旅立ち"]


def test_quality_checks_report_japanese_residual() -> None:
    """Inclui o idioma resolvido e o resíduo asiático no QA final."""
    report = run_translation_quality_checks(
        "彼女は静かに窓を開けた。",
        "Ela abriu 静かな窓 e esperou.",
        [],
    )

    assert report["source_language"] == "ja"
    assert report["issues_by_type"]["residual_source_language"] == 1


def test_cli_accepts_openai_and_japanese_source() -> None:
    """Expõe as novas opções no parser público da aplicação."""
    args = build_parser(AppConfig()).parse_args(
        [
            "traduz-md",
            "--input",
            "volume.md",
            "--backend",
            "openai",
            "--source-language",
            "ja",
        ]
    )

    assert args.backend == "openai"
    assert args.source_language == "ja"


def test_glossary_matches_cjk_term_adjacent_to_text() -> None:
    """Localiza termos CJK mesmo sem espaços ou limites de palavra ocidentais."""
    terms = [{"key": "第1章", "pt": "Capítulo 1", "enforce": True}]

    selected, matched_count = select_terms_for_chunk(
        terms,
        "物語は第1章から始まる。",
        fallback_limit=0,
    )
    replaced, replacements = enforce_canonical_terms("A referência 第1章から continua.", terms)

    assert selected == terms
    assert matched_count == 1
    assert replaced == "A referência Capítulo 1から continua."
    assert replacements == {"第1章": 1}
