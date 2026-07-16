from types import SimpleNamespace

from tradutor.config import AppConfig
from tradutor.main import (
    _refine_review_is_acceptable,
    _should_run_refine_after_translate,
)


def test_refine_is_opt_in_after_translation() -> None:
    """Valida as regras de conteúdo válido no refino."""
    args = SimpleNamespace(no_refine=False, refine=False)
    cfg = AppConfig(refine_after_translate=False)

    assert not _should_run_refine_after_translate(args, cfg)
    assert _should_run_refine_after_translate(SimpleNamespace(no_refine=False, refine=True), cfg)
    assert not _should_run_refine_after_translate(SimpleNamespace(no_refine=True, refine=True), cfg)


def test_refine_quality_gate_rejects_lower_score() -> None:
    """Confirma a detecção de problemas em conteúdo válido no refino."""
    base = {"quality": {"score": 100}}
    worse = {"quality": {"score": 94}}
    equal = {"quality": {"score": 100}}

    assert not _refine_review_is_acceptable(base, worse)
    assert _refine_review_is_acceptable(base, equal)
