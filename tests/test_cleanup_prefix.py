from tradutor.cleanup import cleanup_before_refine


def test_dedupe_prefix_lines_removes_truncated_and_is_idempotent():
    """Confirma que a limpeza determinística é idempotente."""
    text = (
        "A habilidade única do Theo   agora estava no nível 3, e ele\n"
        "A habilidade única do Theo agora estava no nível 3, e ele havia aprendido...\n"
        "Linha normal."
    )

    cleaned, stats = cleanup_before_refine(text)
    assert "havia aprendido" in cleaned
    assert cleaned.count("A habilidade única do Theo") == 1
    assert stats["prefix_lines_removed"] == 1

    cleaned2, stats2 = cleanup_before_refine(cleaned)
    assert cleaned2 == cleaned
    assert stats2["prefix_lines_removed"] == 0


def test_cleanup_does_not_merge_line_followed_by_quote():
    """Confirma que a limpeza determinística distingue o caso válido do artefato que deve corrigir."""
    text = "Ele olhou para o pano branco.\n“Mmm?”\n"
    cleaned, stats = cleanup_before_refine(text)
    assert "pano branco.\n“Mmm?”" in cleaned
    assert stats.get("prefix_lines_removed", 0) == 0
    assert stats.get("prefix_merges_blocked", 0) >= 1
