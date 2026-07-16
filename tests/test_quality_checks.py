from tradutor.quality_checks import format_quality_cell, run_translation_quality_checks


def test_quality_checks_flag_glossary_leak_and_missing_canonical() -> None:
    """Confirma a detecção de problemas em termos de glossário nas verificações de qualidade."""
    source = "The Four Tower Wardens entered with the Ash Wolves."
    translated = "Os Guardiões das Quatro Torres entraram com as Ash Wolves."
    terms = [
        {
            "key": "Four Tower Wardens",
            "pt": "Quatro Santos",
            "aliases": ["Guardiões das Quatro Torres"],
            "enforce": True,
        },
        {
            "key": "Ash Wolves",
            "pt": "Tigres Dente-de-Sabre",
            "aliases": ["Ash Wolves"],
            "enforce": True,
        },
    ]

    report = run_translation_quality_checks(source, translated, terms)

    issue_types = {issue["type"] for issue in report["issues"]}
    assert "source_term_in_target" in issue_types
    assert "missing_canonical_term" in issue_types
    assert report["score"] < 100


def test_quality_checks_flag_bad_name_alias() -> None:
    """Confirma a detecção de problemas em termos de glossário nas verificações de qualidade."""
    source = "Suou Kayako bowed her head."
    translated = "Kayado baixou a cabeça."
    terms = [
        {
            "key": "Suou Kayako",
            "pt": "Suou Kayako",
            "category": "personagem",
            "gender": "feminino",
            "bad_aliases": ["Kayado"],
        }
    ]

    report = run_translation_quality_checks(source, translated, terms)

    assert report["issues_by_type"]["bad_alias_in_target"] == 1


def test_quality_checks_handles_case_only_bad_alias_without_flagging_canonical() -> None:
    """Valida as regras de termos de glossário nas verificações de qualidade."""
    terms = [{"key": "Fio Limite", "pt": "Fio Limite", "bad_aliases": ["fio limite"]}]

    canonical = run_translation_quality_checks(
        "Fio Limite", "A técnica Fio Limite funciona.", terms
    )
    lowercase = run_translation_quality_checks(
        "Fio Limite", "A técnica fio limite funciona.", terms
    )

    assert canonical["issues_by_type"].get("bad_alias_in_target") is None
    assert lowercase["issues_by_type"]["bad_alias_in_target"] == 1


def test_quality_checks_flag_possible_gender_mismatch() -> None:
    """Confirma a detecção de problemas em consistência de gênero nas verificações de qualidade."""
    source = "Rhea Angun was taller than her brother."
    translated = "Rhea Angun era mais alto que seu irmão."
    terms = [
        {
            "key": "Rhea Angun",
            "pt": "Rhea Angun",
            "category": "personagem",
            "gender": "feminino",
        }
    ]

    report = run_translation_quality_checks(source, translated, terms)

    assert report["issues_by_type"]["possible_gender_mismatch"] == 1


def test_quality_checks_does_not_bind_object_name_to_speaker_adjective() -> None:
    """Confirma a preservação de consistência de gênero nas verificações de qualidade."""
    terms = [
        {
            "key": "Theo Ardent",
            "pt": "Theo Ardent",
            "category": "personagem",
            "gender": "masculino",
        }
    ]

    report = run_translation_quality_checks(
        "Lina could not have defeated Theo alone.",
        "Eu não teria derrotado o Theo Ardent sozinha.",
        terms,
    )

    assert report["issues_by_type"].get("possible_gender_mismatch") is None


def test_quality_checks_flag_mixed_gender_adjectives_without_name() -> None:
    """Confirma a detecção de problemas em consistência de gênero nas verificações de qualidade."""
    source = "Be careful about what comes next."
    translated = "É bom que você esteja sendo cuidadosa, atento ao que vem a seguir."

    report = run_translation_quality_checks(source, translated, [])

    assert report["issues_by_type"]["possible_gender_mismatch"] == 1


def test_quality_checks_does_not_flag_masculine_noun_near_feminine_name() -> None:
    """Não confunde um substantivo masculino com a flexão de uma personagem."""
    source = "Suou Kayako looked embarrassed and guilty."
    translated = "Suou Kayako estava parada, com um olhar envergonhado e culpado."
    terms = [
        {
            "key": "Suou Kayako",
            "pt": "Suou Kayako",
            "category": "personagem",
            "gender": "feminino",
        }
    ]

    report = run_translation_quality_checks(source, translated, terms)

    assert report["issues_by_type"].get("possible_gender_mismatch") is None


def test_quality_checks_does_not_flag_alias_inside_canonical_translation() -> None:
    """Não sinaliza um alias quando ele integra a própria forma canônica."""
    source = "Goddess Aurelia smiled."
    translated = "A Deusa Aurelia sorriu."
    terms = [
        {
            "key": "Goddess Aurelia",
            "pt": "Deusa Aurelia",
            "aliases": ["Aurelia"],
            "enforce": True,
        }
    ]

    report = run_translation_quality_checks(source, translated, terms)

    assert report["issues_by_type"].get("source_term_in_target") is None


def test_quality_checks_allows_explicit_target_alias() -> None:
    """Confirma a preservação de termos de glossário nas verificações de qualidade."""
    source = "The Wildly Beautiful Emperor arrived."
    translated = "Zine chegou ao acampamento."
    terms = [
        {
            "key": "Wildly Beautiful Emperor",
            "pt": "Imperador Selvagemente Belo",
            "source_aliases": ["Beautiful Wild Emperor"],
            "allowed_target_aliases": ["Zine"],
        }
    ]

    report = run_translation_quality_checks(source, translated, terms)

    assert report["issues_by_type"].get("source_term_in_target") is None
    assert report["issues_by_type"].get("missing_canonical_term") is None


def test_quality_checks_allows_contextual_noun_form_for_skill() -> None:
    """Confirma a preservação de conteúdo válido nas verificações de qualidade."""
    report = run_translation_quality_checks(
        "Cael used Paralyze.",
        "O alcance é igual ao da Paralisia.",
        [
            {
                "key": "Paralyze",
                "pt": "Paralisar",
                "source_case_sensitive": True,
                "allowed_target_aliases": ["Paralisia"],
            }
        ],
    )

    assert report["issues_by_type"].get("missing_canonical_term") is None


def test_quality_checks_flags_bad_alias_separately_from_source_alias() -> None:
    """Confirma a detecção de problemas em termos de glossário nas verificações de qualidade."""
    source = "The Order of Aurelia were summoned."
    translated = "Os Discípulos de Aurelia foram convocados."
    terms = [
        {
            "key": "Order of Aurelia",
            "pt": "Ordem de Aurelia",
            "source_aliases": ["Aurelia's Disciples"],
            "bad_aliases": ["Discípulos de Aurelia"],
        }
    ]

    report = run_translation_quality_checks(source, translated, terms)

    assert report["issues_by_type"]["bad_alias_in_target"] == 1


def test_quality_checks_flags_contextual_target_replacement_alias() -> None:
    """Confirma a detecção de problemas em termos de glossário nas verificações de qualidade."""
    report = run_translation_quality_checks(
        "After the Last Trial",
        "Após a Batalha de Morte.",
        [
            {
                "key": "Last Trial",
                "pt": "Confronto Final",
                "target_replacements": {"Após a Batalha de Morte": "Após o Confronto Final"},
            }
        ],
    )

    assert report["issues_by_type"]["bad_alias_in_target"] == 1


def test_quality_checks_respects_case_sensitive_source_term() -> None:
    """Valida as regras de termos de glossário nas verificações de qualidade."""
    terms = [{"key": "Freeze", "pt": "Congelar", "source_case_sensitive": True}]

    common_word_report = run_translation_quality_checks(
        "After she watched his body freeze, she lost consciousness.",
        "Depois que ela viu o corpo dele ficar imóvel, perdeu a consciência.",
        terms,
    )
    skill_report = run_translation_quality_checks(
        "Cael cast Freeze on the undead enemy.",
        "Cael lançou a habilidade no inimigo morto-vivo.",
        terms,
    )

    assert common_word_report["issues_by_type"].get("missing_canonical_term") is None
    assert skill_report["issues_by_type"]["missing_canonical_term"] == 1


def test_quality_checks_does_not_require_full_canonical_form_for_source_alias() -> None:
    """Confirma a preservação de termos de glossário nas verificações de qualidade."""
    report = run_translation_quality_checks(
        "The refugees are headed to Yonato.",
        "Os refugiados estão indo para Yonato.",
        [
            {
                "key": "State of Yonato",
                "pt": "Estado de Yonato",
                "source_aliases": ["Yonato"],
            }
        ],
    )

    assert report["issues_by_type"].get("missing_canonical_term") is None


def test_quality_checks_requires_canonical_form_for_enforced_source_alias() -> None:
    """Confirma a detecção de problemas em termos de glossário nas verificações de qualidade."""
    report = run_translation_quality_checks(
        "The Nimbo Clan arrived.",
        "O grupo da Munin chegou.",
        [
            {
                "key": "Forbidden Words Clan",
                "pt": "Clã das Palavras Proibidas",
                "source_aliases": ["Nimbo Clan"],
                "enforce": True,
            }
        ],
    )

    assert report["issues_by_type"]["missing_canonical_term"] == 1


def test_quality_checks_does_not_flag_fixed_expression_de_surpresa() -> None:
    """Não trata uma expressão fixa como divergência de gênero."""
    source = "Theo was caught off guard."
    translated = "Mesmo tendo sido pego de surpresa, Theo permaneceu quieto."
    terms = [
        {
            "key": "Theo Ardent",
            "pt": "Theo Ardent",
            "aliases": ["Theo"],
            "category": "personagem",
            "gender": "masculino",
        }
    ]

    report = run_translation_quality_checks(source, translated, terms)

    assert report["issues_by_type"].get("possible_gender_mismatch") is None


def test_quality_checks_clean_text_scores_high() -> None:
    """Valida as regras de conteúdo válido nas verificações de qualidade."""
    source = "The Dragonslayer spoke."
    translated = "O Matador de Dragões falou."
    terms = [{"key": "Dragonslayer", "pt": "Matador de Dragões", "enforce": True}]

    report = run_translation_quality_checks(source, translated, terms)

    assert report["score"] == 100
    assert format_quality_cell(report) == "100/100"


def test_quality_checks_flag_refine_marker() -> None:
    """Confirma a detecção de problemas em marcadores de controle nas verificações de qualidade."""
    report = run_translation_quality_checks(
        "Source",
        "### TEXTO_REFINADO_INICIO\nTexto.\n### TEXTO_REFINADO_FIM",
        [],
    )

    assert report["issues_by_type"]["residual_translation_marker"] == 1


def test_quality_checks_flag_common_english_interjection() -> None:
    """Confirma a detecção de problemas em idioma residual nas verificações de qualidade."""
    report = run_translation_quality_checks(
        "Phew, I am thirsty.", "Phew, fiquei com sede. Huh?", []
    )

    assert report["issues_by_type"]["residual_english"] == 2


def test_quality_checks_flags_known_game_jargon_left_in_english() -> None:
    """Confirma a detecção de problemas em idioma residual nas verificações de qualidade."""
    report = run_translation_quality_checks("The buffs faded.", "Os buffs desapareceram.", [])

    assert report["issues_by_type"]["residual_english"] == 1


def test_quality_checks_flag_malformed_quote_boundary() -> None:
    """Confirma a detecção de problemas em aspas e estrutura de diálogos nas verificações de qualidade."""
    report = run_translation_quality_checks("Source", "”“Ah, tudo bem.”", [])

    assert report["issues_by_type"]["malformed_quote_boundary"] == 1


def test_quality_checks_allows_dialogues_separated_by_newlines() -> None:
    """Confirma a preservação de conteúdo válido nas verificações de qualidade."""
    report = run_translation_quality_checks("Source", "“Oi.”\n\n“Tudo bem.”", [])

    assert report["issues_by_type"].get("malformed_quote_boundary") is None


def test_quality_checks_flags_missing_quote_spacing() -> None:
    """Confirma a detecção de problemas em aspas e estrutura de diálogos nas verificações de qualidade."""
    report = run_translation_quality_checks("Source", '"Uma fala.""Outra fala."', [])

    assert report["issues_by_type"]["missing_quote_spacing"] == 1


def test_quality_checks_flags_stray_format_marker_after_dialogue() -> None:
    """Confirma a detecção de problemas em aspas e estrutura de diálogos nas verificações de qualidade."""
    report = run_translation_quality_checks("Source", "“Tudo bem.”*", [])

    assert report["issues_by_type"]["stray_format_marker"] == 1
