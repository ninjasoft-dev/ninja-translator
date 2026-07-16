from tradutor.section_splitter import split_into_sections


def test_split_into_sections_detects_titles() -> None:
    """Confirma a detecção de problemas em conteúdo válido na divisão em seções."""
    text = "Front\n\nPrologue\nLine A\n\nChapter 1: Start\nLine B\n\nEpilogue\nLine C"
    sections = split_into_sections(text)
    assert len(sections) == 4  # inclui preâmbulo
    assert sections[0]["title"] == "Full Text"
    assert sections[1]["title"].lower().startswith("prologue")
    assert "Line A" in sections[1]["body"]
    assert sections[2]["title"].lower().startswith("chapter 1")
    assert "Line B" in sections[2]["body"]
    assert sections[3]["title"].lower().startswith("epilogue")
    assert "Line C" in sections[3]["body"]


def test_split_into_sections_when_no_markers_returns_full_text() -> None:
    """Confirma o registro correto de conteúdo válido na divisão em seções."""
    text = "No markers here.\nJust text."
    sections = split_into_sections(text)
    assert len(sections) == 1
    assert sections[0]["title"] == "Full Text"
    assert "No markers" in sections[0]["body"]


def test_split_adds_preamble_before_first_marker() -> None:
    """Valida as regras de marcadores de controle na divisão em seções."""
    text = "Intro text here.\n\nChapter 1:\nBody of chapter one."
    sections = split_into_sections(text)
    assert len(sections) == 2
    assert sections[0]["title"] == "Full Text"
    assert "Intro text" in sections[0]["body"]
    assert sections[1]["title"].lower().startswith("chapter 1")
    assert "Body of chapter one" in sections[1]["body"]


def test_split_handles_colon_without_title_and_epilogue() -> None:
    """Valida as regras de títulos estruturais na divisão em seções."""
    text = "Chapter 1:\nBody\n\nEpilogue\nThe end."
    sections = split_into_sections(text)
    assert len(sections) == 2
    assert sections[0]["title"].lower().startswith("chapter 1")
    assert sections[1]["title"].lower().startswith("epilogue")


def test_split_ignores_empty_toc_entries() -> None:
    """Valida as regras de sumários e conteúdo narrativo na divisão em seções."""
    text = "Prologue\n\nChapter 1\n\nEpilogue\nClosing text."
    sections = split_into_sections(text)
    # Prologue e Chapter 1 sem corpo são ignorados; mantém Epilogue
    assert len(sections) == 1
    assert sections[0]["title"].lower().startswith("epilogue")
    assert "Closing text." in sections[0]["body"]


def test_split_ignores_numeric_toc_entries() -> None:
    """Valida as regras de sumários e conteúdo narrativo na divisão em seções."""
    text = "Chapter 1\n1\nChapter 2\n2\nChapter 3\nBody text."
    sections = split_into_sections(text)
    assert len(sections) == 1
    assert sections[0]["title"].lower().startswith("chapter 3")
    assert "Body text." in sections[0]["body"]
