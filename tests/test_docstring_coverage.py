import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = (ROOT / "tradutor", ROOT / "scripts", ROOT / "tests")
GENERIC_DOCSTRING = "Processamento interno auxiliar."


def _iter_python_files() -> list[Path]:
    """Lista os arquivos Python que fazem parte do código versionado."""
    files = list(ROOT.glob("*.py"))
    for directory in SOURCE_DIRS:
        files.extend(directory.rglob("*.py"))
    return sorted(files)


def test_functions_and_classes_have_meaningful_docstrings() -> None:
    """Impede a inclusão de definições sem documentação ou com texto-placeholder."""
    undocumented: list[str] = []
    for path in _iter_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            ):
                continue
            docstring = ast.get_docstring(node)
            if not docstring or docstring.strip() == GENERIC_DOCSTRING:
                relative_path = path.relative_to(ROOT)
                undocumented.append(f"{relative_path}:{node.lineno} ({node.name})")

    assert not undocumented, "Docstrings ausentes ou genéricas:\n" + "\n".join(undocumented)
