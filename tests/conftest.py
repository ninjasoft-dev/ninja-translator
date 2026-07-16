import sys
from pathlib import Path

# Garante que pytest encontre o pacote `tradutor` mesmo fora de venv/editável.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
