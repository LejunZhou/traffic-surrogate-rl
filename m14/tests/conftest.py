"""Keep the standalone test suite on its own source tree."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARENT_SOURCE = ROOT.parent / "src"
PARENT_SCRIPTS = ROOT.parent / "scripts"
sys.path[:] = [
    entry for entry in sys.path
    if Path(entry or ".").resolve() not in {PARENT_SOURCE, PARENT_SCRIPTS}
]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
