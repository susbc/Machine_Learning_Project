"""Code resources and writable artifacts have separate roots.

ML_ARTIFACT_ROOT isolates training outputs without relocating static resources.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path(os.getenv("ML_ARTIFACT_ROOT", PROJECT_ROOT)).expanduser().resolve()
SAVED_DIR = ARTIFACT_ROOT / "saved"
OUTPUT_DIR = ARTIFACT_ROOT / "output"
LOGS_DIR = ARTIFACT_ROOT / "logs"
