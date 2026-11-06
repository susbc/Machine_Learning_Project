"""Training defaults must be independent of the shell's working directory."""

import os
from pathlib import Path
import subprocess
import sys


def test_legacy_and_api_outputs_share_isolated_artifact_root(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    artifact_root = tmp_path / "artifacts"
    working_dir = tmp_path / "working"
    working_dir.mkdir()
    environment = dict(os.environ, ML_ARTIFACT_ROOT=str(artifact_root), PYTHONPATH=str(repository))
    result = subprocess.run(
        [sys.executable, "-c", """
from pathlib import Path
from core.paths import ARTIFACT_ROOT, PROJECT_ROOT, SAVED_DIR, OUTPUT_DIR
from model_code.config.config import MODEL_DIR, CONFIG_DIR
from model_code.main_model import BuildingModel
from backend.api import training, automl, models

engine = BuildingModel('unused.csv', 'unused.xlsx', '202611')
assert engine.model_dir == MODEL_DIR == ARTIFACT_ROOT / 'model_code' / 'saved'
assert engine.output_dir == OUTPUT_DIR == ARTIFACT_ROOT / 'output'
assert CONFIG_DIR == PROJECT_ROOT / 'model_code' / 'config'
assert training.ARTIFACT_DIR == SAVED_DIR / 'manual_training'
assert automl.ARTIFACT_DIR == SAVED_DIR / 'automl'
assert models.PROJECT_ROOT == ARTIFACT_ROOT
assert not (Path.cwd() / 'saved').exists()
assert not (Path.cwd() / 'output').exists()
"""],
        cwd=working_dir, env=environment, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
