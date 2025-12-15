# scripts/feature_extraction/SWIN/backbones/import_utils.py
from __future__ import annotations

from pathlib import Path
import importlib.util
from types import ModuleType


def get_main_project_root() -> Path:
    """
    Assumes this file is located at:
        main_project/scripts/feature_extraction/SWIN/backbones/import_utils.py
    """
    return Path(__file__).resolve().parents[4]


def import_module_from_file(module_name: str, file_path: str | Path) -> ModuleType:
    """
    Import a module from an explicit .py file path WITHOUT touching sys.path.

    NOTE:
    - This executes the target file.
    - It does NOT automatically make sibling imports inside that file work
      unless that repo is self-contained using relative imports.
    """
    file_path = Path(file_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Missing module file: {file_path}")

    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for: {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
