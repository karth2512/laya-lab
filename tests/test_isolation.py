"""Isolation (§4.1, §10.4): arms never import each other; only mlflow_utils imports mlflow."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = "." * node.level + (node.module or "")
            names.add(base)
            names.update(f"{base}.{a.name}" for a in node.names)
    return names


def _py_files() -> list[Path]:
    dirs = ["shared", "arms", "scripts", "tests", "spikes"]
    return [p for d in dirs for p in (ROOT / d).rglob("*.py")]


def test_arms_do_not_import_other_arms() -> None:
    for arm_dir in sorted(p for p in (ROOT / "arms").iterdir() if p.is_dir()):
        if arm_dir.name.startswith("__"):
            continue
        for path in arm_dir.rglob("*.py"):
            for name in _imports(path):
                if name.startswith("arms."):
                    assert name.split(".")[1] == arm_dir.name, f"{path} imports {name}"
                assert not name.startswith(".."), f"{path} uses a parent-relative import {name}"


def test_only_mlflow_utils_imports_mlflow() -> None:
    allowed = {ROOT / "shared" / "mlflow_utils.py"}
    for path in _py_files():
        if path in allowed or path.parent.name == "tests":
            continue  # tests may read MLflow back to check what was logged
        for name in _imports(path):
            assert name.split(".")[0] != "mlflow", f"{path} imports {name}"
