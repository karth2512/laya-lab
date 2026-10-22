"""Config loading: base.yaml + arms/<arm>/config.yaml (under `arm:`) + `--set a.b=c` overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG = ROOT / "configs" / "base.yaml"


class _Open(BaseModel):
    model_config = ConfigDict(extra="allow")


class DatasetCfg(_Open):
    hf_id: str
    version: str
    raw_dir: str
    split_manifest: str
    val_fraction: float
    split_seed: int
    max_length: int


class MlflowCfg(_Open):
    tracking_uri: str
    artifact_location: str


class EvalCfg(_Open):
    bootstrap_resamples: int
    ece_bins: int
    latency_examples: int
    save_model: bool


class BaseCfg(_Open):
    """Validates the shared part of the config; arm sections are free-form."""

    experiment: str
    dataset: DatasetCfg
    seeds: list[int]
    fractions: list[float]
    stage: Literal["dev", "final"]
    seed: int
    fraction: float
    device: str
    mlflow: MlflowCfg
    eval: EvalCfg
    arm: dict[str, Any] = {}


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def apply_override(cfg: dict[str, Any], override: str) -> None:
    """Apply `a.b=c` in place; `c` is parsed as YAML. A key not found at the top level is looked
    up under `arm:` first, so `--set lora.r=16` reaches the arm's own config."""
    key, sep, raw = override.partition("=")
    if not sep:
        raise ValueError(f"override must look like key=value, got {override!r}")
    parts = key.strip().split(".")
    if parts[0] not in cfg and parts[0] in cfg.get("arm", {}):
        parts = ["arm", *parts]
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = yaml.safe_load(raw)


def load_config(
    arm: str,
    overrides: list[str] | None = None,
    base_path: Path = BASE_CONFIG,
    use_tuned: bool = True,
) -> dict[str, Any]:
    """base.yaml + arm config.yaml + configs/tuned/<arm>.yaml (written by scripts.tune) + CLI."""
    cfg = _read_yaml(base_path)
    arm_cfg_path = ROOT / "arms" / arm / "config.yaml"
    if arm_cfg_path.exists():
        cfg = _deep_merge(cfg, {"arm": _read_yaml(arm_cfg_path)})
    tuned_path = ROOT / "configs" / "tuned" / f"{arm}.yaml"
    tuned = _read_yaml(tuned_path).get("overrides", []) if use_tuned and tuned_path.exists() else []
    for o in [*tuned, *(overrides or [])]:
        apply_override(cfg, o)
    BaseCfg.model_validate(cfg)
    return cfg


def flatten(cfg: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """{'a': {'b': 1}} -> {'a.b': 1}. Lists are kept as values."""
    out: dict[str, Any] = {}
    for k, v in cfg.items():
        name = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, f"{name}."))
        else:
            out[name] = v
    return out
