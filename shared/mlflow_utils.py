"""The only module that imports mlflow (enforced by tests/test_isolation.py)."""

from __future__ import annotations

import contextlib
import importlib.metadata
import math
import os
import platform
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from mlflow.entities import Metric, Param, RunStatus, RunTag
from mlflow.tracking import MlflowClient

LIBRARIES = ["torch", "transformers", "peft", "laya", "mlflow"]
_BATCH = 100  # MLflow's per-request limit for params


def _check_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"metric {name!r} is not finite: {value}")
    return value


class ChildRun:
    """Logging handle for one child run; these methods back RunContext's log_* callables."""

    def __init__(self, client: MlflowClient, run_id: str) -> None:
        self.client = client
        self.run_id = run_id
        self.logged: set[str] = set()

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        self.client.log_metric(self.run_id, name, _check_finite(name, value), step=step or 0)
        self.logged.add(name)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        now = int(time.time() * 1000)
        items = [Metric(k, _check_finite(k, v), now, step or 0) for k, v in metrics.items()]
        for i in range(0, len(items), _BATCH):
            self.client.log_batch(self.run_id, metrics=items[i : i + _BATCH])
        self.logged.update(metrics)

    def log_param(self, key: str, value: Any) -> None:
        self.client.log_param(self.run_id, key, str(value)[:6000])

    def log_params(self, params: dict[str, Any]) -> None:
        items = [Param(k, str(v)[:6000]) for k, v in params.items()]
        for i in range(0, len(items), _BATCH):
            self.client.log_batch(self.run_id, params=items[i : i + _BATCH])

    def set_tag(self, key: str, value: Any) -> None:
        self.client.set_tag(self.run_id, key, str(value))

    def set_tags(self, tags: dict[str, Any]) -> None:
        items = [RunTag(k, str(v)) for k, v in tags.items()]
        self.client.log_batch(self.run_id, tags=items)

    def log_artifact(self, path: str | Path) -> None:
        self.client.log_artifact(self.run_id, str(path))

    def log_dict(self, data: dict[str, Any], artifact_file: str) -> None:
        self.client.log_dict(self.run_id, data, artifact_file)


class Tracker:
    """Parent run per arm, nested child run per (seed, fraction, variant, stage)."""

    def __init__(self, tracking_uri: str, experiment: str, artifact_location: str) -> None:
        self.client = MlflowClient(tracking_uri=tracking_uri)
        exp = self.client.get_experiment_by_name(experiment)
        if exp is None:
            location = Path(artifact_location).resolve().as_uri()
            self.experiment_id = self.client.create_experiment(experiment, location)
        else:
            self.experiment_id = exp.experiment_id

    def _runs(self, filter_string: str) -> list[Any]:
        return list(
            self.client.search_runs([self.experiment_id], filter_string, max_results=50000)
        )

    def parent_run_id(self, arm: str) -> str:
        runs = self._runs(f"tags.arm = '{arm}' and tags.run_kind = 'parent'")
        if runs:
            return str(runs[0].info.run_id)
        run = self.client.create_run(
            self.experiment_id, run_name=arm, tags={"arm": arm, "run_kind": "parent"}
        )
        self.client.set_terminated(run.info.run_id)
        return str(run.info.run_id)

    def completed_run_names(self, arm: str) -> set[str]:
        runs = self._runs(f"tags.arm = '{arm}' and tags.run_kind = 'child'")
        return {
            r.info.run_name for r in runs if r.info.status == RunStatus.to_string(RunStatus.FINISHED)
        }

    @contextlib.contextmanager
    def child(self, arm: str, run_name: str) -> Iterator[ChildRun]:
        parent = self.parent_run_id(arm)
        run = self.client.create_run(
            self.experiment_id,
            run_name=run_name,
            tags={"mlflow.parentRunId": parent, "arm": arm, "run_kind": "child"},
        )
        child = ChildRun(self.client, run.info.run_id)
        try:
            yield child
        except BaseException:
            self.client.set_terminated(child.run_id, "FAILED")
            raise
        self.client.set_terminated(child.run_id, "FINISHED")


def _cmd(*args: str) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _device_name() -> str:
    return (
        os.environ.get("LAYA_BENCH_DEVICE_NAME")
        or _cmd("sysctl", "-n", "machdep.cpu.brand_string")
        or platform.processor()
        or platform.machine()
    )


def _on_ac_power() -> str:
    out = _cmd("pmset", "-g", "batt")
    if not out:
        return "unknown"
    return str("AC Power" in out).lower()


def _library_versions() -> str:
    versions = []
    for lib in LIBRARIES:
        try:
            versions.append(f"{lib}=={importlib.metadata.version(lib)}")
        except importlib.metadata.PackageNotFoundError:
            versions.append(f"{lib}==none")
    return ",".join(versions)


def env_tags(device: str) -> dict[str, str]:
    """Environment tags from REQUIREMENTS §5 (arm, stage, data_hash are added by the runner)."""
    return {
        "git_sha": _cmd("git", "rev-parse", "HEAD") or "unknown",
        "git_dirty": str(bool(_cmd("git", "status", "--porcelain"))).lower(),
        "hostname": socket.gethostname(),
        "device_name": _device_name(),
        "macos_version": platform.mac_ver()[0] or "n/a",
        "torch_backend": device,
        "on_ac_power": _on_ac_power(),
        "python_version": sys.version.split()[0],
        "library_versions": _library_versions(),
    }
