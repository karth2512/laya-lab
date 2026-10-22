"""Disk cache keyed by content hash (used by API arms). One JSON object per line, append-only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ResponseCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self._data: dict[str, Any] = {}
        if self.path.exists():
            with self.path.open() as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        self._data[row["key"]] = row["value"]

    @staticmethod
    def key(*parts: str) -> str:
        return hashlib.sha256("|".join(parts).encode()).hexdigest()

    def get(self, key: str) -> Any | None:
        value = self._data.get(key)
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
        return value

    def put(self, key: str, value: Any) -> None:
        self._data[key] = value
        with self.path.open("a") as f:
            f.write(json.dumps({"key": key, "value": value}) + "\n")

    def __len__(self) -> int:
        return len(self._data)
