from __future__ import annotations

import json
import pickle
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if config_path.suffix == ".json":
        return json.loads(config_path.read_text(encoding="utf-8"))
    if config_path.suffix in {".yaml", ".yml"}:
        return yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raise ValueError(f"unsupported config suffix: {config_path.suffix}")


def save_config(config: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix == ".json":
        target.write_text(json.dumps(config, indent=2, default=_json_default), encoding="utf-8")
        return
    if target.suffix in {".yaml", ".yml"}:
        target.write_text(yaml.safe_dump(json.loads(json.dumps(config, default=_json_default))), encoding="utf-8")
        return
    raise ValueError(f"unsupported config suffix: {target.suffix}")


def save_results(results: Any, path: str | Path) -> None:
    save_config(results, path)


def save_model(model: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump(model, handle)


def load_model(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)
