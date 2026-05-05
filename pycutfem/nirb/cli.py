from __future__ import annotations

import argparse

from pycutfem.mor.io import load_config

from .offline import OfflineConfig, run_offline_pipeline
from .online import OnlineConfig, run_online_pipeline
from .validation import ValidationConfig, validate_rom


def main_train(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a non-intrusive reduced-basis model.")
    parser.add_argument("--config", required=True, help="Path to a JSON or YAML training config.")
    args = parser.parse_args(argv)
    run_offline_pipeline(OfflineConfig.from_mapping(load_config(args.config)))
    return 0


def main_run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an online NIRB evaluation.")
    parser.add_argument("--config", required=True, help="Path to a JSON or YAML online config.")
    args = parser.parse_args(argv)
    run_online_pipeline(OnlineConfig.from_mapping(load_config(args.config)))
    return 0


def main_validate(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate ROM predictions against reference data.")
    parser.add_argument("--config", required=True, help="Path to a JSON or YAML validation config.")
    args = parser.parse_args(argv)
    validate_rom(ValidationConfig.from_mapping(load_config(args.config)))
    return 0
