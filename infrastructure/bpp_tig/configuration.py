"""Portable TIG paths and declared C45 optimizer/sample-budget contracts.

Only explicit environment variables supply dataset/runtime locations. This
module performs no model construction, dataset reads, CUDA calls or training.
The C45 sampler balances task-level groups with replacement; the reference
budget matches approximately ten times the frame-anchor count, not ten epochs.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[2]
RECIPE = "bpp-tig-split-clip-frozen-prompt-v2"
TIG_REFERENCE_REVISION = "3193957e3b0187d086e44d79e0e4af9845fbd4fd"
REQUIRED_ENVIRONMENT = ("TIG_SOURCE_ROOT", "IG_HUMAN_ROOT", "IG_SIM_ROOT")
DATA_PATHS = ("human_root", "sim_root", "human_config", "sim_config", "mapping_file")
_ENVIRONMENT_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z_0-9]*)\}")


def load_training_config(path: Path, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Resolve required variables and repository-relative model configuration.

    Environment values must be nonempty absolute directory paths. The config
    may subsequently override task/episode filters only under development mode.
    Existence of data directories is checked by the dataset, not by this reader.
    """
    environment = os.environ if environment is None else environment
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Training config must be a mapping: {path}")
    required = tuple(config.get("required_environment", REQUIRED_ENVIRONMENT))
    missing = [name for name in required if not environment.get(name, "").strip()]
    if missing:
        raise ValueError("Set required environment variable(s) before running: " + ", ".join(missing))
    for name in required:
        if not Path(environment[name]).expanduser().is_absolute():
            raise ValueError(f"{name} must be an absolute path, got {environment[name]!r}")

    def resolve(value: Any) -> Any:
        if isinstance(value, str):
            def replace(match: re.Match[str]) -> str:
                name = match.group(1)
                if name not in required:
                    raise ValueError(f"Undeclared config environment variable: {name}")
                return environment[name]
            return _ENVIRONMENT_PATTERN.sub(replace, value)
        if isinstance(value, dict):
            return {key: resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item) for item in value]
        return value

    config = resolve(config)
    for key in DATA_PATHS:
        config["data"][key] = str(Path(config["data"][key]).expanduser().resolve())
    config["tig_source_root"] = str(Path(config["tig_source_root"]).expanduser().resolve())
    model_path = Path(config["model_config_file"]).expanduser()
    config["model_config_file"] = str((ROOT / model_path).resolve() if not model_path.is_absolute()
                                      else model_path.resolve())
    return config


def configure_tig_imports(config: Mapping[str, Any]) -> Path:
    """Import runtime helpers only from the explicitly selected TIG checkout."""
    root = Path(config["tig_source_root"]).resolve()
    expected = root / "examples/baselines/lerobot_dataset/normalizer.py"
    if not expected.is_file():
        raise FileNotFoundError(f"TIG_SOURCE_ROOT is not a TIG source checkout; missing {expected}")
    validate_tig_reference(config)
    for name, module in tuple(sys.modules.items()):
        if name == "examples" or name.startswith("examples."):
            filename = getattr(module, "__file__", None)
            if filename and not Path(filename).resolve().is_relative_to(root):
                raise RuntimeError(f"Already imported {name} from a different source: {filename}")
            for entry in getattr(module, "__path__", ()):
                if not Path(entry).resolve().is_relative_to(root):
                    raise RuntimeError(f"Already imported namespace {name} from a different source: {entry}")
    sys.path.insert(0, str(root))
    return root


def validate_tig_reference(config: Mapping[str, Any]) -> None:
    """Pin clean official sources for formal runs; label other sources development."""
    if config["training"]["development"]:
        return
    reference = config.get("tig_reference_revision")
    if reference != TIG_REFERENCE_REVISION:
        raise ValueError(f"Published profile requires TIG reference {TIG_REFERENCE_REVISION}")
    root = Path(config["tig_source_root"]).resolve()
    arguments = ["git", "-C", str(root)]
    commit = subprocess.check_output(arguments + ["rev-parse", "HEAD"], text=True).strip()
    if commit != reference:
        raise ValueError(f"TIG_SOURCE_ROOT HEAD={commit} differs from pinned {reference}; "
                         "select the pinned clean checkout or use --development")
    status = subprocess.check_output(arguments + ["status", "--porcelain", "--untracked-files=all"],
                                     text=True).strip()
    if status:
        raise ValueError("Formal training requires a clean pinned TIG checkout; "
                         "modified/untracked files found. Use --development for modified sources.")


def validate_training_contract(config: Mapping[str, Any], world_size: int) -> int:
    """Validate batch arithmetic before any CUDA initialization; return accumulation."""
    train = config["training"]
    if world_size <= 0 or train["microbatch"] <= 0 or train["global_batch"] <= 0:
        raise ValueError("world_size, microbatch and global_batch must be positive")
    if train["checkpoint_every"] <= 0 or train["log_every"] <= 0 or train["probe_every"] < 0:
        raise ValueError("checkpoint/log intervals must be positive; probe interval must be nonnegative")
    if train["num_workers"] < 0:
        raise ValueError("num_workers must be nonnegative")
    denominator = train["microbatch"] * world_size
    if train["global_batch"] % denominator:
        raise ValueError(f"global_batch={train['global_batch']} must be divisible by "
                         f"microbatch={train['microbatch']} * world_size={world_size}; "
                         "choose the matching profile or explicitly set batch sizes")
    accumulation = train["global_batch"] // denominator
    if not 0 < train["max_updates"] <= train["scheduler_updates"]:
        raise ValueError("Require 0 < max_updates <= scheduler_updates")
    if not 0 <= train["warmup_updates"] < train["scheduler_updates"]:
        raise ValueError("Require 0 <= warmup_updates < scheduler_updates")
    restricted = bool(config["data"].get("task_filter") or config["data"].get("episodes")
                      or train.get("fixed_batch"))
    if restricted and not train["development"]:
        raise ValueError("Restricted data/fixed batch must be labelled --development")
    if not train["development"]:
        if train["max_updates"] != train["scheduler_updates"]:
            raise ValueError("A shortened run requires --development; formal max_updates equals its fixed budget")
        if config.get("recipe") != RECIPE:
            raise ValueError(f"Published C45 training requires recipe={RECIPE}")
        budget = config["budget"]
        draws = train["scheduler_updates"] * train["global_batch"]
        if draws != budget["window_draws"]:
            raise ValueError("Batch/scheduler changes alter the declared window budget; "
                             "use --development or supply a distinct, explicit budget config")
        if budget["reference_frame_anchors"] != 3_039_478 or budget["reference_passes"] != 10:
            raise ValueError("Published C45 budget requires 3,039,478 reference anchors and 10 passes")
        exact = budget["reference_frame_anchors"] * budget["reference_passes"]
        if not exact <= draws < exact + train["global_batch"]:
            raise ValueError("Budget must round the reference exposure upward to a whole update")
        warmup = (train["scheduler_updates"] * 5 + 50) // 100
        if train["warmup_updates"] != warmup:
            raise ValueError("Published profile uses 5% warmup rounded half up to the nearest update")
    return accumulation


def validate_c45_manifest(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    """Reject incomplete source data from being reported as the full C45 recipe."""
    if config["training"]["development"]:
        return
    expected = {"robot_repositories": 180, "robot_episodes": 9000,
                "human_repositories": 45, "human_episodes": 2250,
                "frame_anchors": 3_039_478}
    mismatches = {key: {"expected": value, "actual": manifest.get(key)}
                  for key, value in expected.items() if manifest.get(key) != value}
    if mismatches:
        raise ValueError(f"Full C45 data contract failed: {mismatches}. "
                         "Use --development for a labelled subset or define a separate recipe.")


def validate_model_recipe(config: Mapping[str, Any], model: Mapping[str, Any]) -> None:
    """Prevent launching the new profile against the former shared-CLIP model."""
    if config["training"]["development"]:
        return
    vision = model["vision"]
    valid = (model.get("schema_version") == 2 and model.get("recipe") == RECIPE
             and vision.get("separate_prompt_and_receding_obs_encoders") is True
             and vision.get("prompt_frozen") is True and vision.get("receding_frozen") is False)
    if not valid:
        raise ValueError("C45 v2 requires the split model with frozen prompt CLIP and trainable robot CLIP")


def self_test() -> None:
    """CPU-only path, missing-variable, budget and manifest boundary checks."""
    from unittest.mock import patch
    profiles = [("c45.yaml", 4, 128, 237460, 11873),
                ("c45_h100x8.yaml", 8, 256, 118730, 5937)]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        environment = {name: str(root / name) for name in REQUIRED_ENVIRONMENT}
        for name, world, batch, updates, warmup in profiles:
            path = ROOT / "infrastructure/bpp_tig/configs" / name
            config = load_training_config(path, environment)
            assert config["data"]["human_root"] == environment["IG_HUMAN_ROOT"]
            assert config["data"]["sim_root"] == environment["IG_SIM_ROOT"]
            assert config["tig_source_root"] == environment["TIG_SOURCE_ROOT"]
            assert Path(config["model_config_file"]).is_relative_to(ROOT)
            train = config["training"]
            assert (train["global_batch"], train["max_updates"], train["scheduler_updates"],
                    train["warmup_updates"]) == (batch, updates, updates, warmup)
            assert validate_training_contract(config, world) == 1
            with patch("subprocess.check_output", side_effect=[TIG_REFERENCE_REVISION + "\n", ""]):
                validate_tig_reference(config)
            with patch("subprocess.check_output", return_value="wrong-commit\n"):
                try:
                    validate_tig_reference(config)
                except ValueError as error:
                    assert "differs from pinned" in str(error)
                else:
                    raise AssertionError("Wrong TIG runtime accepted as formal")
            with patch("subprocess.check_output", side_effect=[TIG_REFERENCE_REVISION + "\n", " M source.py"]):
                try:
                    validate_tig_reference(config)
                except ValueError as error:
                    assert "clean pinned" in str(error)
                else:
                    raise AssertionError("Modified TIG runtime accepted as formal")
            assert train["scheduler_updates"] * batch == 30_394_880
            try:
                validate_training_contract(config, world * 3)
            except ValueError as error:
                assert "divisible" in str(error)
            else:
                raise AssertionError("Invalid distributed batch arithmetic accepted")
            try:
                load_training_config(path, {"TIG_SOURCE_ROOT": environment["TIG_SOURCE_ROOT"]})
            except ValueError as error:
                assert "IG_HUMAN_ROOT" in str(error) and "IG_SIM_ROOT" in str(error)
            else:
                raise AssertionError("Missing variables accepted")
            try:
                validate_c45_manifest(config, {"frame_anchors": 1})
            except ValueError as error:
                assert "Full C45" in str(error)
            else:
                raise AssertionError("Incomplete data accepted as C45")
            validate_c45_manifest(config, {"robot_repositories": 180, "robot_episodes": 9000,
                                          "human_repositories": 45, "human_episodes": 2250,
                                          "frame_anchors": 3_039_478})
            development = copy.deepcopy(config)
            development["training"]["development"] = True
            development["training"]["max_updates"] = 2
            development["data"]["episodes"] = [0, 1]
            validate_c45_manifest(development, {"frame_anchors": 1})
            assert validate_training_contract(development, world) == 1
            with patch("subprocess.check_output", side_effect=AssertionError("No git required for development")):
                validate_tig_reference(development)
            model = yaml.safe_load(Path(config["model_config_file"]).read_text(encoding="utf-8"))
            validate_model_recipe(config, model)
            old_model = copy.deepcopy(model)
            old_model["schema_version"] = 1
            try:
                validate_model_recipe(config, old_model)
            except ValueError:
                pass
            else:
                raise AssertionError("Old shared model accepted for v2 profile")
    print("PASS: portable explicit paths, both sample budgets, batch arithmetic, full C45 and model recipe")


if __name__ == "__main__":
    self_test()
