"""Evaluate human-video BPP through TIG's native ACT evaluation loop.

Physical observations are qpos18 and RGB224. The dataset applies its TIG
normalizer once; BPP receives two consecutive control observations and emits
normalized joint16 chunks. The dataset inverts action normalization once.
The native loop executes four actions before a new H16 prediction. Official
episode metrics are captured unchanged before reset; no task truth enters BPP.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
import traceback
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_TIG_COMMIT = "3193957e3b0187d086e44d79e0e4af9845fbd4fd"
SUBSR_REFERENCE = ("https://github.com/imitator-game/The-Imitator-Game/blob/"
    + OFFICIAL_TIG_COMMIT + "/examples/baselines/astra/docs/METRICS.md")
METRIC_FIELDS = ("success_at_end", "success_once", "last_reward", "return")


def runtime_identity(runtime: Path) -> dict[str, Any]:
    """Require the declared official source, including unmodified tracked files."""
    revision = subprocess.run(["git", "-C", str(runtime), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True).stdout.strip()
    if revision != OFFICIAL_TIG_COMMIT:
        raise ValueError(f"Expected official TIG {OFFICIAL_TIG_COMMIT}; found {revision}")
    if subprocess.run(["git", "-C", str(runtime), "diff", "--quiet", "HEAD", "--"],
                      check=False).returncode:
        raise ValueError("TIG has tracked changes; declare and audit a separate protocol")
    files = ["examples/baselines/act/eval_act_imitator.py",
             "examples/baselines/lerobot_dataset/normalizer.py",
             "examples/baselines/lerobot_dataset/video_utils.py",
             "examples/baselines/lerobot_dataset/lerobot_human_dataset.py"]
    return {"runtime_commit": revision,
            "runtime_files_sha256": {name: sha256_file(runtime / name) for name in files}}


def evaluation_training_config(payload: dict[str, Any], supplied: str | None = None
                               ) -> dict[str, Any]:
    """Resolve deployment paths explicitly; never reuse another host's paths."""
    from infrastructure.bpp_tig.configuration import load_training_config, configure_tig_imports
    if supplied:
        config = load_training_config(Path(supplied))
    else:
        config = dict(payload["config"])
        missing = [key for key in ("TIG_SOURCE_ROOT", "IG_HUMAN_ROOT", "IG_SIM_ROOT")
                   if not os.environ.get(key)]
        if missing:
            raise ValueError("Set deployment environment variables: " + ", ".join(missing))
        runtime = Path(os.environ["TIG_SOURCE_ROOT"]).resolve()
        config["tig_source_root"] = str(runtime)
        config["data"] = {**config["data"],
            "human_root": os.environ["IG_HUMAN_ROOT"], "sim_root": os.environ["IG_SIM_ROOT"],
            "human_config": str(runtime / "examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_45.json"),
            "sim_config": str(runtime / "examples/baselines/lerobot_dataset/config/exp_configs/sim_train_config_45.json"),
            "mapping_file": str(runtime / "examples/baselines/lerobot_dataset/task_mapping.json")}
    configure_tig_imports(config)
    return config


def plain(value: Any) -> Any:
    """Convert native metric tensors to JSON without changing their values."""
    if isinstance(value, torch.Tensor):
        return plain(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(value: Any) -> float:
    result = np.asarray(plain(value))
    if result.size != 1 or not np.isfinite(result).all():
        raise ValueError("Expected one finite official episode metric")
    return float(result.reshape(-1)[0])


def resolved_environment_id(env_id: str) -> str:
    """Use the official independent L3 registration, including its suffix."""
    from examples.baselines.act.eval_act_imitator import extract_base_env_name
    return extract_base_env_name(env_id)


def policy_noise_seed(env_id: str, scene_seed: int, episode_id: int) -> int:
    """Bind diffusion noise to one episode independently of scheduling/resume."""
    key = f"bpp-native-eval-v1/{env_id}/{scene_seed}/{episode_id}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")


class HistoryProcessor:
    """Native-loop processor; update O2 history on every control step."""

    def __init__(self, dataset: Any, env_id: str,
                 prompt_identity: dict[str, Any] | None = None, *,
                 official_eval: bool = False, prompt_draw_seed: int = 0) -> None:
        self.dataset, self.env_id = dataset, env_id
        self.prompt_identity = prompt_identity
        self.official_eval, self.prompt_draw_seed = official_eval, prompt_draw_seed
        self.states: deque[torch.Tensor] = deque(maxlen=2)
        self.images: deque[torch.Tensor] = deque(maxlen=2)
        self.human_metadata: dict[str, Any] = {}

    def clear_history(self) -> None:
        self.states.clear()
        self.images.clear()

    def get_video(self, env_id: str, num_envs: int) -> torch.Tensor:
        if env_id != self.env_id or num_envs != 1:
            raise ValueError("This evaluator uses one explicitly seeded environment")
        video, metadata = self.dataset.get_human_prompt(
            env_id, episode_index=None if self.official_eval else 0,
            draw_seed=self.prompt_draw_seed, official_eval=self.official_eval)
        result = torch.as_tensor(video, dtype=torch.float32)
        if result.shape != (10, 3, 224, 224) or not torch.isfinite(result).all():
            raise ValueError(f"Expected human RGB prompt [10,3,224,224], got {tuple(result.shape)}")
        if result.min() < 0 or result.max() > 1:
            raise ValueError("Human prompt must use RGB float [0,1]")
        self.human_metadata = plain(metadata)
        if self.prompt_identity is not None:
            actual = {"metadata": self.human_metadata,
                      "rgb_sha256": hashlib.sha256(result.contiguous().numpy().tobytes()).hexdigest()}
            if actual != self.prompt_identity:
                raise ValueError("Human prompt differs from the frozen run identity")
        mask = torch.as_tensor(metadata.get("mask", [False] * 10), dtype=torch.bool)
        if mask.shape != (10,) or bool(mask.all()):
            raise ValueError("Malformed human prompt padding mask")
        # Let the native BPP agent recreate padding and its validity mask.
        result = result[~mask]
        return result.unsqueeze(0)

    def normalize_state_rgb(self, state: torch.Tensor, rgb: torch.Tensor,
                            env_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        if env_id != self.env_id:
            raise ValueError("Normalizer task identity changed")
        qpos = state
        if qpos.shape[-1] > 27:
            qpos = torch.cat([qpos[..., 0:9], qpos[..., 18:27]], dim=-1)
        if qpos.shape != (1, 1, 18) or rgb.shape != (1, 1, 224, 224, 3):
            raise ValueError(f"Expected q18/RGB224 B1,T1; got {qpos.shape}/{rgb.shape}")
        normalized = self.dataset.normalize_state(qpos.detach().cpu().numpy(), env_id)
        normalized = torch.as_tensor(normalized, dtype=torch.float32, device=state.device)
        image = rgb.to(torch.float32) / 255 if rgb.dtype == torch.uint8 else rgb.to(torch.float32)
        if (not torch.isfinite(normalized).all() or not torch.isfinite(image).all()
                or image.min() < 0 or image.max() > 1):
            raise ValueError("Nonfinite normalized observation or RGB outside [0,1]")
        self.states.append(normalized[:, -1].clone())
        self.images.append(image[:, -1].clone())
        return normalized, image

    def history(self) -> dict[str, torch.Tensor]:
        if not self.states:
            raise RuntimeError("Observe the current control step before querying BPP")
        states, images = list(self.states), list(self.images)
        if len(states) == 1:
            states = [states[0], states[0]]
            images = [images[0], images[0]]
        return {"state": torch.stack(states, dim=1), "rgb": torch.stack(images, dim=1)}

    def denormalize_action(self, action: torch.Tensor, env_id: str) -> torch.Tensor:
        if env_id != self.env_id or action.shape != (1, 16):
            raise ValueError("Expected B1 normalized action16 for the active task")
        result = self.dataset.denormalize_action(action.detach().cpu().numpy(), env_id)
        physical = torch.as_tensor(result, dtype=torch.float32, device=action.device)
        if physical.shape != (1, 16) or not torch.isfinite(physical).all():
            raise ValueError("Malformed physical joint16 target")
        return physical


class NativeLoopAgent:
    """Adapt native hooks while keeping control-step history outside queries."""

    def __init__(self, agent: Any, processor: HistoryProcessor) -> None:
        self.agent, self.processor = agent, processor

    def eval(self) -> "NativeLoopAgent":
        self.agent.eval()
        return self

    def train(self, mode: bool = True) -> "NativeLoopAgent":
        self.agent.train(mode)
        return self

    def clear_cache(self) -> None:
        self.agent.clear_cache()
        self.processor.clear_history()

    def prepare_for_eval(self, **kwargs: Any) -> None:
        self.agent.prepare_for_eval(**kwargs)

    def get_action(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        del obs
        result = self.agent.get_action(self.processor.history())
        if result.shape != (1, 16, 16) or not torch.isfinite(result).all():
            raise ValueError(f"BPP must return normalized [1,16,16], got {result.shape}")
        return result


class SeededRecorder:
    """Delegate native execution; save original metrics at each final transition."""

    def __init__(self, env: Any, episode_ids: list[int], seeds: list[int], folder: Path,
                 binding: dict[str, Any]) -> None:
        if not episode_ids or len(episode_ids) != len(seeds):
            raise ValueError("An explicit nonempty episode/seed schedule is required")
        self.env, self.episode_ids, self.seeds = env, episode_ids, seeds
        self.folder, self.binding = folder, binding
        self.reset_index = -1
        self.control_steps = 0
        self.completed: set[int] = set()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self, **kwargs: Any) -> Any:
        self.reset_index += 1
        self.control_steps = 0
        # The official loop performs one unused reset after the final episode.
        index = min(self.reset_index, len(self.seeds) - 1)
        kwargs["seed"] = self.seeds[index]
        result = self.env.reset(**kwargs)
        seed = policy_noise_seed(self.binding["env_id"], self.seeds[index], self.episode_ids[index])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return result

    def step(self, action: Any) -> Any:
        result = self.env.step(action)
        self.control_steps += 1
        if bool(np.asarray(plain(result[3])).any()):
            if not 0 <= self.reset_index < len(self.seeds) or self.reset_index in self.completed:
                raise RuntimeError("Unexpected duplicate/unscheduled native final transition")
            info = result[4]["final_info"]
            if not isinstance(info, dict):
                if len(info) != 1:
                    raise ValueError("Expected one finalized environment")
                info = info[0]
            metrics = plain(info["episode"])
            if (self.binding.get("expected_control_steps") is not None
                    and self.control_steps != self.binding["expected_control_steps"]):
                raise RuntimeError("Native development episode did not execute its fixed 500 control steps")
            if any(key not in metrics for key in METRIC_FIELDS):
                raise ValueError("Native final-info is missing SR, reward or return metrics")
            if any(scalar(metrics[key]) not in (0., 1.) for key in ("success_at_end", "success_once")):
                raise ValueError("Native success metrics must be binary")
            episode_id = self.episode_ids[self.reset_index]
            path = self.folder / f"{episode_id:06d}.json"
            if path.exists():
                raise RuntimeError("Completed episode records are immutable")
            atomic_json(path, {**self.binding, "episode_id": episode_id,
                               "scene_seed": self.seeds[self.reset_index],
                               "policy_noise_seed": policy_noise_seed(self.binding["env_id"],
                                   self.seeds[self.reset_index], episode_id),
                               "control_steps": self.control_steps,
                               "official_metrics": metrics,
                               "success_at_end": bool(scalar(metrics["success_at_end"])),
                               "success_once": bool(scalar(metrics["success_once"])),
                               "last_reward": scalar(metrics["last_reward"]),
                               "sub_sr_last_reward": scalar(metrics["last_reward"]),
                               "sub_sr_reference": SUBSR_REFERENCE,
                               "return": scalar(metrics["return"]), "status": "completed"})
            self.completed.add(self.reset_index)
        return result



def checkpoint_model_config(payload: dict[str, Any], model_config: str | None = None) -> dict[str, Any]:
    """A shared/split schema change cannot be inferred from state-dict keys."""
    import yaml
    saved_model = payload.get("model_config")
    if saved_model is None:
        config_file = payload.get("model_config_file") or payload["config"].get("model_config_file")
        if not config_file or not payload.get("model_config_sha256"):
            raise ValueError("Checkpoint must embed model_config or identify its exact original YAML")
        config_path = Path(model_config or config_file)
        if sha256_file(config_path) != payload["model_config_sha256"]:
            raise ValueError("Model YAML differs from the checkpoint identity")
        saved_model = yaml.safe_load(config_path.read_text())
    if model_config and yaml.safe_load(Path(model_config).read_text()) != saved_model:
        raise ValueError("Model override changes the checkpoint architecture/recipe; strict state keys alone are insufficient")
    return saved_model

def load_model(checkpoint: Path, device: torch.device,
               model_config: str | None = None, *, require_ema: bool = False
               ) -> tuple[Any, dict[str, Any], str]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict) or "config" not in payload:
        raise ValueError("Expected a BPP TIG training checkpoint with config")
    if require_ema and payload.get("ema_model") is None:
        raise ValueError("Source development evaluation requires an EMA checkpoint")
    saved_model = checkpoint_model_config(payload, model_config)
    from infrastructure.bpp_tig.policy import BPPAgent, BPPVideoPolicy
    policy = BPPVideoPolicy(saved_model, pretrained=False)
    key = "ema_model" if payload.get("ema_model") is not None else "model"
    policy.load_state_dict(payload[key], strict=True)
    policy.to(device).eval()
    return BPPAgent(policy), payload, key


def formal_rows(output: Path, binding_id: str,
                schedule: dict[tuple[str, int], int]) -> dict[tuple[str, int], dict[str, Any]]:
    """Reject unexpected/mixed/duplicate episode records before resuming."""
    result = {}
    for path in sorted(output.glob("*/*.json")):
        if not path.stem.isdigit() or len(path.stem) != 6:
            continue
        row = json.loads(path.read_text())
        key = (row.get("env_id"), row.get("episode_id"))
        if (key not in schedule or key in result or row.get("run_identity") != binding_id
                or row.get("scene_seed") != schedule.get(key) or row.get("status") != "completed"
                or row.get("control_steps") != 500 or path.parent.name != key[0]
                or path.stem != f"{key[1]:06d}"):
            raise ValueError(f"Episode does not match immutable formal schedule: {path}")
        for field in METRIC_FIELDS:
            if scalar(row.get(field)) != scalar(row.get("official_metrics", {}).get(field)):
                raise ValueError("Episode metric differs from native final_info")
        if type(row["success_at_end"]) is not bool or scalar(row["success_at_end"]) not in (0., 1.):
            raise ValueError("Expected binary final success")
        result[key] = row
    return result


def summarize(output: Path, binding_id: str, expected: int,
              schedule: dict[tuple[str, int], int]) -> None:
    rows = list(formal_rows(output, binding_id, schedule).values())
    complete = len(rows) == expected
    groups = {}
    for row in rows:
        groups.setdefault(row["env_id"], []).append(row)
    task_metrics = {env: {"episodes": len(values), **{
        name: float(np.mean([item[name] for item in values])) for name in METRIC_FIELDS}}
        for env, values in groups.items()}
    splits = {}
    for env, values in groups.items():
        splits.setdefault(values[0]["split"] + "/" + env.split("_", 1)[0], []).append(task_metrics[env])
    atomic_json(output / "summary.json", {
        "status": "completed" if complete else "partial", "run_identity": binding_id,
        "completed_episodes": len(rows), "expected_episodes": expected,
        "primary_metric": "success_at_end", "per_task_level_completed_metrics": task_metrics,
        "sr": float(np.mean([row["success_at_end"] for row in rows])) if complete else None,
        "sub_sr_last_reward": float(np.mean([row["last_reward"] for row in rows])) if complete else None,
        "macro_by_split_level": {key: {metric: float(np.mean([item[metric] for item in values]))
            for metric in METRIC_FIELDS} for key, values in splits.items()} if complete else None,
        "sub_sr": "mean last_reward (native dense-reward proxy; reference-table code mapping)",
        "sub_sr_reference": SUBSR_REFERENCE,
        "technical_failure_attempts": len(list((output / "failures").glob("*.json"))),
        "missing_episodes": [{"env_id": env, "episode_id": episode, "scene_seed": seed}
            for (env, episode), seed in schedule.items() if (env, episode) not in {(r["env_id"], r["episode_id"]) for r in rows}]})


def source_dev_protocol(path: Path, runtime: Path) -> dict[str, Any]:
    """Validate a fixed C45 development design; never expand the formal list.

    The JSON contains four semantic source tasks, their complete L0--L3 set,
    and exactly three explicit scene seeds per task-level. No data, model,
    prompt, horizon or seed override is allowed by this mode. The source
    training configurations and the official test list remain authoritative.
    """
    protocol = json.loads(path.read_text())
    required = {"schema_version": "bpp-tig-source-dev-v2", "scope": "source_dev",
                "source_split": "C45", "num_episodes": 3, "weights": "ema_model",
                "max_episode_steps": 500, "obs_horizon": 2, "pred_horizon": 16,
                "execute_horizon": 4, "primary_metric": "success_at_end",
                "sub_sr": "last_reward"}
    if (not isinstance(protocol, dict) or protocol.get("template")
            or protocol.get("model_recipe") != "bpp-tig-split-clip-frozen-prompt-v2"
            or any(protocol.get(k) != v for k, v in required.items())):
        raise ValueError("Unsupported source-dev protocol schema, weights, horizons or metric")
    if protocol.get("protocol_id") != "source_dev_c45_v2":
        raise ValueError("Expected the frozen source_dev_c45_v2 protocol")
    actual_source = runtime_identity(runtime)
    if any(protocol.get("source_identity", {}).get(key) != value for key, value in actual_source.items()):
        raise ValueError("Development runtime identity differs from frozen official source")
    budget = protocol.get("selection", {}).get("fixed_budget_primary_update")
    if (budget not in (237460, 118730)
            or protocol.get("selection", {}).get("scheduler_updates_unchanged") != budget):
        raise ValueError("Choose the declared batch128 or batch256 C45 budget")
    expected_candidates = ([23746, 47492, 94984, 166222, 237460] if budget == 237460
                           else [11873, 23746, 47492, 83111, 118730])
    if protocol.get("checkpoint_candidates") != expected_candidates:
        raise ValueError("Development candidates must match the declared exposure schedule")
    if protocol.get("human_reference") != {"episode_index": 0, "draw_seed": 0, "prompt_frames": 10}:
        raise ValueError("Development prompts must fix episode 0, draw seed 0 and ten frames")
    seeds = protocol.get("scene_seeds")
    if (not isinstance(seeds, list) or len(seeds) != 3
            or any(type(seed) is not int or seed < 0 or seed >= 2**32 for seed in seeds)
            or len(set(seeds)) != 3):
        raise ValueError("Development evaluation requires three unique uint32 scene seeds")
    candidates = protocol.get("checkpoint_candidates")
    if (not isinstance(candidates, list) or not candidates
            or any(type(step) is not int or step <= 0 for step in candidates)
            or candidates != sorted(set(candidates))):
        raise ValueError("Checkpoint candidates must be an ordered unique positive update list")
    config_root = runtime / "examples/baselines/lerobot_dataset/config/exp_configs"
    for key in ("human_config", "sim_config"):
        given = Path(protocol.get(key, ""))
        resolved = (runtime / given).resolve()
        expected = (config_root / f"{key.removesuffix('_config')}_train_config_45.json").resolve()
        if resolved != expected or not resolved.is_file():
            raise ValueError("Source-dev must use the official complete C45 source configurations")
        expected_sha = protocol.get("source_identity", {}).get(f"{key}_sha256")
        if expected_sha != sha256_file(resolved):
            raise ValueError(f"Frozen C45 configuration changed: {key}")
    source_rows = json.loads((runtime / protocol["sim_config"]).read_text())
    source_ids = {row["repo_id"] for row in source_rows if row.get("train") == "0:50"}
    if len(source_ids) != 180 or len(source_rows) != 180:
        raise ValueError("Expected all 180 C45 task-level source repositories")
    official_list = runtime / "examples/baselines/lerobot_dataset/eval/exp_list/seen_plus_unseen_10tasks_env_list.txt"
    formal_ids = {line.strip() for line in official_list.read_text().splitlines() if line.strip()}
    if len(formal_ids) != 40:
        raise ValueError("The authoritative formal evaluation list must contain 40 cases")
    formal_tasks = {env.split("_", 1)[1] for env in formal_ids}
    mapping_file = runtime / "examples/baselines/lerobot_dataset/task_mapping.json"
    formal_config = config_root / "sim_eval_config_seen_plus_unseen_10tasks.json"
    for name, file in (("task_mapping_sha256", mapping_file), ("formal_eval_config_sha256", formal_config)):
        if protocol.get("source_identity", {}).get(name) != sha256_file(file):
            raise ValueError(f"Frozen source/formal membership identity changed: {name}")
    tasks = protocol.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 4 or any(not isinstance(task, dict) for task in tasks):
        raise ValueError("A source-dev design must contain four semantic source tasks")
    task_ids = [task.get("env_id") for task in tasks]
    if (any(not isinstance(task, str) or not task.startswith("TwoRobot") or not task.endswith("-v1")
            for task in task_ids) or len(set(task_ids)) != 4):
        raise ValueError("Malformed or duplicate semantic source tasks")
    if any(task.get("levels") != [0, 1, 2, 3] for task in tasks):
        raise ValueError("Each development task must include all four levels")
    env_ids = protocol.get("env_ids")
    expected_ids = [f"L{level}_{task}" for task in task_ids for level in range(4)]
    if env_ids != expected_ids:
        raise ValueError("Development cases must be the ordered complete four-task by four-level product")
    if formal_tasks.intersection(task_ids) or set(env_ids).intersection(formal_ids):
        raise ValueError("Formal Seen/Unseen evaluation tasks cannot be used for development")
    if not set(env_ids).issubset(source_ids):
        raise ValueError("Every development case must belong to the C45 source split")
    if protocol.get("total_rollouts") != len(env_ids) * 3:
        raise ValueError("Development rollout budget must include every scheduled episode")
    human_rows = json.loads((runtime / protocol["human_config"]).read_text())
    human_ids = {row["repo_id"] for row in human_rows if row.get("train") == "0:50"}
    mapping = {sim_id: row["human_task_id"] for row in json.loads(mapping_file.read_text())["task_mappings"]
               for sim_id in row["sim_task_id"]}
    for task in tasks:
        expected_human = {str(level): mapping[f"L{level}_{task['env_id']}"] for level in range(4)}
        if (task.get("human_repo_by_level") != expected_human
                or set(expected_human.values()) != {task.get("human_repo")}
                or not set(expected_human.values()).issubset(human_ids)):
            raise ValueError("Development human references must follow the official C45 source mapping")
    return protocol



def freeze_source_dev(template: Path, training_config: Path, output: Path) -> dict[str, Any]:
    """Generate real source identities; never ship fabricated hashes as a protocol."""
    from infrastructure.bpp_tig.configuration import (
        RECIPE, configure_tig_imports, load_training_config, validate_training_contract)
    config = load_training_config(training_config)
    validate_training_contract(config, config["training"]["global_batch"] // config["training"]["microbatch"])
    runtime = configure_tig_imports(config)
    if config.get("recipe") != RECIPE or config["training"].get("development"):
        raise ValueError("Freeze source-dev against a full declared v2 training profile")
    protocol = json.loads(template.read_text())
    if protocol.pop("template", None) is not True:
        raise ValueError("Use the unfrozen v2 source-development template")
    budget = config["training"]["scheduler_updates"]
    protocol["checkpoint_candidates"] = list(config["training"]["milestones"])
    protocol["selection"]["fixed_budget_primary_update"] = budget
    protocol["selection"]["scheduler_updates_unchanged"] = budget
    protocol["model_recipe"] = RECIPE
    protocol["source_identity"] = {**runtime_identity(runtime),
        "training_config_sha256": sha256_file(training_config),
        "model_config_sha256": sha256_file(Path(config["model_config_file"]))}
    for key in ("human_config", "sim_config"):
        protocol["source_identity"][f"{key}_sha256"] = sha256_file(runtime / protocol[key])
    for key, relative in (
        ("task_mapping_sha256", "examples/baselines/lerobot_dataset/task_mapping.json"),
        ("formal_eval_config_sha256", "examples/baselines/lerobot_dataset/config/exp_configs/sim_eval_config_seen_plus_unseen_10tasks.json")):
        protocol["source_identity"][key] = sha256_file(runtime / relative)
    if output.exists():
        raise FileExistsError("Frozen source-dev protocols are immutable; choose a new path")
    output.parent.mkdir(parents=True, exist_ok=True)
    import tempfile
    with tempfile.TemporaryDirectory(prefix="bpp-dev-freeze-") as temporary:
        candidate = Path(temporary) / "protocol.json"
        atomic_json(candidate, protocol)
        source_dev_protocol(candidate, runtime)
    atomic_json(output, protocol)
    return protocol

def source_dev_rows(output: Path, binding_id: str, protocol: dict[str, Any]
                    ) -> dict[tuple[str, int], dict[str, Any]]:
    """Read immutable completed records and reject mixed, duplicate or unscheduled cases."""
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    seeds = protocol["scene_seeds"]
    for path in sorted(output.glob("*/*.json")):
        if not (path.stem.isdigit() and len(path.stem) == 6):
            continue
        row = json.loads(path.read_text())
        env, episode = row.get("env_id"), row.get("episode_id")
        if (row.get("run_identity") != binding_id or env not in protocol["env_ids"]
                or type(episode) is not int or not 0 <= episode < 3
                or row.get("scene_seed") != seeds[episode]
                or row.get("policy_noise_seed") != policy_noise_seed(env, seeds[episode], episode)
                or row.get("status") != "completed" or row.get("weight_key") != "ema_model"
                or row.get("split") != "source_dev"
                or row.get("control_steps") != 500
                or path.parent.name != env or path.stem != f"{episode:06d}"):
            raise ValueError(f"Mismatched development record identity or schedule: {path}")
        metrics = row.get("official_metrics", {})
        success = scalar(metrics.get("success_at_end"))
        if (success not in (0., 1.) or type(row.get("success_at_end")) is not bool
                or row["success_at_end"] != bool(success)):
            raise ValueError("Development SR must preserve the native binary final-success metric")
        for key in ("success_once", "last_reward", "return"):
            if scalar(metrics.get(key)) != scalar(row.get(key)):
                raise ValueError("Development records must retain original native metrics")
        case = (env, episode)
        if case in rows:
            raise ValueError("Duplicate development episode")
        rows[case] = row
    return rows


def summarize_source_dev(output: Path, binding_id: str, protocol: dict[str, Any]) -> None:
    """Do not score incomplete or technically failed rollouts as task failures.

    Official SR is available only once all 48 scheduled episodes complete.
    Completed task failures (success_at_end=False) always count in the mean.
    Partial estimates are named explicitly and the missing schedule is retained.
    """
    rows = source_dev_rows(output, binding_id, protocol)
    per_task_level: dict[str, Any] = {}
    missing: list[dict[str, Any]] = []
    for env in protocol["env_ids"]:
        values = [rows[(env, episode)] for episode in range(3) if (env, episode) in rows]
        missing.extend({"env_id": env, "episode_id": episode,
                        "scene_seed": protocol["scene_seeds"][episode]}
                       for episode in range(3) if (env, episode) not in rows)
        partial = {key: float(np.mean([row[key] for row in values])) if values else None
                   for key in METRIC_FIELDS}
        per_task_level[env] = {"completed_episodes": len(values), "expected_episodes": 3,
                               "sr": partial["success_at_end"] if len(values) == 3 else None,
                               "partial_completed_metrics": partial}
    failure_files = sorted((output / "failures").glob("*.json"))
    complete = not missing
    completed_group_means = [row["partial_completed_metrics"]["success_at_end"]
                             for row in per_task_level.values() if row["completed_episodes"]]
    atomic_json(output / "summary.json", {
        "status": "completed" if complete else "partial", "scope": "source_dev",
        "protocol_id": protocol["protocol_id"], "run_identity": binding_id,
        "completed_episodes": len(rows), "expected_episodes": 48,
        "primary_metric": "success_at_end", "aggregation": "macro average of 16 task-level SRs; three episodes each",
        "sr": float(np.mean([row["sr"] for row in per_task_level.values()])) if complete else None,
        "partial_completed_macro_sr": float(np.mean(completed_group_means)) if completed_group_means else None,
        "per_task_level": per_task_level, "missing_episodes": missing,
        "technical_failure_attempts": len(failure_files),
        "technical_failure_files": [str(path.relative_to(output)) for path in failure_files],
        "sub_sr": "last_reward", "sub_sr_reference": SUBSR_REFERENCE,
        "sub_sr_last_reward": float(np.mean([row["last_reward"] for row in rows.values()])) if complete else None,
        "formal_test_results": "NOT_RUN"})


def run_source_dev(args: argparse.Namespace) -> None:
    """Evaluate the frozen source-only design without admitting formal test tasks."""
    # Checkpoint is trusted local model input; map CPU before loading CUDA modules.
    header = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    deployment = evaluation_training_config(header, args.data_config)
    del header
    from infrastructure.bpp_tig.data import TIGBPPDataset
    from examples.baselines.act.eval_act_imitator import (
        clear_l_level, evaluate_with_task_encoder, make_eval_envs_with_level,
    )
    from examples.baselines.lerobot_dataset.lerobot_paired_dataset import InputMode
    from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
    runtime = Path(deployment["tig_source_root"])
    protocol = source_dev_protocol(args.source_dev_protocol, runtime)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device(args.device)
    agent, payload, weight_key = load_model(args.checkpoint, device, require_ema=True)
    updates = payload.get("optimizer_updates")
    if (payload.get("model_config", {}).get("recipe") != protocol.get("model_recipe")
            or payload.get("model_config_sha256") != protocol["source_identity"].get("model_config_sha256")):
        raise ValueError("Source-dev checkpoint recipe/model differs from frozen training profile")
    if (payload["config"]["training"]["scheduler_updates"] != protocol["selection"]["fixed_budget_primary_update"]
            or payload["config"]["training"]["global_batch"] !=
               (128 if protocol["selection"]["fixed_budget_primary_update"] == 237460 else 256)):
        raise ValueError("Checkpoint training exposure/batch differs from frozen development profile")
    if updates not in protocol["checkpoint_candidates"]:
        raise ValueError("Checkpoint update is not one of the frozen development candidates")
    data_config = deployment["data"]
    for name in ("human_config", "sim_config"):
        if sha256_file(Path(data_config[name])) != sha256_file(runtime / protocol[name]):
            raise ValueError("Checkpoint was not trained with the frozen C45 source configurations")
    for name in ("human_config", "sim_config"):
        original_sha = payload.get("data_manifest", {}).get("config_files", {}).get(name, {}).get("sha256")
        if original_sha != protocol["source_identity"].get(f"{name}_sha256"):
            raise ValueError("Original checkpoint data membership differs from frozen source protocol")
    source_groups = payload.get("data_manifest", {}).get("groups", {})
    if len(source_groups) != 180 or not set(protocol["env_ids"]).issubset(source_groups):
        raise ValueError("Development checkpoint must have used the full C45 source split")
    evaluation_config = {**data_config,
        "human_config": str(runtime / protocol["human_config"]),
        "sim_config": str(runtime / protocol["sim_config"]), "episodes": None,
        "task_filter": protocol["env_ids"], "split": "train", "training": False}
    dataset = TIGBPPDataset(evaluation_config, training=False)
    try:
        prompts: dict[str, Any] = {}
        for env_id in protocol["env_ids"]:
            video, metadata = dataset.get_human_prompt(env_id, episode_index=0, draw_seed=0)
            video = torch.as_tensor(video, dtype=torch.float32)
            if video.shape != (10, 3, 224, 224) or not torch.isfinite(video).all():
                raise ValueError("Development prompt must contain ten finite RGB224 frames")
            prompts[env_id] = {"metadata": plain(metadata),
                               "rgb_sha256": hashlib.sha256(video.contiguous().numpy().tobytes()).hexdigest()}
        resolved_ids = {env: resolved_environment_id(env) for env in protocol["env_ids"]}
        checkpoint_sha = sha256_file(args.checkpoint)
        identity = {"scope": "source_dev", "evaluation_scope": "source_dev", "protocol": protocol,
            "protocol_file_sha256": sha256_file(args.source_dev_protocol),
            "source_dev_protocol_sha256": sha256_file(args.source_dev_protocol),
            "checkpoint_sha256": checkpoint_sha, "weights": weight_key,
            "optimizer_updates": updates, "human_references": prompts,
            "resolved_environment_ids": resolved_ids, "control_mode": "pd_joint_pos",
            "policy_noise_seed_strategy": "sha256(bpp-native-eval-v1/env_id/scene_seed/episode_id)[:4] little endian; after native reset",
            "data_config": plain(data_config), "data_manifest": plain(payload["data_manifest"]),
            "evaluation_data_manifest": dataset.manifest,
            "runtime_source": str(runtime), "evaluate_source_sha256": sha256_file(Path(__file__)),
            "runtime_eval_source_sha256": sha256_file(runtime / "examples/baselines/act/eval_act_imitator.py"),
            "current_model_source_sha256": sha256_file(ROOT / "infrastructure/bpp_tig/policy.py"),
            "model_source_identity": plain(payload.get("source_identity")),
            "model_schema": payload.get("model_config", {}).get("schema_version"),
            "model_recipe": payload.get("model_config", {}).get("recipe", "legacy-shared-clip-v1"),
            "runtime_identity": runtime_identity(runtime), "sub_sr_reference": SUBSR_REFERENCE}
        run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        identity_file = args.output / "identity.json"
        if identity_file.exists() and json.loads(identity_file.read_text()).get("run_identity") != run_id:
            raise ValueError("Development output belongs to a different checkpoint/protocol/code/prompt identity")
        atomic_json(identity_file, {**identity, "run_identity": run_id})
        rows = source_dev_rows(args.output, run_id, protocol)
        summarize_source_dev(args.output, run_id, protocol)
        for env_id in protocol["env_ids"]:
            todo = [episode for episode in range(3) if (env_id, episode) not in rows]
            if not todo:
                continue
            processor = HistoryProcessor(dataset, env_id, prompts[env_id])
            wrapped_agent = NativeLoopAgent(agent, processor)
            wrapped_agent.clear_cache()
            env = None
            recording = None
            attempt_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            try:
                env = make_eval_envs_with_level(resolved_ids[env_id], env_id.split("_", 1)[0], 1, "physx_cpu",
                    {"control_mode": "pd_joint_pos", "reward_mode": "dense", "obs_mode": "rgb",
                     "render_mode": "rgb_array", "max_episode_steps": 500,
                     "sensor_configs": {"shader_pack": "rt-fast"},
                     "human_render_camera_configs": {"shader_pack": "rt-fast"}},
                    {"obs_horizon": 1}, str(args.output / "videos" / env_id / attempt_id) if args.video else None,
                    [FlattenRGBDObservationWrapper])
                recording = SeededRecorder(env, todo, [protocol["scene_seeds"][episode] for episode in todo],
                    args.output / env_id, {"run_identity": run_id, "env_id": env_id, "split": "source_dev",
                    "resolved_environment_id": resolved_ids[env_id], "checkpoint_sha256": checkpoint_sha,
                    "weight_key": weight_key, "protocol_file_sha256": identity["protocol_file_sha256"],
                    "evaluation_attempt": attempt_id,
                    "expected_control_steps": 500,
                    "human_reference": prompts[env_id]})
                metrics = evaluate_with_task_encoder(n=len(todo), agent=wrapped_agent, eval_envs=recording,
                    eval_kwargs={"env_id": env_id, "delta_control": False, "pred_horizon": 4,
                                 "temporal_agg": False, "light_temporal_agg": False,
                                 "max_timesteps": 500, "device": device, "sim_backend": "physx_cpu"},
                    evaluate_processor=processor, input_mode=InputMode.VIDEO_ONLY, progress_bar=True)
                if recording.completed != set(range(len(todo))):
                    raise RuntimeError("Native loop returned without recording every scheduled episode")
                native_path = args.output / env_id / f"native-metrics-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
                atomic_json(native_path, {"run_identity": run_id, "raw_official_metrics": plain(metrics),
                                         "human_reference": processor.human_metadata,
                                         "evaluated_episode_ids": todo})
            except BaseException as error:
                index = recording.reset_index if recording is not None else -1
                episode = todo[index] if 0 <= index < len(todo) else None
                failure_path = args.output / "failures" / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-{env_id}.json"
                atomic_json(failure_path, {"status": "technical_failure", "run_identity": run_id,
                    "env_id": env_id, "episode_id": episode, "scheduled_episode_ids": todo,
                    "exception_type": type(error).__name__, "exception": str(error), "traceback": traceback.format_exc()})
                raise
            finally:
                if env is not None:
                    env.close()
                # Match native evaluator cleanup between sixteen scene builds.
                # The recorder also owns the environment, so release both
                # references before collecting SAPIEN/Vulkan render contexts.
                recording = None
                env = None
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                time.sleep(1)
                clear_l_level()
                summarize_source_dev(args.output, run_id, protocol)
        summarize_source_dev(args.output, run_id, protocol)
    finally:
        dataset.close()


def run(args: argparse.Namespace) -> None:
    """Run pinned ACT environment behavior and one recorded human prompt per group.

    Formal reference pools and uniform-jitter sampling follow the latest ACT/DP
    launchers. The prompt is deterministic per task-level; resume therefore uses
    the same reference rather than silently sampling a new video.
    """
    header = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    deployment = evaluation_training_config(header, args.data_config)
    del header
    from infrastructure.bpp_tig.data import TIGBPPDataset
    from examples.baselines.act.eval_act_imitator import (
        clear_l_level, evaluate_with_task_encoder, make_eval_envs_with_level)
    from examples.baselines.lerobot_dataset.lerobot_paired_dataset import InputMode
    from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
    runtime = Path(deployment["tig_source_root"])
    source_identity = runtime_identity(runtime)
    official_list = runtime / "examples/baselines/lerobot_dataset/eval/exp_list/seen_plus_unseen_10tasks_env_list.txt"
    all_envs = [line.strip() for line in official_list.read_text().splitlines() if line.strip()]
    env_ids = args.env_ids or all_envs
    if (len(all_envs) != 40 or len(set(env_ids)) != len(env_ids)
            or any(env not in all_envs for env in env_ids)):
        raise ValueError("Use unique IDs from the official 40-case evaluation list")
    if (args.num_episodes < 1 or args.max_steps != 500 or args.seed < 0
            or args.seed + args.num_episodes >= 2**32):
        raise ValueError("Use positive episode count, uint32 seeds and native 500 controls")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    agent, payload, weight_key = load_model(args.checkpoint, device, args.model_config)
    config_root = runtime / "examples/baselines/lerobot_dataset/config/exp_configs"
    data_config = {**deployment["data"],
        "human_config": str(config_root / "human_eval_config_seen_plus_unseen_10tasks.json"),
        "sim_config": str(config_root / "sim_eval_config_seen_plus_unseen_10tasks.json"),
        "episodes": None, "task_filter": env_ids, "split": "train", "training": False}
    dataset = TIGBPPDataset(data_config, training=False)
    try:
        prompts = {}
        for env in env_ids:
            video, metadata = dataset.get_human_prompt(env, draw_seed=args.seed, official_eval=True)
            prompts[env] = {"metadata": plain(metadata),
                "rgb_sha256": hashlib.sha256(video.contiguous().numpy().tobytes()).hexdigest()}
        checkpoint_sha = sha256_file(args.checkpoint)
        resolved_ids = {env: resolved_environment_id(env) for env in env_ids}
        identity = {"scope": "formal" if len(env_ids) == 40 and args.num_episodes == 10 else "formal_subset",
            "checkpoint_sha256": checkpoint_sha, "weights": weight_key,
            "optimizer_updates": payload.get("optimizer_updates"), "env_ids": env_ids,
            "num_episodes": args.num_episodes, "scene_seed_start": args.seed,
            "max_episode_steps": 500, "observation_horizon": 2, "prompt_frames": 10,
            "prediction_horizon": 16, "execute_actions": 4, "control_mode": "pd_joint_pos",
            "resolved_environment_ids": resolved_ids, "runtime_identity": source_identity,
            "environment_behavior": "Pinned ACT make_eval_envs_with_level; native legacy level flags retained",
            "human_references": prompts, "human_reference_sampling": "official uniform_jitter; combined pool 0:50; one deterministic prompt per task-level group",
            "policy_noise_seed_strategy": "sha256(bpp-native-eval-v1/env_id/scene_seed/episode_id)[:4] little endian; after reset",
            "data_config": plain(data_config), "data_manifest": plain(payload.get("data_manifest")),
            "evaluation_data_manifest": dataset.manifest,
            "normalization_calibration": "Official per-repository released q01/q99, including target statistics; no target parameter updates",
            "runtime_source": str(runtime), "evaluate_source_sha256": sha256_file(Path(__file__)),
            "data_source_sha256": sha256_file(Path(__file__).with_name("data.py")),
            "policy_source_sha256": sha256_file(Path(__file__).with_name("policy.py")),
            "configuration_source_sha256": sha256_file(Path(__file__).with_name("configuration.py")),
            "model_source_identity": plain(payload.get("source_identity")),
            "model_schema": payload.get("model_config", {}).get("schema_version"),
            "model_recipe": payload.get("model_config", {}).get("recipe", "legacy-shared-clip-v1"),
            "sub_sr": "mean last_reward, dense-reward proxy", "sub_sr_reference": SUBSR_REFERENCE}
        run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        identity_file = args.output / "identity.json"
        if identity_file.exists() and json.loads(identity_file.read_text()).get("run_identity") != run_id:
            raise ValueError("Output belongs to a different checkpoint/protocol/code/reference")
        atomic_json(identity_file, {**identity, "run_identity": run_id})
        schedule = {(env, ep): args.seed + ep for env in env_ids for ep in range(args.num_episodes)}
        rows = formal_rows(args.output, run_id, schedule)
        summarize(args.output, run_id, len(schedule), schedule)
        for env_id in env_ids:
            todo = [ep for ep in range(args.num_episodes) if (env_id, ep) not in rows]
            if not todo:
                continue
            processor = HistoryProcessor(dataset, env_id, prompts[env_id],
                official_eval=True, prompt_draw_seed=args.seed)
            wrapped_agent = NativeLoopAgent(agent, processor)
            wrapped_agent.clear_cache()
            env, recording = None, None
            attempt = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            try:
                env = make_eval_envs_with_level(resolved_ids[env_id], env_id.split("_", 1)[0], 1, "physx_cpu",
                    {"control_mode": "pd_joint_pos", "reward_mode": "dense", "obs_mode": "rgb",
                     "render_mode": "rgb_array", "max_episode_steps": 500,
                     "sensor_configs": {"shader_pack": "rt-fast"},
                     "human_render_camera_configs": {"shader_pack": "rt-fast"}},
                    {"obs_horizon": 1}, str(args.output / "videos" / env_id / attempt) if args.video else None,
                    [FlattenRGBDObservationWrapper])
                recording = SeededRecorder(env, todo, [schedule[(env_id, ep)] for ep in todo],
                    args.output / env_id, {"run_identity": run_id, "env_id": env_id,
                    "split": "seen" if all_envs.index(env_id) < 20 else "unseen",
                    "resolved_environment_id": resolved_ids[env_id], "checkpoint_sha256": checkpoint_sha,
                    "weight_key": weight_key, "human_reference": prompts[env_id],
                    "evaluation_attempt": attempt, "expected_control_steps": 500})
                metrics = evaluate_with_task_encoder(n=len(todo), agent=wrapped_agent, eval_envs=recording,
                    eval_kwargs={"env_id": env_id, "delta_control": False, "pred_horizon": 4,
                        "temporal_agg": False, "light_temporal_agg": False,
                        "max_timesteps": 500, "device": device, "sim_backend": "physx_cpu"},
                    evaluate_processor=processor, input_mode=InputMode.VIDEO_ONLY, progress_bar=True)
                if recording.completed != set(range(len(todo))):
                    raise RuntimeError("Native loop omitted a scheduled finalized episode")
                atomic_json(args.output / env_id / f"native-metrics-{attempt}.json", {
                    "run_identity": run_id, "raw_official_metrics": plain(metrics),
                    "human_reference": processor.human_metadata, "evaluated_episode_ids": todo})
            except BaseException as error:
                atomic_json(args.output / "failures" / f"{attempt}-{env_id}.json", {
                    "status": "technical_failure", "run_identity": run_id, "env_id": env_id,
                    "scheduled_episode_ids": todo, "exception_type": type(error).__name__,
                    "exception": str(error), "traceback": traceback.format_exc()})
                raise
            finally:
                if env is not None:
                    env.close()
                recording, env = None, None
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                clear_l_level()
                summarize(args.output, run_id, len(schedule), schedule)
    finally:
        dataset.close()


def self_test() -> None:
    """CPU-only fixtures: no simulator, model download, dataset or CUDA context."""
    import copy
    import tempfile
    from unittest.mock import patch
    class Dataset:
        def normalize_state(self, value: np.ndarray, env: str) -> np.ndarray:
            return value / 2
        def denormalize_action(self, value: np.ndarray, env: str) -> np.ndarray:
            return value * 3
    proc = HistoryProcessor(Dataset(), "L0_Task")
    for step in range(5):
        proc.normalize_state_rgb(torch.full((1, 1, 18), float(step)),
            torch.full((1, 1, 224, 224, 3), step, dtype=torch.uint8), "L0_Task")
    assert proc.history()["state"][0, :, 0].tolist() == [1.5, 2.]
    assert proc.history()["rgb"].shape == (1, 2, 224, 224, 3)
    assert torch.equal(proc.denormalize_action(torch.ones(1, 16), "L0_Task"), torch.full((1, 16), 3.))
    class Env:
        def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
            self.seed, self.steps = kwargs["seed"], 0
            return {}, {}
        def step(self, action: Any) -> tuple[Any, ...]:
            self.steps += 1
            return {}, 0., False, np.array([self.steps == 500]), {"final_info": {"episode": {
                "success_at_end": np.array([False]), "success_once": np.array([True]),
                "last_reward": np.array([.25]), "return": np.array([2.5])}}}
    with tempfile.TemporaryDirectory(prefix="bpp-evaluate-cpu-") as directory:
        root = Path(directory)
        import yaml
        legacy = {"schema_version": 1, "vision": {"frozen": False}}
        split = {"schema_version": 2, "recipe": "bpp-tig-split-clip-frozen-prompt-v2"}
        override = root / "override.yaml"
        override.write_text(yaml.safe_dump(split))
        try:
            checkpoint_model_config({"config": {}, "model_config": legacy}, str(override))
        except ValueError:
            pass
        else:
            raise AssertionError("Legacy shared weights accepted a split-model override")
        assert checkpoint_model_config({"config": {}, "model_config": split}, str(override)) == split
        template = json.loads((ROOT / "infrastructure/bpp_tig/configs/source_dev_c45_v2.template.json").read_text())
        protocol = copy.deepcopy(template)
        protocol.pop("template")
        protocol["checkpoint_candidates"] = [23746, 47492, 94984, 166222, 237460]
        protocol["selection"]["fixed_budget_primary_update"] = 237460
        protocol["selection"]["scheduler_updates_unchanged"] = 237460
        protocol["model_recipe"] = "bpp-tig-split-clip-frozen-prompt-v2"
        tasks = [(task["env_id"], task["human_repo"]) for task in protocol["tasks"]]
        tasks += [(f"TwoRobotFixture{i}-v1", f"human_fixture{i}") for i in range(41)]
        config_root = root / "examples/baselines/lerobot_dataset/config/exp_configs"
        config_root.mkdir(parents=True)
        (root / protocol["sim_config"]).write_text(json.dumps([
            {"repo_id": f"L{level}_{task}", "train": "0:50"} for task, _ in tasks for level in range(4)]))
        (root / protocol["human_config"]).write_text(json.dumps([
            {"repo_id": human, "train": "0:50"} for _, human in tasks]))
        mapping = root / "examples/baselines/lerobot_dataset/task_mapping.json"
        mapping.write_text(json.dumps({"task_mappings": [
            {"human_task_id": human, "sim_task_id": [f"L{level}_{task}" for level in range(4)]}
            for task, human in tasks]}))
        formal_config = config_root / "sim_eval_config_seen_plus_unseen_10tasks.json"
        formal_config.write_text("[]")
        formal_list = root / "examples/baselines/lerobot_dataset/eval/exp_list/seen_plus_unseen_10tasks_env_list.txt"
        formal_list.parent.mkdir(parents=True)
        formal_list.write_text("\n".join(f"L{level}_TwoRobotFormal{i}-v1" for i in range(10) for level in range(4)))
        source = {"runtime_commit": OFFICIAL_TIG_COMMIT, "runtime_files_sha256": {}}
        protocol["source_identity"] = {**source,
            "human_config_sha256": sha256_file(root / protocol["human_config"]),
            "sim_config_sha256": sha256_file(root / protocol["sim_config"]),
            "task_mapping_sha256": sha256_file(mapping),
            "formal_eval_config_sha256": sha256_file(formal_config)}
        protocol_file = root / "protocol.json"
        atomic_json(protocol_file, protocol)
        with patch(__name__ + ".runtime_identity", return_value=source):
            assert source_dev_protocol(protocol_file, root)["total_rollouts"] == 48
            for key, value in (("scene_seeds", [1, 1, 2]), ("checkpoint_candidates", [18000, 100000]),
                               ("weights", "model"), ("env_ids", protocol["env_ids"][:-1])):
                bad = {**protocol, key: value}
                atomic_json(root / "bad.json", bad)
                try:
                    source_dev_protocol(root / "bad.json", root)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"Malformed protocol accepted: {key}")
        results = root / "results"
        results.mkdir()
        summarize_source_dev(results, "fixture", protocol)
        assert json.loads((results / "summary.json").read_text())["sr"] is None
        for env in protocol["env_ids"]:
            for episode, seed in enumerate(protocol["scene_seeds"]):
                recorder = SeededRecorder(Env(), [episode], [seed], results / env,
                    {"run_identity": "fixture", "env_id": env, "split": "source_dev",
                     "weight_key": "ema_model", "expected_control_steps": 500})
                recorder.reset()
                initial_noise = torch.randn(2, 16)
                for _ in range(500):
                    recorder.step(None)
                recorder.reset()
                assert torch.equal(torch.randn(2, 16), initial_noise)
        summarize_source_dev(results, "fixture", protocol)
        summary = json.loads((results / "summary.json").read_text())
        assert summary["status"] == "completed" and summary["sr"] == 0.
        assert all(row["last_reward"] == .25 for row in source_dev_rows(results, "fixture", protocol).values())
        schedule = {(env, ep): seed for env in protocol["env_ids"]
                    for ep, seed in enumerate(protocol["scene_seeds"])}
        summarize(results, "fixture", 48, schedule)
        assert json.loads((results / "summary.json").read_text())["status"] == "completed"
        damaged = results / protocol["env_ids"][0] / "000000.json"
        row = json.loads(damaged.read_text())
        row["last_reward"] = .9
        atomic_json(damaged, row)
        try:
            formal_rows(results, "fixture", schedule)
        except ValueError:
            pass
        else:
            raise AssertionError("Modified native reward was accepted")
    assert not torch.cuda.is_initialized()
    print("BPP_EVALUATE_SELF_TEST_PASS: O2; normalize/inverse; native terminal SR/last_reward; deterministic noise; complete 48-case source schedule; partial SR=null; malformed protocol/native-metric rejection; no CUDA")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--freeze-source-dev-template", type=Path)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-config")
    parser.add_argument("--data-config")
    parser.add_argument("--source-dev-protocol", type=Path,
                        help="Use only the frozen C45 source-development protocol, separate from formal testing")
    parser.add_argument("--env-ids", nargs="+")
    parser.add_argument("--num-episodes", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.freeze_source_dev_template:
        if not args.training_config or not args.output:
            parser.error("Freezing requires --training-config and --output")
        freeze_source_dev(args.freeze_source_dev_template, args.training_config, args.output)
        print("SOURCE_DEV_PROTOCOL_FROZEN:", args.output)
    elif args.checkpoint and args.output:
        if args.source_dev_protocol and any(value is not None for value in (
                args.model_config, args.env_ids, args.num_episodes, args.max_steps, args.seed)):
            parser.error("Source-dev cases, data/model configuration, seeds and horizons are fixed by the protocol")
        if not args.source_dev_protocol:
            args.num_episodes = 10 if args.num_episodes is None else args.num_episodes
            args.max_steps = 500 if args.max_steps is None else args.max_steps
            args.seed = 0 if args.seed is None else args.seed
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / ".evaluation.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.source_dev_protocol:
                run_source_dev(args)
            else:
                run(args)
    else:
        parser.error("Supply --checkpoint and --output, or --self-test")


if __name__ == "__main__":
    main()
