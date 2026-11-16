"""Full BPP training on TIG, with optimizer-update budgets and resumable draws.

Batch contract: obs.rgb (B,2,3,224,224), obs.qpos (B,2,18),
obs.prompt.obs.rgb (B,P,3,224,224), metadata.mask (B,P; True=pad),
action (B,16,16), with TIG-normalized joint targets. No target-test feedback
enters this trainer. Sampling tickets contain their own augmentation seeds.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterator, Mapping

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler
import yaml

from infrastructure.bpp_tig.configuration import (
    configure_tig_imports, load_training_config, validate_c45_manifest,
    validate_model_recipe, validate_training_contract,
)

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(config: Mapping[str, Any] | None = None, *, fork_root: Path = ROOT) -> dict[str, Any]:
    """Identify this fork and the explicitly selected TIG source, including diffs.

    There is no assumed parent workspace or nested BPP checkout. Configuration
    hashes include new, not-yet-committed adapter files as well as tracked files.
    """
    selected = config.get("tig_source_root") if config is not None else os.environ.get("TIG_SOURCE_ROOT")
    if not selected:
        raise ValueError("Set TIG_SOURCE_ROOT explicitly before recording source identity")
    runtime_root = Path(selected).expanduser().resolve()
    fork_root = fork_root.resolve()
    result: dict[str, Any] = {"files": {}, "repositories": {}}
    for path in sorted((fork_root / "infrastructure/bpp_tig").rglob("*")):
        if path.is_file() and path.suffix in {".py", ".yaml", ".json"}:
            result["files"][str(path.relative_to(fork_root))] = sha256(path)
    for name, root in (("bpp_fork", fork_root), ("tig_runtime", runtime_root)):
        args = ["git", "-C", str(root)]
        result["repositories"][name] = {
            "path": str(root),
            "commit": subprocess.check_output(args + ["rev-parse", "HEAD"], text=True).strip(),
            "diff_sha256": hashlib.sha256(subprocess.check_output(args + ["diff", "HEAD"])).hexdigest(),
        }
    return result


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def write_run_provenance(output: Path, config: Mapping[str, Any], source: Mapping[str, Any],
                         world_size: int, accumulation: int, resume_signature: str,
                         *, resume: bool, fork_root: Path = ROOT,
                         command: list[str] | None = None) -> None:
    """Commit a small source snapshot and initial launch identity once, or verify it.

    No datasets, model weights or arbitrary environment variables are copied.
    The checkpoint signature ties launch identity to the exact resumed sampler,
    data, model and optimizer recipe; original launch metadata is never replaced.
    """
    launch_path = output / "launch.json"
    snapshot = output / "source"
    hashes_path = snapshot / "sha256.json"
    identity_path = snapshot / "identity.json"
    config_path = output / "config.yaml"
    files = dict(source["files"])
    if resume:
        for path in (launch_path, hashes_path, identity_path, config_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing original run provenance for resume: {path}")
        launch = json.loads(launch_path.read_text(encoding="utf-8"))
        if (launch["resume_signature"] != resume_signature or launch["world_size"] != world_size
                or launch["gradient_accumulation"] != accumulation):
            raise ValueError("Resume launch identity differs from current recipe/batch/source")
        if json.loads(identity_path.read_text(encoding="utf-8")) != source:
            raise ValueError("Resume source identity differs from the original source snapshot")
        if json.loads(hashes_path.read_text(encoding="utf-8")) != files:
            raise ValueError("Resume source SHA256 stamp differs from current source")
        if yaml.safe_load(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Resume resolved config differs from original config.yaml")
        for relative, digest in files.items():
            path = snapshot / relative
            if not path.is_file() or sha256(path) != digest:
                raise ValueError(f"Resume source snapshot missing or changed: {relative}")
        patch = snapshot / "bpp-working-tree.patch"
        if (not patch.is_file() or sha256(patch)
                != source["repositories"]["bpp_fork"]["diff_sha256"]):
            raise ValueError("Resume BPP working-tree patch missing or changed")
        return
    if launch_path.exists() or snapshot.exists() or config_path.exists():
        raise FileExistsError(f"Existing run provenance must not be overwritten: {output}; use --resume")
    output.mkdir(parents=True, exist_ok=True)
    for relative, digest in files.items():
        original = fork_root / relative
        if not original.resolve().is_relative_to(fork_root.resolve()):
            raise ValueError(f"Source path escapes the fork: {relative}")
        content = original.read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f"Source changed while preparing launch: {relative}")
        atomic_bytes(snapshot / relative, content)
    fork_patch = subprocess.check_output(["git", "-C", str(fork_root), "diff", "HEAD"])
    if hashlib.sha256(fork_patch).hexdigest() != source["repositories"]["bpp_fork"]["diff_sha256"]:
        raise ValueError("BPP source changed while preparing launch provenance")
    atomic_bytes(snapshot / "bpp-working-tree.patch", fork_patch)
    atomic_json(identity_path, source)
    atomic_json(hashes_path, files)
    atomic_bytes(config_path, yaml.safe_dump(dict(config), allow_unicode=True).encode("utf-8"))
    supplied = ([sys.executable, "-m", "infrastructure.bpp_tig.train", *sys.argv[1:]]
                if command is None else command)
    # Parser accepts no secret/token options. Bound this process-level argv and
    # deliberately avoid dumping launcher state or the process environment.
    recorded_command = [str(argument)[:4096] for argument in supplied[:128]]
    train = config["training"]
    atomic_json(launch_path, {
        "schema_version": 2,
        "launched_at_utc": datetime.now(timezone.utc).isoformat(),
        "output": str(output.resolve()), "command": recorded_command,
        "resolved_config": config, "world_size": world_size,
        "seed": train["seed"], "optimizer_update_budget": train["max_updates"],
        "scheduler_updates": train["scheduler_updates"],
        "global_batch": train["global_batch"], "microbatch_per_gpu": train["microbatch"],
        "gradient_accumulation": accumulation, "resume_signature": resume_signature,
        "development": train["development"], "recipe": config["recipe"],
        "source_identity_file": "source/identity.json",
        "source_sha256_file": "source/sha256.json",
    })


def cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def device_tree(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: device_tree(item, device) for key, item in value.items()}
    return value


def rng_state() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None}


def restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])


class TicketBatchSampler(Sampler[list[tuple[int, int]]]):
    """Counter-based group-balanced batches; resume never depends on prefetch.

    A global microbatch chooses one task-level stratum. All ranks draw uniformly
    within that stratum and receive distinct augmentation seeds per draw.
    """

    def __init__(self, groups: Mapping[str, range], microbatch: int, rank: int,
                 world_size: int, seed: int, first_microstep: int,
                 total_microsteps: int) -> None:
        self.groups = [groups[key] for key in sorted(groups)]
        assert self.groups and all(len(group) > 0 for group in self.groups)
        assert microbatch > 0 and 0 <= rank < world_size
        assert 0 <= first_microstep <= total_microsteps
        self.microbatch, self.rank, self.world_size = microbatch, rank, world_size
        self.seed, self.first, self.total = seed, first_microstep, total_microsteps

    def __len__(self) -> int:
        return self.total - self.first

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        for counter in range(self.first, self.total):
            generator = np.random.default_rng(np.random.SeedSequence([self.seed, counter]))
            group = self.groups[int(generator.integers(len(self.groups)))]
            slots = self.microbatch * self.world_size
            offsets = generator.integers(len(group), size=slots)
            seeds = generator.integers(0, 2**32, size=slots, dtype=np.uint64)
            start = self.rank * self.microbatch
            yield [(int(group[int(offsets[i])]), int(seeds[i]))
                   for i in range(start, start + self.microbatch)]


class LossModule(nn.Module):
    def __init__(self, policy: nn.Module) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        return self.policy.compute_loss(batch)


def lr_factor(update: int, warmup: int, total: int) -> float:
    assert 0 <= warmup < total
    if update < warmup:
        return (update + 1) / max(warmup, 1)
    progress = min(max((update - warmup) / (total - warmup), 0.0), 1.0)
    return 0.5 * (1 + math.cos(math.pi * progress))


def should_checkpoint(completed: int, config: Mapping[str, Any], stopping: bool) -> bool:
    """Persist every declared evaluation milestone, including off-grid updates."""
    return (completed % config["checkpoint_every"] == 0 or completed in config["milestones"]
            or completed == config["max_updates"] or stopping)


def signature(config: dict[str, Any], manifest: Any, model_hash: str,
              world_size: int, accumulation: int) -> str:
    train = config["training"]
    fields = ["seed", "scheduler_updates", "warmup_updates", "global_batch",
              "microbatch", "lr", "weight_decay", "betas", "clip_grad_norm", "fixed_batch", "pretrained",
              "development"]
    value = {"data": config["data"], "training": {k: train[k] for k in fields},
             "recipe": config["recipe"], "budget": config["budget"],
             "manifest": manifest, "model_config_sha256": model_hash,
             "world_size": world_size, "accumulation": accumulation}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save_checkpoint(path: Path, policy: nn.Module, ema: Any,
                    optimizer: torch.optim.Optimizer, scheduler: Any,
                    payload: dict[str, Any], rank: int, world_size: int) -> None:
    local_rng = rng_state()
    rank_rngs: list[Any] = [None] * world_size
    if dist.is_initialized():
        dist.all_gather_object(rank_rngs, local_rng)
    else:
        rank_rngs[0] = local_rng
    if rank == 0:
        raw = cpu_tree(policy.state_dict())
        ema.store(policy.parameters())
        try:
            ema.copy_to(policy.parameters())
            averaged = cpu_tree(policy.state_dict())
        finally:
            ema.restore(policy.parameters())
        payload = {**payload, "model": raw, "ema_model": averaged,
                   "ema": cpu_tree(ema.state_dict()),
                   "optimizer": cpu_tree(optimizer.state_dict()),
                   "scheduler": scheduler.state_dict(), "rank_rngs": rank_rngs}
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".incomplete")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        atomic_json(path.parent / "latest.json", {"checkpoint": str(path.resolve()),
                    "optimizer_updates": payload["optimizer_updates"],
                    "bytes": path.stat().st_size})
    if dist.is_initialized():
        dist.barrier()
    restore_rng(local_rng)


def probe(policy: nn.Module, batch: dict[str, Any], device: torch.device) -> float:
    was_training = policy.training
    policy.eval()
    devices = [device.index] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(1729)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                result = policy.compute_loss(batch)
        assert torch.isfinite(result)
        return float(result)
    finally:
        policy.train(was_training)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "infrastructure/bpp_tig/configs/c45.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--microbatch", type=int)
    parser.add_argument("--global-batch", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--warmup-updates", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--probe-every", type=int)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--task-filter", nargs="+")
    parser.add_argument("--episodes", type=int, nargs=2)
    parser.add_argument("--fixed-batch", action="store_true")
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    assert args.output_dir is not None, "--output-dir is required for training"
    # Resolve required paths and validate the profile before imports which can
    # initialize external runtimes, load pretrained models or touch CUDA.
    config = load_training_config(args.config)
    cfg = config["training"]
    for name in ("max_updates", "microbatch", "global_batch", "num_workers",
                 "warmup_updates", "checkpoint_every", "probe_every", "log_every"):
        value = getattr(args, name)
        if value is not None:
            cfg[name] = value
    if args.task_filter:
        config["data"]["task_filter"] = args.task_filter
    if args.episodes:
        config["data"]["episodes"] = args.episodes
    cfg["fixed_batch"] = cfg["fixed_batch"] or args.fixed_batch
    cfg["development"] = cfg["development"] or args.development
    if cfg["development"]:
        config["name"] += "-Development"
    config["run_scope"] = "development" if cfg["development"] else "full_c45"
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    accumulation = validate_training_contract(config, world)
    model_path = Path(config["model_config_file"])
    model_config = yaml.safe_load(model_path.read_text(encoding="utf-8"))
    validate_model_recipe(config, model_config)
    configure_tig_imports(config)
    from diffusers.training_utils import EMAModel
    from infrastructure.bpp_tig.data import TIGBPPDataset, collate_bpp
    from infrastructure.bpp_tig.policy import build_policy

    assert torch.cuda.is_available(), "GPU training requires CUDA; use --self-test on CPU"
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    torch.set_num_threads(4)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    model_hash = sha256(model_path)
    config["model_config_file"] = str(model_path.resolve())
    policy = build_policy(model_path, pretrained=cfg["pretrained"]).to(device)
    optimizer = policy.get_optimizer(lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                                     betas=tuple(cfg["betas"]))
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    assert any(math.isclose(lr, cfg["lr"] * .1) for lr in base_lrs), base_lrs
    assert any(math.isclose(lr, cfg["lr"]) for lr in base_lrs), base_lrs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, cfg["warmup_updates"], cfg["scheduler_updates"]))
    ema = EMAModel(policy.parameters(), decay=.9999, inv_gamma=1., power=.75,
                   update_after_step=0, use_ema_warmup=False)
    ema.to(device)
    dataset = TIGBPPDataset(config["data"])
    manifest = dataset.manifest
    validate_c45_manifest(config, manifest)
    source = identity(config)
    resume_signature = signature(config, manifest, model_hash, world, accumulation)
    completed = 0
    checkpoint: dict[str, Any] | None = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        assert checkpoint["resume_signature"] == resume_signature, "Resume configuration/data identity mismatch"
        assert checkpoint["source_identity"]["files"] == source["files"], "Resume adapter/config source changed"
        assert checkpoint["source_identity"]["repositories"] == source["repositories"], "Resume BPP/TIG source changed"
        policy.load_state_dict(checkpoint["model"], strict=True)
        # Native DictOfTensorMixin rebuilds normalizer ParameterDicts on the
        # checkpoint's CPU device. Move again before they normalize CUDA input.
        policy.to(device)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        ema.load_state_dict(checkpoint["ema"])
        ema.to(device)
        completed = int(checkpoint["optimizer_updates"])
        assert completed <= cfg["max_updates"]
    model: nn.Module = LossModule(policy)
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank],
                                        find_unused_parameters=True, broadcast_buffers=False)
    torch.manual_seed(cfg["seed"] + rank)
    np.random.seed(cfg["seed"] + rank)
    random.seed(cfg["seed"] + rank)
    output = args.output_dir.resolve()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        if (output / "metrics.jsonl").exists() and not args.resume:
            raise FileExistsError(f"Existing run requires --resume: {output}")
        write_run_provenance(output, config, source, world, accumulation, resume_signature,
                             resume=bool(args.resume))
        atomic_json(output / "data-manifest.json", manifest)
    sampler = TicketBatchSampler(dataset.group_indices, cfg["microbatch"], rank, world,
                                 cfg["seed"], 0 if cfg["fixed_batch"] else completed * accumulation,
                                 1 if cfg["fixed_batch"] else cfg["max_updates"] * accumulation)
    loader_args: dict[str, Any] = {"dataset": dataset, "batch_sampler": sampler,
                                  "collate_fn": collate_bpp, "pin_memory": True,
                                  "num_workers": cfg["num_workers"]}
    if cfg["num_workers"]:
        loader_args.update(prefetch_factor=2, persistent_workers=True)
    loader = DataLoader(**loader_args)
    iterator = iter(loader)
    # Worker creation must happen before restoring the model-noise RNG.
    if checkpoint is not None:
        restore_rng(checkpoint["rank_rngs"][rank])
        del checkpoint
    fixed: dict[str, Any] | None = None
    probe_tickets = next(iter(TicketBatchSampler(dataset.group_indices, cfg["microbatch"],
                                               rank, world, cfg["seed"], 0, 1)))
    probe_sample = collate_bpp([dataset[ticket] for ticket in probe_tickets])
    source_probe = device_tree({"obs": probe_sample["obs"], "action": probe_sample["action"]}, device)
    initial_probe = probe(policy, source_probe, device)
    if rank == 0:
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"optimizer_updates": completed,
                "source_probe_loss": initial_probe, "probe_tickets": probe_tickets,
                "event": "initial_source_probe"}) + "\n")
    stop = [False]
    def request_stop(signum: int, frame: Any) -> None:
        stop[0] = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    if rank == 0:
        atomic_json(output / "status.json", {"status": "running", "pid": os.getpid(),
                    "optimizer_updates": completed, "world_size": world,
                    "accumulation": accumulation, "global_batch": cfg["global_batch"],
                    "base_lrs": base_lrs, "development": cfg["development"],
                    "source_identity": source, "started_at": datetime.now(timezone.utc).isoformat()})
    policy.train()
    train_started = time.perf_counter()
    initial_update = completed
    last_checkpoint = -1
    while completed < cfg["max_updates"]:
        torch.cuda.synchronize()
        started = time.perf_counter()
        data_seconds = 0.
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=device)
        for microstep in range(accumulation):
            reading = time.perf_counter()
            if fixed is None or not cfg["fixed_batch"]:
                sample = next(iterator)
                batch = device_tree({"obs": sample["obs"], "action": sample["action"]}, device)
                if cfg["fixed_batch"]:
                    fixed = batch
            else:
                batch = fixed
            data_seconds += time.perf_counter() - reading
            context = model.no_sync() if world > 1 and microstep + 1 < accumulation else nullcontext()
            with context:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(batch)
                assert loss.ndim == 0 and torch.isfinite(loss), f"Nonfinite loss at update {completed}"
                loss_sum += loss.detach() / accumulation
                (loss / accumulation).backward()
        grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), cfg["clip_grad_norm"], error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        ema.step(policy.parameters())
        completed += 1
        if world > 1:
            dist.all_reduce(loss_sum, op=dist.ReduceOp.AVG)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        probe_loss = None
        if cfg["probe_every"] and completed % cfg["probe_every"] == 0:
            probe_loss = probe(policy, source_probe, device)
        if completed % cfg["log_every"] == 0 or completed == initial_update + 1 or completed == cfg["max_updates"]:
            row = {"optimizer_updates": completed, "loss": float(loss_sum),
                   "grad_norm": float(grad_norm), "seconds_per_update": seconds,
                   "data_seconds": data_seconds, "samples_per_second": cfg["global_batch"] / seconds,
                   "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
                   "lr_groups": [group["lr"] for group in optimizer.param_groups],
                   "elapsed_seconds": time.perf_counter() - train_started,
                   "source_probe_loss": probe_loss}
            if rank == 0:
                with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                atomic_json(output / "progress.json", row)
        if rank == 0 and (output / "STOP").exists():
            stop[0] = True
        stop_tensor = torch.tensor(int(stop[0]), device=device)
        if world > 1:
            dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
        stopping = bool(stop_tensor)
        if should_checkpoint(completed, cfg, stopping):
            path = output / "checkpoints" / f"update-{completed:08d}.pt"
            save_checkpoint(path, policy, ema, optimizer, scheduler,
                {"config": config, "optimizer_updates": completed,
                 "model_config_file": str(model_path), "model_config_sha256": model_hash,
                 "model_config": policy.model_config,
                 "data_manifest": manifest, "source_identity": source,
                 "resume_signature": resume_signature}, rank, world)
            last_checkpoint = completed
            if rank == 0:
                files = sorted(path.parent.glob("update-*.pt"))
                keep = set(files[-2:])
                keep.update(p for p in files if int(p.stem.split("-")[-1]) in cfg["milestones"])
                for old in files:
                    if old not in keep:
                        old.unlink()
        if stopping:
            break
    if rank == 0:
        atomic_json(output / "status.json", {"status": "complete" if completed == cfg["max_updates"] else "stopped",
                    "optimizer_updates": completed, "checkpoint_update": last_checkpoint,
                    "elapsed_seconds": time.perf_counter() - train_started,
                    "world_size": world, "global_batch": cfg["global_batch"],
                    "development": cfg["development"], "source_identity": source})
    if dist.is_initialized():
        dist.destroy_process_group()


def self_test() -> None:
    """CPU checks for ticket resume and an actual optimizer/EMA update resume."""
    from diffusers.training_utils import EMAModel
    from unittest.mock import patch
    from infrastructure.bpp_tig.configuration import self_test as configuration_self_test
    configuration_self_test()
    checkpoint_config = {"checkpoint_every": 1000, "milestones": [23746], "max_updates": 237460}
    assert should_checkpoint(23746, checkpoint_config, False)
    assert should_checkpoint(24000, checkpoint_config, False)
    assert should_checkpoint(237460, checkpoint_config, False)
    assert should_checkpoint(23747, checkpoint_config, True)
    assert not should_checkpoint(23747, checkpoint_config, False)
    with patch.dict(os.environ, {}, clear=True), patch("torch.cuda.is_available", side_effect=AssertionError(
            "Missing environment must fail before CUDA is queried")):
        try:
            train(argparse.Namespace(config=ROOT / "infrastructure/bpp_tig/configs/c45.yaml",
                                     output_dir=Path("unused-development-output")))
        except ValueError as error:
            assert "TIG_SOURCE_ROOT" in str(error) and "IG_HUMAN_ROOT" in str(error)
        else:
            raise AssertionError("Missing explicit paths accepted by training entry")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fork = root / "fork"
        runtime = root / "runtime"
        source_file = fork / "infrastructure/bpp_tig/train.py"
        source_file.parent.mkdir(parents=True)
        source_file.write_text("# synthetic adapter source\n", encoding="utf-8")
        runtime.mkdir()
        calls: list[list[str]] = []
        def git_output(arguments: list[str], **kwargs: Any) -> str | bytes:
            calls.append(arguments)
            assert arguments[:2] == ["git", "-C"]
            assert arguments[2] in {str(fork), str(runtime)}
            if arguments[3:] == ["rev-parse", "HEAD"]:
                return "synthetic-commit\n"
            assert arguments[3:] == ["diff", "HEAD"]
            return b"synthetic-source-diff"
        with patch("subprocess.check_output", side_effect=git_output):
            source = identity({"tig_source_root": str(runtime)}, fork_root=fork)
        assert set(source["repositories"]) == {"bpp_fork", "tig_runtime"}
        assert source["files"]["infrastructure/bpp_tig/train.py"] == sha256(source_file)
        assert len(calls) == 4 and all("/BPP" not in argument for call in calls for argument in call)
        provenance_config = {
            "name": "synthetic-development", "recipe": "synthetic-recipe",
            "training": {"seed": 0, "max_updates": 2, "scheduler_updates": 2,
                         "global_batch": 4, "microbatch": 2, "development": True},
        }
        run = root / "synthetic-run"
        with patch("subprocess.check_output", return_value=b"synthetic-source-diff"), patch.dict(
                os.environ, {"UNRELATED_PRIVATE_ENVIRONMENT": "must-not-be-recorded"}):
            write_run_provenance(run, provenance_config, source, 2, 1, "synthetic-signature",
                                 resume=False, fork_root=fork, command=["python", "synthetic-test"])
        original_launch = (run / "launch.json").read_bytes()
        original_stamp = (run / "source/sha256.json").read_bytes()
        assert b"must-not-be-recorded" not in original_launch
        assert json.loads(original_launch)["optimizer_update_budget"] == 2
        with patch("subprocess.check_output", side_effect=AssertionError("Resume must preserve provenance")):
            write_run_provenance(run, provenance_config, source, 2, 1, "synthetic-signature",
                                 resume=True, fork_root=fork)
        assert (run / "launch.json").read_bytes() == original_launch
        assert (run / "source/sha256.json").read_bytes() == original_stamp
        try:
            write_run_provenance(run, provenance_config, source, 2, 1, "synthetic-signature",
                                 resume=False, fork_root=fork)
        except FileExistsError:
            pass
        else:
            raise AssertionError("Existing initial provenance overwritten")
        snapshot_file = run / "source/infrastructure/bpp_tig/train.py"
        snapshot_file.write_text("# altered snapshot\n", encoding="utf-8")
        try:
            write_run_provenance(run, provenance_config, source, 2, 1, "synthetic-signature",
                                 resume=True, fork_root=fork)
        except ValueError as error:
            assert "snapshot missing or changed" in str(error)
        else:
            raise AssertionError("Modified source snapshot accepted for resume")
        assert not list(run.rglob("*.incomplete"))
    print("PASS: required environment fails before CUDA; fork/TIG identities reference explicit roots")
    print("PASS: atomic launch/source provenance, unchanged resume metadata and corrupted-snapshot rejection")
    groups = {"a": range(0, 11), "b": range(11, 15)}
    all_batches = list(TicketBatchSampler(groups, 2, 0, 2, 7, 0, 8))
    resumed_batches = list(TicketBatchSampler(groups, 2, 0, 2, 7, 4, 8))
    assert all_batches[4:] == resumed_batches
    other = list(TicketBatchSampler(groups, 2, 1, 2, 7, 0, 8))
    assert all(a[0][1] != b[0][1] for a, b in zip(all_batches, other))
    assert lr_factor(0, 5, 100) == .2 and lr_factor(100, 5, 100) == 0
    torch.manual_seed(4)
    policy = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda n: lr_factor(n, 2, 20))
    ema = EMAModel(policy.parameters(), use_ema_warmup=True)
    def update() -> torch.Tensor:
        optimizer.zero_grad()
        value = policy(torch.randn(2, 3)).square().mean()
        value.backward()
        optimizer.step()
        scheduler.step()
        ema.step(policy.parameters())
        return value.detach()
    update()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoints/state.pt"
        save_checkpoint(path, policy, ema, optimizer, scheduler, {"optimizer_updates": 1}, 0, 1)
        expected_loss = update()
        expected_model = cpu_tree(policy.state_dict())
        expected_ema = cpu_tree(ema.state_dict())
        saved = torch.load(path, weights_only=False)
        policy.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        ema.load_state_dict(saved["ema"])
        restore_rng(saved["rank_rngs"][0])
        actual_loss = update()
        assert torch.equal(actual_loss, expected_loss)
        assert all(torch.equal(policy.state_dict()[k], v) for k, v in expected_model.items())
        assert all(torch.equal(a, b) for a, b in zip(ema.state_dict()["shadow_params"], expected_ema["shadow_params"]))
    print("PASS: counter-based distributed draws and optimizer/scheduler/EMA/RNG exact CPU resume; output (2,2)")


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.self_test:
        self_test()
    else:
        train(arguments)
