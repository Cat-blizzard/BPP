"""Evaluate frozen source-development checkpoints as their archives become ready.

This process never updates policy parameters or controls the training service.
It launches the unchanged full evaluation protocol once per candidate, retains
technical failures, and waits without a CUDA context between checkpoints.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def checkpoint_ready(archive: Path, update: int) -> Path | None:
    """Consume only checkpoints committed by the separate retention process."""
    manifest_file = archive / "manifest.json"
    if not manifest_file.exists():
        return None
    manifest = read_json(manifest_file)
    entry = manifest.get("checkpoints", {}).get(str(update))
    if entry is None:
        return None
    path = Path(entry["archive"])
    if (path.is_symlink() or path.resolve().parent != archive.resolve()
            or path.name != f"update-{update:08d}.pt"
            or entry.get("optimizer_updates") != update):
        raise ValueError("Archive entry does not identify the requested checkpoint")
    identity = entry["archive_identity"]
    stat = path.stat()
    if (not path.is_file() or stat.st_size != identity["bytes"]
            or stat.st_ino != identity["inode"] or stat.st_dev != identity["device"]
            or stat.st_mtime_ns != identity["mtime_ns"]):
        raise ValueError("Archived checkpoint changed after retention committed it")
    return path


def measure_completed(output: Path, protocol: dict[str, Any],
                      protocol_hash: str) -> dict[str, Any] | None:
    """Recompute the source macro SR from the complete, exact episode schedule."""
    identity_file, summary_file = output / "identity.json", output / "summary.json"
    if not identity_file.exists() or not summary_file.exists():
        return None
    identity, summary = read_json(identity_file), read_json(summary_file)
    if summary.get("status") != "completed":
        return None
    if (identity.get("scope") != "source_dev"
            or identity.get("source_dev_protocol_sha256") != protocol_hash
            or identity.get("weights") != "ema_model"):
        raise ValueError("Completed evaluation has another protocol, scope or weight type")
    expected = {(env, episode) for env in protocol["env_ids"]
                for episode in range(protocol["num_episodes"])}
    records = [read_json(path) for path in sorted(output.glob("*/*.json"))
               if path.stem.isdigit() and len(path.stem) == 6]
    actual = {(row["env_id"], row["episode_id"]) for row in records}
    if actual != expected or len(records) != len(expected):
        raise ValueError("Completed evaluation has missing, duplicate or extra episodes")
    task_level_sr: dict[str, float] = {}
    for row in records:
        if (row.get("status") != "completed"
                or row.get("run_identity") != identity.get("run_identity")
                or row["scene_seed"] != protocol["scene_seeds"][row["episode_id"]]
                or row.get("weight_key") != "ema_model"
                or not isinstance(row.get("success_at_end"), bool)):
            raise ValueError("Episode identity, seed, EMA weights or metric differs")
    for env in protocol["env_ids"]:
        values = [float(row["success_at_end"]) for row in records if row["env_id"] == env]
        task_level_sr[env] = sum(values) / len(values)
    successes = sum(int(row["success_at_end"]) for row in records)
    # Every group has exactly three assigned episodes. This equal-weight macro
    # mean is successes/N; compute it once to avoid group-order rounding ties.
    macro_sr = successes / len(records)
    assert math.isfinite(macro_sr) and 0 <= macro_sr <= 1
    return {"optimizer_updates": identity["optimizer_updates"],
            "run_identity": identity["run_identity"],
            "checkpoint_sha256": identity["checkpoint_sha256"],
            "weights": identity["weights"], "episodes": len(records),
            "successes": successes,
            "macro_success_at_end": macro_sr, "task_level_success_at_end": task_level_sr,
            "summary": str(summary_file.resolve())}


def write_curve(root: Path, protocol: dict[str, Any], protocol_hash: str) -> dict[str, Any]:
    results = []
    for update in protocol["checkpoint_candidates"]:
        result = measure_completed(root / f"update-{update:08d}", protocol, protocol_hash)
        if result is not None:
            if result["optimizer_updates"] != update:
                raise ValueError("Evaluation directory and checkpoint update differ")
            results.append(result)
    best = min(results, key=lambda row: (-row["successes"],
                                        row["optimizer_updates"])) if results else None
    value = {"scope": "source_dev", "protocol_id": protocol["protocol_id"],
             "protocol_sha256": protocol_hash, "updated_at_utc": now_utc(),
             "candidate_updates": protocol["checkpoint_candidates"],
             "completed_candidates": len(results), "results": results,
             "development_selected": best,
             "fixed_budget_primary_update": protocol["selection"]["fixed_budget_primary_update"],
             "selection_is_final": len(results) == len(protocol["checkpoint_candidates"]),
             "sub_sr": "last_reward (native dense-reward proxy)", "automatic_training_stop": False,
             "interpretation": "Source-task development only; small SR differences are not significance evidence."
             }
    atomic_json(root / "learning_curve.json", value)
    return value


def gpu_available(uuid: str) -> tuple[bool, str]:
    """Wait for an otherwise idle evaluation GPU; never select training GPUs."""
    if not uuid.startswith("GPU-"):
        raise ValueError("Specify a physical evaluation GPU UUID")
    query = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                            "--format=csv,noheader,nounits"], check=True,
                           capture_output=True, text=True)
    occupied = [line for line in query.stdout.splitlines() if line.split(",")[0].strip() == uuid]
    if occupied:
        return False, "evaluation GPU has another compute process"
    memory = subprocess.run(["nvidia-smi", f"--id={uuid}", "--query-gpu=memory.free",
                             "--format=csv,noheader,nounits"], check=True,
                            capture_output=True, text=True)
    if int(memory.stdout.strip()) < 24000:
        return False, "evaluation GPU has less than 24000 MiB free"
    return True, "idle"


def run(args: argparse.Namespace) -> None:
    protocol = read_json(args.protocol)
    if (protocol.get("schema_version") != "bpp-tig-source-dev-v2" or protocol.get("template")
            or protocol.get("model_recipe") != "bpp-tig-split-clip-frozen-prompt-v2"
            or protocol.get("scope") != "source_dev" or protocol.get("weights") != "ema_model"
            or protocol.get("total_rollouts") != 48
            or protocol.get("selection", {}).get("automatic_early_stopping") is not False):
        raise ValueError("Expected the frozen 48-rollout EMA source-development protocol")
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_hash = sha256(args.protocol)
    evaluator = Path(__file__).with_name("evaluate.py")
    source_hashes = {name: sha256(Path(__file__).with_name(name)) for name in
                     ["evaluate.py", "data.py", "policy.py", "configuration.py", "source_dev_runner.py"]}
    binding = {"protocol_sha256": protocol_hash, "source_sha256": source_hashes,
               "archive": str(args.archive.resolve()), "gpu_uuid": args.gpu_uuid}
    binding_file = args.output / "runner-identity.json"
    if binding_file.exists() and read_json(binding_file) != binding:
        raise ValueError("Runner output belongs to another protocol, code, archive or GPU")
    atomic_json(binding_file, binding)
    status_file = args.output / "runner-status.json"
    try:
        while True:
            if sha256(args.protocol) != protocol_hash or any(
                    sha256(Path(__file__).with_name(name)) != value
                    for name, value in source_hashes.items()):
                raise ValueError("Frozen evaluation sources changed during the run")
            curve = write_curve(args.output, protocol, protocol_hash)
            completed = {row["optimizer_updates"] for row in curve["results"]}
            if len(completed) == len(protocol["checkpoint_candidates"]):
                atomic_json(status_file, {"status": "completed", "updated_at_utc": now_utc(),
                                          "completed_updates": sorted(completed)})
                return
            if (args.output / "STOP").exists():
                atomic_json(status_file, {"status": "stopped", "updated_at_utc": now_utc()})
                return
            update = next(value for value in protocol["checkpoint_candidates"] if value not in completed)
            checkpoint = checkpoint_ready(args.archive, update)
            reason = "waiting for retained checkpoint"
            available = False
            if checkpoint is not None:
                available, reason = gpu_available(args.gpu_uuid)
            if checkpoint is None or not available:
                atomic_json(status_file, {"status": "waiting", "next_update": update,
                                          "reason": reason, "updated_at_utc": now_utc(),
                                          "completed_updates": sorted(completed)})
                time.sleep(args.poll_seconds)
                continue
            output = args.output / f"update-{update:08d}"
            output.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(evaluator), "--checkpoint", str(checkpoint),
                       "--output", str(output), "--source-dev-protocol", str(args.protocol),
                       "--device", "cuda:0", "--video"]
            atomic_json(status_file, {"status": "evaluating", "optimizer_updates": update,
                                      "command": command, "updated_at_utc": now_utc(),
                                      "completed_updates": sorted(completed)})
            print(json.dumps({"event": "evaluation_started", "optimizer_updates": update,
                              "at_utc": now_utc()}, ensure_ascii=False), flush=True)
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu_uuid)
            with (output / "evaluate.log").open("a", encoding="utf-8") as log:
                result = subprocess.run(command, env=environment, stdout=log,
                                        stderr=subprocess.STDOUT, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"Evaluation {update} failed with exit {result.returncode}; see {output / 'evaluate.log'}")
            measurement = measure_completed(output, protocol, protocol_hash)
            if measurement is None:
                raise RuntimeError("Evaluator exited without the complete frozen episode schedule")
            print(json.dumps({"event": "evaluation_completed", **measurement,
                              "at_utc": now_utc()}, ensure_ascii=False), flush=True)
    except BaseException as error:
        atomic_json(status_file, {"status": "failed", "updated_at_utc": now_utc(),
                                  "error": f"{type(error).__name__}: {error}"})
        raise


def self_test() -> None:
    """CPU-only tests of committed archives, exact schedules and curve selection."""
    with tempfile.TemporaryDirectory(prefix="bpp-source-dev-runner-") as directory:
        root = Path(directory)
        archive = root / "checkpoints"
        archive.mkdir()
        assert checkpoint_ready(archive, 18) is None
        checkpoint = archive / "update-00000018.pt"
        checkpoint.write_bytes(b"committed-test-fixture")
        stat = checkpoint.stat()
        atomic_json(archive / "manifest.json", {"checkpoints": {"18": {
            "archive": str(checkpoint), "optimizer_updates": 18,
            "archive_identity": {"device": stat.st_dev, "inode": stat.st_ino,
                                 "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}}}})
        assert checkpoint_ready(archive, 18) == checkpoint
        checkpoint.write_bytes(b"changed")
        try:
            checkpoint_ready(archive, 18)
            raise AssertionError("Changed archive accepted")
        except ValueError:
            pass
        protocol = {"protocol_id": "fixture",
                    "env_ids": [f"L{index % 4}_A{index // 4}" for index in range(16)],
                    "num_episodes": 3, "scene_seeds": [10000, 10001, 10002],
                    "checkpoint_candidates": [18, 40], "selection": {"fixed_budget_primary_update": 40}}
        for update in protocol["checkpoint_candidates"]:
            output = root / f"update-{update:08d}"
            atomic_json(output / "identity.json", {"run_identity": str(update),
                "optimizer_updates": update, "checkpoint_sha256": "fixture", "weights": "ema_model",
                "scope": "source_dev", "source_dev_protocol_sha256": "fixture"})
            atomic_json(output / "summary.json", {"status": "completed"})
            counts = [0] * 13 + ([2, 2, 3] if update == 18 else [1, 3, 3])
            for index, env in enumerate(protocol["env_ids"]):
                for episode in range(3):
                    atomic_json(output / env / f"{episode:06d}.json", {
                        "status": "completed", "run_identity": str(update), "env_id": env,
                        "episode_id": episode, "scene_seed": 10000 + episode,
                        "weight_key": "ema_model", "success_at_end": episode < counts[index]})
        curve = write_curve(root, protocol, "fixture")
        assert curve["selection_is_final"] and curve["development_selected"]["optimizer_updates"] == 18
        assert curve["results"][0]["macro_success_at_end"] == 7 / 48
        assert curve["results"][0]["macro_success_at_end"] == curve["results"][1]["macro_success_at_end"]
        missing = root / "update-00000018" / "L0_A0" / "000000.json"
        missing.unlink()
        try:
            write_curve(root, protocol, "fixture")
            raise AssertionError("Incomplete schedule accepted as completed")
        except ValueError:
            pass
    print("BPP_SOURCE_DEV_RUNNER_SELF_TEST_PASS: archive identity; exact episode schedule; macro SR; earliest tie; no CUDA")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not all([args.protocol, args.archive, args.output, args.gpu_uuid]) or not 1 <= args.poll_seconds <= 60:
        parser.error("Supply protocol/archive/output/GPU UUID and a 1–60 second polling interval")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".runner.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == "__main__":
    main()
