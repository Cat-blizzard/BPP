"""Retain committed checkpoints without changing a running trainer.

Contract: this CPU-only utility reads this run's atomic ``latest.json`` commit
watermark. It never reads tensor payloads, updates training configuration, or
deletes source checkpoints. Target archives are immutable hard links; EXDEV alone
permits copying into a temporary file followed by an exclusive final publication.
The run is bound to launch/source-manifest hashes and its checkpoint resume
signature. ZIP metadata and pickle opcodes are inspected without unpickling.

The trainer publishes ``.pt.incomplete -> .pt -> latest.json`` and then prunes.
Consequently a named checkpoint at or below the committed watermark is complete.
An opened source descriptor, pre/post link stat comparisons, and metadata checks
catch deletion/replacement races. A failed preserve exits visibly rather than
claiming success. Use ``--self-test`` for independent tempfile checks, or
``--config PATH --once``; omit ``--once`` to watch at <=30-second intervals.
"""
from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickletools
import shutil
import stat
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, BinaryIO
import zipfile


class PreservationError(RuntimeError):
    """Invalid identity, checkpoint, or publication state."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_path(path: Path, *, must_exist: bool = True) -> Path:
    """Require absolute paths with no symlink component or parent traversal."""
    if not path.is_absolute() or ".." in path.parts:
        raise PreservationError(f"Unsafe path: {path}")
    cursor = Path(path.anchor)
    for part in path.parts[1:]:
        cursor = cursor / part
        try:
            info = cursor.lstat()
        except FileNotFoundError:
            if must_exist:
                raise PreservationError(f"Missing path: {cursor}") from None
            continue
        if stat.S_ISLNK(info.st_mode):
            raise PreservationError(f"Symlink forbidden: {cursor}")
    return path


def open_regular(path: Path) -> BinaryIO:
    safe_path(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    stream = os.fdopen(descriptor, "rb")
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        stream.close()
        raise PreservationError(f"Not a regular file: {path}")
    return stream


def read_json(path: Path) -> dict[str, Any]:
    with open_regular(path) as stream:
        value = json.loads(stream.read(4 * 1024 * 1024 + 1))
    if not isinstance(value, dict):
        raise PreservationError(f"JSON object required: {path}")
    return value


def digest(path: Path) -> str:
    with open_regular(path) as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    safe_path(path, must_exist=False)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def identity(info: os.stat_result) -> dict[str, int]:
    return {"device": info.st_dev, "inode": info.st_ino, "bytes": info.st_size,
            "mtime_ns": info.st_mtime_ns}


def pickle_field(data: bytes, field: str) -> str | int:
    """Read a unique scalar field using opcode inspection, never pickle execution."""
    operations = list(pickletools.genops(data))
    values: list[str | int] = []
    memo = {"BINPUT", "LONG_BINPUT", "PUT", "MEMOIZE"}
    scalar = {"BINUNICODE", "SHORT_BINUNICODE", "UNICODE", "BININT", "BININT1",
              "BININT2", "INT", "LONG", "LONG1", "LONG4"}
    for index, (opcode, argument, _) in enumerate(operations):
        if opcode.name in {"BINUNICODE", "SHORT_BINUNICODE", "UNICODE"} and argument == field:
            index += 1
            while index < len(operations) and operations[index][0].name in memo:
                index += 1
            if index == len(operations) or operations[index][0].name not in scalar:
                raise PreservationError(f"Unsupported checkpoint scalar: {field}")
            value = operations[index][1]
            if not isinstance(value, (str, int)):
                raise PreservationError(f"Invalid checkpoint scalar: {field}")
            values.append(value)
    if len(values) != 1:
        raise PreservationError(f"Expected one checkpoint field {field}, found {len(values)}")
    return values[0]


def validate_checkpoint(stream: BinaryIO, update: int, config: dict[str, Any]) -> None:
    """Validate complete Torch ZIP layout and small metadata, without tensor I/O."""
    if os.fstat(stream.fileno()).st_size < config["minimum_checkpoint_bytes"]:
        raise PreservationError("Checkpoint is smaller than this run's lower bound")
    try:
        with zipfile.ZipFile(stream) as archive:
            members = archive.infolist()
            if len({item.filename for item in members}) != len(members):
                raise PreservationError("Duplicate ZIP members")
            metadata = [item for item in members if item.filename.endswith("/data.pkl")]
            if len(metadata) != 1 or metadata[0].file_size > 4 * 1024 * 1024:
                raise PreservationError("Missing/oversized checkpoint metadata")
            prefix = metadata[0].filename.removesuffix("data.pkl")
            names = {item.filename for item in members}
            if prefix + "version" not in names or prefix + "byteorder" not in names:
                raise PreservationError("Incomplete Torch ZIP layout")
            if not any(name.startswith(prefix + "data/") for name in names):
                raise PreservationError("Missing tensor storage members")
            if any(item.flag_bits & 1 or item.compress_type != zipfile.ZIP_STORED for item in members):
                raise PreservationError("Unexpected encrypted/compressed checkpoint")
            data = archive.read(metadata[0])
    except (zipfile.BadZipFile, EOFError, OSError) as error:
        raise PreservationError(f"Incomplete/invalid checkpoint ZIP: {error}") from error
    if pickle_field(data, "optimizer_updates") != update:
        raise PreservationError("Checkpoint optimizer_updates differs from filename")
    if pickle_field(data, "resume_signature") != config["resume_signature"]:
        raise PreservationError("Checkpoint belongs to a different training run")
    strings = {argument for opcode, argument, _ in pickletools.genops(data)
               if opcode.name in {"BINUNICODE", "SHORT_BINUNICODE", "UNICODE"}}
    if not {"model", "ema_model", "ema", "optimizer", "scheduler", "rank_rngs"}.issubset(strings):
        raise PreservationError("Checkpoint is missing full training/EMA state")


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    config = read_json(path)
    if config.get("schema_version") not in (1, 2):
        raise PreservationError("Unsupported retention schema")
    run = safe_path(Path(config["run_dir"]))
    source = safe_path(run / "checkpoints")
    archive = safe_path(Path(config["archive_dir"]), must_exist=False)
    if archive == source or archive.is_relative_to(source) or source.is_relative_to(archive):
        raise PreservationError("Archive must be outside training checkpoint directory")
    targets = config["target_updates"]
    if not targets or targets != sorted(set(targets)) or any(type(step) is not int or step <= 0 for step in targets):
        raise PreservationError("Target updates must be unique ascending positive integers")
    if not 0 < config["poll_seconds"] <= 30 or config["minimum_checkpoint_bytes"] <= 0:
        raise PreservationError("Invalid polling interval/checkpoint lower bound")
    for relative, expected in config["run_identity_sha256"].items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise PreservationError("Unsafe run identity path")
        if digest(run / relative) != expected:
            raise PreservationError(f"Run identity changed: {relative}")
    if (run / "source/sha256.json").exists():
        for relative, expected in read_json(run / "source/sha256.json").items():
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise PreservationError("Unsafe source snapshot identity path")
            if digest(run / "source" / relative) != expected:
                raise PreservationError(f"Frozen source file changed: {relative}")
    if config["schema_version"] == 2:
        import yaml
        run_config = yaml.safe_load((run / "config.yaml").read_text())
        budget = run_config["training"]["scheduler_updates"]
        if config.get("optimizer_update_budget") != budget:
            raise PreservationError("Retention budget differs from real training configuration")
    else:
        budget = read_json(run / "launch.json")["optimizer_update_budget"]
    if any(step > budget for step in targets):
        raise PreservationError("Retention target exceeds run budget")
    return config, digest(path)


class Preserver:
    def __init__(self, config: dict[str, Any], config_hash: str) -> None:
        self.config = config
        self.config_hash = config_hash
        self.run = Path(config["run_dir"])
        self.source = self.run / "checkpoints"
        self.archive = Path(config["archive_dir"])
        safe_path(self.archive, must_exist=False)
        self.archive.mkdir(parents=True, exist_ok=True)
        safe_path(self.archive)
        lock_path = self.archive / ".preserve.lock"
        safe_path(lock_path, must_exist=False)
        self.lock = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(self.lock)
            raise PreservationError("Another checkpoint preserver holds the archive lock") from None
        self.manifest_path = self.archive / "manifest.json"
        if self.manifest_path.exists():
            self.manifest = read_json(self.manifest_path)
            if self.manifest.get("config_sha256") != config_hash:
                self.close()
                raise PreservationError("Archive retention configuration differs")
        else:
            self.manifest = {"schema_version": 1, "config_sha256": config_hash,
                             "run_dir": str(self.run), "resume_signature": config["resume_signature"],
                             "created_at_utc": now(), "checkpoints": {}}
        for key, entry in self.manifest["checkpoints"].items():
            path = self.archive / f"update-{int(key):08d}.pt"
            with open_regular(path) as stream:
                if identity(os.fstat(stream.fileno())) != entry["archive_identity"]:
                    raise PreservationError(f"Archived checkpoint changed: {path}")
                validate_checkpoint(stream, int(key), config)

    def close(self) -> None:
        if getattr(self, "lock", None) is not None:
            os.close(self.lock)
            self.lock = None

    def committed(self) -> tuple[int, dict[str, Any]]:
        latest = read_json(self.source / "latest.json")
        update = latest["optimizer_updates"]
        if type(update) is not int or update <= 0:
            raise PreservationError("Invalid committed update")
        expected = self.source / f"update-{update:08d}.pt"
        if latest.get("checkpoint") != str(expected) or type(latest.get("bytes")) is not int or latest["bytes"] <= 0:
            raise PreservationError("Invalid latest.json path/size")
        # latest may advance after this read; its captured watermark remains valid.
        return update, latest

    def preserve(self, update: int, watermark: int, latest: dict[str, Any]) -> None:
        key = str(update)
        source = self.source / f"update-{update:08d}.pt"
        target = self.archive / source.name
        if key in self.manifest["checkpoints"]:
            with open_regular(target) as stream:
                if identity(os.fstat(stream.fileno())) != self.manifest["checkpoints"][key]["archive_identity"]:
                    raise PreservationError(f"Archived checkpoint changed: {target}")
            return
        if target.exists() or target.is_symlink():
            # Recover a crash after final publication but before manifest commit.
            with open_regular(target) as stream:
                validate_checkpoint(stream, update, self.config)
                target_identity = identity(os.fstat(stream.fileno()))
            source_identity = None
            method = "recovered_archive"
        else:
            with open_regular(source) as stream:
                before = identity(os.fstat(stream.fileno()))
                if update == watermark and before["bytes"] != latest["bytes"]:
                    raise PreservationError("Committed checkpoint size differs from latest.json")
                validate_checkpoint(stream, update, self.config)
                try:
                    os.link(source, target, follow_symlinks=False)
                    method = "hardlink"
                except OSError as error:
                    if error.errno != errno.EXDEV:
                        raise
                    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=self.archive)
                    try:
                        stream.seek(0)
                        with os.fdopen(descriptor, "wb") as destination:
                            shutil.copyfileobj(stream, destination, 8 * 1024 * 1024)
                            destination.flush()
                            os.fsync(destination.fileno())
                        # Exclusive final publication; never replace an archive.
                        os.link(temporary, target, follow_symlinks=False)
                        method = "cross_filesystem_copy"
                    finally:
                        Path(temporary).unlink(missing_ok=True)
                with open_regular(target) as archived:
                    target_identity = identity(os.fstat(archived.fileno()))
                after = identity(os.fstat(stream.fileno()))
                if before != after or (method == "hardlink" and before != target_identity):
                    raise PreservationError("Checkpoint replaced/changed during preservation")
                if before["bytes"] != target_identity["bytes"]:
                    raise PreservationError("Archive size mismatch")
                source_identity = before
        entry = {"optimizer_updates": update, "source": str(source), "archive": str(target),
                 "method": method, "source_identity": source_identity,
                 "archive_identity": target_identity, "committed_watermark": watermark,
                 "preserved_at_utc": now(), "tensor_sha256": "not_computed; evaluation verifies identity"}
        self.manifest["checkpoints"][key] = entry
        self.manifest["updated_at_utc"] = now()
        atomic_json(self.manifest_path, self.manifest)
        print(json.dumps({"event": "checkpoint_preserved", **entry}), flush=True)

    def tick(self) -> bool:
        watermark, latest = self.committed()
        for update in self.config["target_updates"]:
            if update <= watermark:
                self.preserve(update, watermark, latest)
        retained = sorted(int(key) for key in self.manifest["checkpoints"])
        done = retained == self.config["target_updates"]
        atomic_json(self.archive / "status.json", {"status": "complete" if done else "watching",
                    "updated_at_utc": now(), "committed_watermark": watermark,
                    "retained_updates": retained, "pending_updates": [step for step in self.config["target_updates"] if step not in retained],
                    "config_sha256": self.config_hash, "run_dir": str(self.run)})
        return done


def self_test() -> None:
    """Exercise publication, retention, run binding and path rejection on CPU."""
    import pickle
    with tempfile.TemporaryDirectory(prefix="bpp-retention-test-") as directory:
        base = Path(directory)
        run = base / "run"
        source = run / "checkpoints"
        source.mkdir(parents=True)
        (run / "launch.json").write_text(json.dumps({"optimizer_update_budget": 10}))
        (run / "identity.json").write_text('{"snapshot": "test"}')
        config = {"schema_version": 1, "run_dir": str(run), "archive_dir": str(base / "archive"),
                  "target_updates": [1, 2], "poll_seconds": .01, "minimum_checkpoint_bytes": 1,
                  "resume_signature": "test-signature", "run_identity_sha256": {
                      "launch.json": digest(run / "launch.json"), "identity.json": digest(run / "identity.json")}}
        config_path = base / "config.json"
        config_path.write_text(json.dumps(config))
        loaded, config_hash = load_config(config_path)

        def checkpoint(update: int, signature: str = "test-signature", *, publish: bool = True) -> Path:
            path = source / f"update-{update:08d}.pt"
            temporary = path.with_suffix(".pt.incomplete")
            with zipfile.ZipFile(temporary, "w") as archive:
                archive.writestr("x/data.pkl", pickle.dumps({"optimizer_updates": update,
                    "resume_signature": signature, "model": {}, "ema_model": {}, "ema": {},
                    "optimizer": {}, "scheduler": {}, "rank_rngs": []}))
                archive.writestr("x/version", "3\n")
                archive.writestr("x/byteorder", "little")
                archive.writestr("x/data/0", b"dummy tensor bytes")
            if publish:
                temporary.rename(path)
                atomic_json(source / "latest.json", {"checkpoint": str(path), "optimizer_updates": update, "bytes": path.stat().st_size})
            return path

        first = checkpoint(1)
        preserver = Preserver(loaded, config_hash)
        assert preserver.tick() is False
        archived = base / "archive" / first.name
        assert archived.stat().st_ino == first.stat().st_ino
        original = identity(archived.stat())
        first.unlink()
        assert archived.read_bytes()
        checkpoint(2, publish=False)
        assert preserver.tick() is False  # .incomplete is never considered committed.
        assert not (base / "archive/update-00000002.pt").exists()
        second = checkpoint(2)
        assert preserver.tick() is True
        assert identity(archived.stat()) == original
        preserver.close()
        restored = Preserver(loaded, config_hash)
        assert restored.tick() is True
        assert identity(archived.stat()) == original
        restored.close()

        import yaml
        (run / "config.yaml").write_text(yaml.safe_dump({"recipe": "bpp-tig-split-clip-frozen-prompt-v2",
            "training": {"scheduler_updates": 10}}))
        protocol_file = base / "protocol.json"
        protocol_file.write_text(json.dumps({"schema_version": "bpp-tig-source-dev-v2",
            "model_recipe": "bpp-tig-split-clip-frozen-prompt-v2", "checkpoint_candidates": [1, 2],
            "selection": {"fixed_budget_primary_update": 10}}))
        initialized_config = base / "initialized.json"
        initialize_config(run, base / "archive-v2", protocol_file, initialized_config)
        generated, _ = load_config(initialized_config)
        assert generated["schema_version"] == 2 and generated["optimizer_update_budget"] == 10
        assert generated["target_updates"] == [1, 2]

        def rejects(action: Any) -> None:
            try:
                action()
            except (PreservationError, FileNotFoundError):
                return
            raise AssertionError("Expected preservation refusal")

        config["archive_dir"] = str(base / "wrong-archive")
        config["target_updates"] = [2]
        checkpoint(2, signature="wrong-run")
        wrong = Preserver(config, "different-hash")
        rejects(wrong.tick)
        wrong.close()
        second.write_bytes(b"unfinished file")
        with open_regular(second) as stream:
            rejects(lambda: validate_checkpoint(stream, 2, config))
        alias = base / "alias"
        alias.symlink_to(source, target_is_directory=True)
        rejects(lambda: safe_path(alias / "latest.json"))
        archived.unlink()
        archived.symlink_to(second)
        rejects(lambda: Preserver(loaded, config_hash))
        (run / "identity.json").write_text("changed")
        rejects(lambda: load_config(config_path))
        print(json.dumps({"self_test": "passed", "checks": ["atomic_commit", "partial_ignored", "hardlink",
                    "source_delete_survival", "restart_idempotent", "wrong_run_rejected", "truncated_rejected",
                    "symlink_rejected", "run_identity_bound", "actual_run_v2_initializer"]}))



def initialize_config(run: Path, archive: Path, protocol_file: Path, output: Path) -> None:
    """Bind a portable retention configuration to an actually committed run.

    Run this after the first checkpoint exists. No tensor payload is loaded,
    and no source training/checkpoint files are modified or deleted.
    """
    run, archive, output = run.absolute(), archive.absolute(), output.absolute()
    safe_path(run)
    protocol = read_json(protocol_file.absolute())
    if (protocol.get("schema_version") != "bpp-tig-source-dev-v2"
            or protocol.get("template") or protocol.get("model_recipe") != "bpp-tig-split-clip-frozen-prompt-v2"):
        raise PreservationError("Supply a frozen v2 protocol, not an unbound template")
    targets = protocol["checkpoint_candidates"]
    committed = read_json(run / "checkpoints/latest.json")
    checkpoint = Path(committed["checkpoint"])
    if checkpoint.parent.resolve() != (run / "checkpoints").resolve():
        raise PreservationError("Committed checkpoint does not belong to this run")
    with open_regular(checkpoint) as stream:
        with zipfile.ZipFile(stream) as archive_zip:
            names = [item for item in archive_zip.infolist() if item.filename.endswith("/data.pkl")]
            if len(names) != 1 or names[0].file_size > 4 * 1024 * 1024:
                raise PreservationError("Invalid checkpoint metadata")
            signature = pickle_field(archive_zip.read(names[0]), "resume_signature")
        stream.seek(0)
        import yaml
        run_config = yaml.safe_load((run / "config.yaml").read_text())
        budget = run_config["training"]["scheduler_updates"]
        if (run_config.get("recipe") != protocol["model_recipe"]
                or budget != protocol["selection"]["fixed_budget_primary_update"]):
            raise PreservationError("Run recipe/budget differs from source-development protocol")
        config = {"schema_version": 2, "run_dir": str(run), "archive_dir": str(archive),
            "optimizer_update_budget": budget,
            "target_updates": targets, "poll_seconds": 20,
            "minimum_checkpoint_bytes": max(1, checkpoint.stat().st_size // 2),
            "resume_signature": signature,
            "run_identity_sha256": {name: digest(run / name) for name in
                ("config.yaml", "launch.json", "source/sha256.json") if (run / name).is_file()},
            "source_dev_protocol_sha256": digest(protocol_file.absolute())}
        validate_checkpoint(stream, committed["optimizer_updates"], config)
    if output.exists():
        raise FileExistsError("Retention identity already exists; do not overwrite it")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, config)
    try:
        load_config(output)
    except BaseException:
        output.unlink()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--initialize-run", type=Path)
    parser.add_argument("--archive-dir", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--write-config", type=Path)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.initialize_run:
        if not all((args.archive_dir, args.protocol, args.write_config)):
            parser.error("Initialization requires --archive-dir --protocol --write-config")
        initialize_config(args.initialize_run, args.archive_dir, args.protocol, args.write_config)
        print("RETENTION_CONFIG_BOUND:", args.write_config)
        return
    if args.config is None:
        parser.error("--config or --self-test is required")
    config, config_hash = load_config(args.config.absolute())
    preserver = Preserver(config, config_hash)
    try:
        while True:
            done = preserver.tick()
            if done or args.once:
                return
            time.sleep(config["poll_seconds"])
    except Exception as error:
        atomic_json(preserver.archive / "status.json", {"status": "failed", "updated_at_utc": now(),
                    "error": f"{type(error).__name__}: {error}", "config_sha256": config_hash,
                    "retained_updates": sorted(int(key) for key in preserver.manifest["checkpoints"])})
        raise
    finally:
        preserver.close()


if __name__ == "__main__":
    main()
