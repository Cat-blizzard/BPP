"""TIG LeRobot RGB/qpos adapter for video-only Behavior Prompting.

Contract: each anchor is a robot frame; query history stays in that episode,
future absolute joint commands repeat at its end, and the human prompt is from
the mapped task rather than a synchronized robot episode. RGB is CHW float32
in [0, 1]. qpos18/action16 use the runtime TIG q01/q99 normalizer exactly once;
the BPP policy must therefore use identity low-dimensional normalizers.

Only decoded uint8 pixels and numerical trajectory arrays are cached. Visual
features are never cached. Runtime imports follow the caller's PYTHONPATH.
"""

from __future__ import annotations

import ast
import bisect
import hashlib
import importlib
import inspect
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


STATE_KEY = "observation.qpos_gripper_states"
ACTION_KEY = "action.qpos_gripper_actions"
RGB_KEY = "observation.images.zed2i"
UNSEEN_TASKS = frozenset({
    "TwoRobotPickRemoteControl-v1", "TwoRobotScanMilkBox-v1",
    "TwoRobotPourKettle-v1", "TwoRobotPickFood-v1", "TwoRobotFoldBox-v1",
})


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_seed(*parts: object) -> int:
    value = json.dumps(parts, separators=(",", ":"), default=str).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


def _episode_ids(value: Any, available: Sequence[int]) -> list[int]:
    """Parse explicit half-open ranges without eval or implicit split fallback."""
    if value is None:
        return sorted(int(i) for i in available)
    if isinstance(value, str):
        if not value.strip():
            return []
        if ":" in value:
            start, stop = map(int, value.split(":"))
            return list(range(start, stop))
        value = ast.literal_eval(value)
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple)):
        return [int(i) for i in value]
    raise ValueError(f"Unsupported episode specification: {value!r}")


def window_indices(length: int, anchor: int, obs_horizon: int,
                   pred_horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Return within-episode indices, repeating the first/last frame as needed."""
    assert length > 0 and 0 <= anchor < length
    assert obs_horizon > 0 and pred_horizon > 0
    history = np.maximum(np.arange(anchor - obs_horizon + 1, anchor + 1), 0)
    future = np.minimum(np.arange(anchor, anchor + pred_horizon), length - 1)
    assert history.shape == (obs_horizon,) and future.shape == (pred_horizon,)
    assert (history >= 0).all() and (future < length).all()
    return history.astype(np.int64), future.astype(np.int64)


def _metadata_rows(root: Path) -> list[dict[str, Any]]:
    columns = ["episode_index", "length", "dataset_from_index", "dataset_to_index",
               "data/chunk_index", "data/file_index",
               f"videos/{RGB_KEY}/chunk_index", f"videos/{RGB_KEY}/file_index",
               f"videos/{RGB_KEY}/from_timestamp"]
    rows: list[dict[str, Any]] = []
    files = sorted((root / "meta/episodes").glob("chunk-*/file-*.parquet"))
    assert files, f"No episode metadata in {root}"
    for path in files:
        table = pq.ParquetFile(path)
        present = [key for key in columns if key in table.schema_arrow.names]
        rows.extend(table.read(columns=present).to_pylist())
    result = sorted(rows, key=lambda row: int(row["episode_index"]))
    assert len({int(row["episode_index"]) for row in result}) == len(result)
    return result


@dataclass(frozen=True)
class _Repository:
    repo_id: str
    root: Path
    info: dict[str, Any]
    episodes: dict[int, dict[str, Any]]


@dataclass(frozen=True)
class _AnchorEpisode:
    repository_index: int
    episode_index: int
    length: int
    global_start: int


class _BoundedDecoders:
    """Bound official cache, explicitly using exact rather than approximate seek.

    TIG 3193957e supplies invalidation and timestamp retry, but its decoder
    factory still requests approximate seek. Override only the factory locally;
    retain official timestamp verification/retry and never change tolerance.
    """

    def __init__(self, runtime_module: Any, capacity: int) -> None:
        class ExactCache(runtime_module.VideoDecoderCache):
            def _create_decoder(self, video_path: str) -> tuple[Any, Any]:
                from torchcodec.decoders import VideoDecoder
                try:
                    return VideoDecoder(str(video_path), seek_mode="exact"), None
                except Exception as error:
                    if not self._is_source_type_error(error):
                        raise
                    import fsspec
                    handle = fsspec.open(str(video_path)).open()
                    try:
                        return VideoDecoder(handle, seek_mode="exact"), handle
                    except BaseException:
                        handle.close()
                        raise
        self.base = ExactCache()
        self.capacity = max(1, int(capacity))
        self.order: OrderedDict[str, None] = OrderedDict()

    def get_decoder(self, video_path: str) -> Any:
        path = str(video_path)
        decoder = self.base.get_decoder(path)
        self.order[path] = None
        self.order.move_to_end(path)
        while len(self.order) > self.capacity:
            expired, _ = self.order.popitem(last=False)
            self.base.invalidate(expired)
        return decoder

    def invalidate(self, video_path: str) -> None:
        self.order.pop(str(video_path), None)
        self.base.invalidate(str(video_path))

    def _create_decoder(self, video_path: str) -> Any:
        return self.base._create_decoder(str(video_path))

    def clear(self) -> None:
        self.base.clear()
        self.order.clear()


class TIGBPPDataset(Dataset):
    """Metadata-first TIG source dataset with reproducible video-only prompts.

    ``group_indices`` maps each task-level repo to a cheap contiguous range.
    Uniform group sampling balances both semantic tasks and their four levels.
    ``__getitem__((anchor_index, draw_seed))`` varies the prompt reproducibly;
    a plain integer uses draw_seed=0. Save sampler global draw position for exact
    resume: decoding/cache state and worker scheduling do not affect the sample.

    ``episodes=[start, stop]`` is an additional half-open range intersected with
    each JSON's selected source split. It never expands the source membership.
    """

    def __init__(self, data_config: dict[str, Any], *, training: bool | None = None) -> None:
        if training is not None:
            data_config = {**data_config, "training": bool(training)}
        self.config = json.loads(json.dumps(data_config, default=str))
        self.obs_horizon = int(data_config.get("obs_horizon", 2))
        self.pred_horizon = int(data_config.get("pred_horizon", 16))
        self.prompt_frames = int(data_config.get("prompt_frames", 10))
        self.image_size = int(data_config.get("image_size", 224))
        self.cache_size = int(data_config.get("cache_size", 64))
        self.numeric_cache_size = int(data_config.get("numeric_cache_size", 16))
        self.decoder_cache_size = int(data_config.get("decoder_cache_size", 8))
        self.training = bool(data_config.get("training", True))
        self.seed = int(data_config.get("seed", 0))
        self.tolerance_s = float(data_config.get("tolerance_s", 0.0001))
        self.split = str(data_config.get("split", "train"))
        assert min(self.obs_horizon, self.pred_horizon, self.prompt_frames,
                   self.image_size, self.numeric_cache_size) > 0
        assert self.cache_size >= 0 and self.decoder_cache_size > 0
        self._paths = {key: Path(data_config[key]).expanduser().resolve()
                       for key in ("human_root", "sim_root", "human_config",
                                   "sim_config", "mapping_file")}
        for key, path in self._paths.items():
            assert path.exists(), f"Missing {key}: {path}"
        override = data_config.get("episodes")
        self._override = None
        if override is not None:
            assert isinstance(override, (list, tuple)) and len(override) == 2
            self._override = range(int(override[0]), int(override[1]))
            assert self._override.start >= 0 and len(self._override) > 0
        normalizer_module = importlib.import_module(
            "examples.baselines.lerobot_dataset.normalizer")
        video_module = importlib.import_module(
            "examples.baselines.lerobot_dataset.video_utils")
        assert hasattr(video_module, "decode_video_frames_torchcodec")
        assert hasattr(video_module.VideoDecoderCache, "invalidate"), (
            "Use TIG 3193957e or a compatible timestamp-checked reader")
        self._runtime_video_module_name = video_module.__name__
        self.normalizer = normalizer_module.ActionNormalizer()
        mapping = json.loads(self._paths["mapping_file"].read_text())["task_mappings"]
        self._sim_to_human: dict[str, str] = {}
        for row in mapping:
            for sim_id in row["sim_task_id"]:
                assert sim_id not in self._sim_to_human
                self._sim_to_human[sim_id] = row["human_task_id"]
        sim_configs = json.loads(self._paths["sim_config"].read_text())
        human_configs = json.loads(self._paths["human_config"].read_text())
        requested = set(data_config.get("task_filter") or [])
        selected = [row for row in sim_configs
                    if not requested or row["repo_id"] in requested]
        assert selected and (not requested or {r["repo_id"] for r in selected} == requested)
        human_by_id = {row["repo_id"]: row for row in human_configs}
        self.repositories: list[_Repository] = []
        self.human_repositories: dict[str, _Repository] = {}
        self.repo_indices: dict[str, int] = {}
        self.anchor_episodes: list[_AnchorEpisode] = []
        self._episode_starts: list[int] = []
        self.group_indices: dict[str, range] = {}
        self.group_offsets: dict[str, tuple[int, int]] = {}
        identities: list[dict[str, Any]] = []
        count = 0
        for row in selected:
            repo_id = row["repo_id"]
            base_name = repo_id.split("_", 1)[1] if repo_id.startswith("L") else repo_id
            if self.training and base_name in UNSEEN_TASKS:
                raise ValueError(f"Unseen robot task cannot enter source training: {repo_id}")
            assert repo_id in self._sim_to_human and repo_id not in self.repo_indices
            human_id = self._sim_to_human[repo_id]
            assert human_id in human_by_id, f"No configured human prompt for {repo_id}"
            repository, identity = self._load_repository(row, "sim_root")
            repository_index = len(self.repositories)
            self.repo_indices[repo_id] = repository_index
            self.repositories.append(repository)
            identities.append(identity)
            stats_path = repository.root / "meta/stats.json"
            stats = json.loads(stats_path.read_text())
            for key, dimension in [(STATE_KEY, 18), (ACTION_KEY, 16)]:
                assert repository.info["features"][key]["shape"] == [dimension]
                for quantile in ("q01", "q99"):
                    values = np.asarray(stats[key][quantile])
                    assert values.shape == (dimension,) and np.isfinite(values).all()
            self.normalizer.add_dataset_stats(repository_index, repo_id, stats,
                                              STATE_KEY, ACTION_KEY, False)
            start = count
            for episode_id, metadata in repository.episodes.items():
                length = int(metadata["length"])
                self._episode_starts.append(count)
                self.anchor_episodes.append(_AnchorEpisode(
                    repository_index, episode_id, length, count))
                count += length
            self.group_indices[repo_id] = range(start, count)
            self.group_offsets[repo_id] = (start, count)
            if human_id not in self.human_repositories:
                human, identity = self._load_repository(human_by_id[human_id], "human_root")
                self.human_repositories[human_id] = human
                identities.append(identity)
        self.group_names = list(self.group_indices)
        self._length = count
        assert count > 0
        runtime_files = [Path(__file__).resolve(),
                         Path(inspect.getfile(normalizer_module.ActionNormalizer)),
                         Path(inspect.getfile(video_module.VideoDecoderCache))]
        self.manifest: dict[str, Any] = {
            "adapter_version": 1, "data_config": self.config,
            "config_files": {key: {"path": str(self._paths[key]),
                                   "sha256": _digest(self._paths[key])}
                             for key in ("human_config", "sim_config", "mapping_file")},
            "runtime_files": [{"path": str(path), "sha256": _digest(path)}
                              for path in runtime_files],
            "repositories": identities, "groups": self.group_offsets,
            "robot_repositories": len(self.repositories),
            "robot_episodes": len(self.anchor_episodes), "frame_anchors": count,
            "human_repositories": len(self.human_repositories),
            "human_episodes": sum(len(r.episodes) for r in self.human_repositories.values()),
            "obs_horizon": self.obs_horizon, "pred_horizon": self.pred_horizon,
            "prompt_frames": self.prompt_frames, "image_size": self.image_size,
            "normalization": "runtime TIG bounds_q99; BPP identity",
            "rgb": "RGB CHW float32 [0,1]; CLIP pixel normalization belongs to model",
            "action": "absolute dual Panda joint positions + normalized gripper, 8+8",
            "sampling": "frame anchors; group ranges permit equal task-level draws",
            "prompt_sampling": "task-mapped human episode; deterministic per (seed, anchor, draw_seed)",
            "history_padding": "repeat episode first observation",
            "action_padding": "repeat episode final command; no masked objective",
            "video_reader": "BPP exact-only factory; official torchcodec PTS checks and fresh-decoder retry; tolerance unchanged",
            "large_file_integrity": "path/size/mtime recorded; no full video/data hash in startup",
        }
        self.manifest["sha256"] = hashlib.sha256(json.dumps(
            self.manifest, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
        self._clear_process_caches()

    def _load_repository(self, config: dict[str, Any], root_key: str
                         ) -> tuple[_Repository, dict[str, Any]]:
        root = (self._paths[root_key] / config["root"]).resolve()
        info_path, stats_path = root / "meta/info.json", root / "meta/stats.json"
        info = json.loads(info_path.read_text())
        assert RGB_KEY in info["features"], f"Missing zed2i RGB: {root}"
        rows = _metadata_rows(root)
        available = {int(row["episode_index"]): row for row in rows}
        ids = _episode_ids(config.get(self.split), sorted(available))
        if self._override is not None:
            ids = [i for i in ids if i in self._override]
        assert ids and len(ids) == len(set(ids)), f"Empty/duplicate episodes: {root}"
        assert set(ids).issubset(available), f"Missing selected episodes: {root}"
        episodes = {i: available[i] for i in sorted(ids)}
        repository = _Repository(config["repo_id"], root, info, episodes)
        paths: set[Path] = {info_path, stats_path}
        paths.update((root / "meta/episodes").glob("chunk-*/file-*.parquet"))
        for row in episodes.values():
            paths.add(self._video_path(repository, row))
            if root_key == "sim_root":
                paths.add(root / info["data_path"].format(
                    chunk_index=int(row["data/chunk_index"]),
                    file_index=int(row["data/file_index"])))
        for path in paths:
            assert path.is_file() and path.stat().st_size > 0, f"Missing data: {path}"
        identity = {"repo_id": config["repo_id"], "root": str(root),
                    "episodes": list(episodes),
                    "frames": sum(int(row["length"]) for row in episodes.values()),
                    "info_sha256": _digest(info_path), "stats_sha256": _digest(stats_path),
                    "files": [{"path": str(path), "bytes": path.stat().st_size,
                               "mtime_ns": path.stat().st_mtime_ns}
                              for path in sorted(paths)]}
        return repository, identity

    @staticmethod
    def _video_path(repository: _Repository, row: dict[str, Any]) -> Path:
        return repository.root / repository.info["video_path"].format(
            video_key=RGB_KEY, chunk_index=int(row[f"videos/{RGB_KEY}/chunk_index"]),
            file_index=int(row[f"videos/{RGB_KEY}/file_index"]))

    def _clear_process_caches(self) -> None:
        self._pid = os.getpid()
        self._numeric_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._frame_cache: OrderedDict[tuple[str, float], torch.Tensor] = OrderedDict()
        self._decoder_cache: _BoundedDecoders | None = None
        self.cache_counters = {"frame_hits": 0, "frame_misses": 0, "numeric_loads": 0}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.update(_numeric_cache=OrderedDict(), _frame_cache=OrderedDict(),
                     _decoder_cache=None, _pid=None)
        return state

    def _ensure_process(self) -> None:
        if self._pid != os.getpid():
            self._clear_process_caches()

    def __len__(self) -> int:
        return self._length

    def _numeric(self, repository_index: int) -> dict[str, Any]:
        self._ensure_process()
        if repository_index in self._numeric_cache:
            self._numeric_cache.move_to_end(repository_index)
            return self._numeric_cache[repository_index]
        repository = self.repositories[repository_index]
        columns = ["index", "episode_index", "frame_index", "timestamp", STATE_KEY, ACTION_KEY]
        chunks = [pq.ParquetFile(path).read(columns=columns)
                  for path in sorted((repository.root / "data").glob("chunk-*/file-*.parquet"))]
        assert chunks, f"No robot parquet: {repository.root}"
        import pyarrow as pa
        table = pa.concat_tables(chunks).sort_by("index")
        indices = table["index"].to_numpy(zero_copy_only=False).astype(np.int64)
        assert len(indices) > 0 and np.all(np.diff(indices) == 1)
        raw_state = np.asarray(table[STATE_KEY].to_pylist(), dtype=np.float32)
        raw_action = np.asarray(table[ACTION_KEY].to_pylist(), dtype=np.float32)
        assert raw_state.shape == (len(indices), 18) and raw_action.shape == (len(indices), 16)
        assert np.isfinite(raw_state).all() and np.isfinite(raw_action).all()
        values = {
            "index_start": int(indices[0]),
            "episode_index": table["episode_index"].to_numpy(zero_copy_only=False),
            "frame_index": table["frame_index"].to_numpy(zero_copy_only=False),
            "timestamp": table["timestamp"].to_numpy(zero_copy_only=False).astype(np.float64),
            "qpos": self.normalizer.normalize_state(torch.from_numpy(raw_state),
                        repository_index, method="bounds_q99").numpy(),
            "action": self.normalizer.normalize_action(torch.from_numpy(raw_action),
                        repository_index, method="bounds_q99").numpy(),
        }
        assert np.isfinite(values["qpos"]).all() and np.isfinite(values["action"]).all()
        self._numeric_cache[repository_index] = values
        self.cache_counters["numeric_loads"] += 1
        while len(self._numeric_cache) > self.numeric_cache_size:
            self._numeric_cache.popitem(last=False)
        return values

    def _read_rgb(self, video_path: Path, timestamps: Sequence[float]) -> torch.Tensor:
        self._ensure_process()
        assert len(timestamps) > 0
        keys = [(str(video_path), float(ts)) for ts in timestamps]
        missing = list(dict.fromkeys(key for key in keys if key not in self._frame_cache))
        retrieved: dict[tuple[str, float], torch.Tensor] = {}
        for key in keys:
            if key in self._frame_cache:
                self.cache_counters["frame_hits"] += 1
                self._frame_cache.move_to_end(key)
                retrieved[key] = self._frame_cache[key]
        if missing:
            module = importlib.import_module(self._runtime_video_module_name)
            if self._decoder_cache is None:
                self._decoder_cache = _BoundedDecoders(module, self.decoder_cache_size)
            decoded = module.decode_video_frames_torchcodec(
                str(video_path), [key[1] for key in missing], self.tolerance_s,
                decoder_cache=self._decoder_cache)
            assert decoded.ndim == 4 and decoded.shape[:2] == (len(missing), 3)
            assert decoded.is_floating_point() and torch.isfinite(decoded).all()
            assert float(decoded.min()) >= 0 and float(decoded.max()) <= 1
            if tuple(decoded.shape[-2:]) != (self.image_size, self.image_size):
                decoded = F.interpolate(decoded, size=(self.image_size, self.image_size),
                                        mode="bilinear", align_corners=False, antialias=True)
            pixels = decoded.mul(255).round().clamp_(0, 255).to(torch.uint8)
            for key, frame in zip(missing, pixels, strict=True):
                # Clone prevents one cached frame retaining a whole decode batch.
                retrieved[key] = frame.contiguous().clone()
                if self.cache_size:
                    self._frame_cache[key] = retrieved[key]
                    self._frame_cache.move_to_end(key)
                    while len(self._frame_cache) > self.cache_size:
                        self._frame_cache.popitem(last=False)
            self.cache_counters["frame_misses"] += len(missing)
        frames = torch.stack([retrieved[key] for key in keys]).float().div_(255)
        assert frames.shape == (len(keys), 3, self.image_size, self.image_size)
        return frames

    def get_human_prompt(self, sim_repo_id: str, episode_index: int | None = None,
                         draw_seed: int = 0, *, official_eval: bool = False
                         ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Read a reproducible human clip; explicit episode uses uniform no-jitter frames."""
        human_id = self._sim_to_human[sim_repo_id]
        repository = self.human_repositories[human_id]
        ids = list(repository.episodes)
        rng = np.random.default_rng(_stable_seed(self.seed, human_id, draw_seed))
        if official_eval:
            # Call the pinned official sampling implementation, isolating its
            # legacy global RNGs so prompt generation cannot change policy noise.
            from types import SimpleNamespace
            import random
            from examples.baselines.lerobot_dataset.lerobot_human_dataset import HumanVideoDataset
            state_np, state_py = np.random.get_state(), random.getstate()
            seed = _stable_seed(self.seed, human_id, draw_seed) % 2**32
            try:
                np.random.seed(seed)
                random.seed(seed)
                if episode_index is None:
                    episode_index = ids[random.randint(0, len(ids) - 1)]
                assert episode_index in repository.episodes
                row = repository.episodes[episode_index]
                length = int(row["length"])
                sampler = SimpleNamespace(config=SimpleNamespace(max_jitter=3))
                indices = np.asarray(HumanVideoDataset._sample_frames_uniform_jitter(
                    sampler, 0, length - 1, self.prompt_frames), dtype=np.int64)
            finally:
                np.random.set_state(state_np)
                random.setstate(state_py)
            jitter = False
        elif episode_index is None:
            episode_index = ids[int(rng.integers(len(ids)))]
            jitter = self.training
        else:
            assert episode_index in repository.episodes, (human_id, episode_index)
            jitter = False
        row = repository.episodes[episode_index]
        length = int(row["length"])
        if not official_eval:
            indices = np.linspace(0, length - 1, min(length, self.prompt_frames)).astype(np.int64)
        if jitter and length > self.prompt_frames:
            max_jitter = min(3, length // (self.prompt_frames * 2))
            if max_jitter:
                indices = np.sort(np.clip(indices + rng.integers(
                    -max_jitter, max_jitter + 1, size=len(indices)), 0, length - 1))
                indices = np.unique(indices)
                while len(indices) < self.prompt_frames:
                    gaps = np.diff(indices)
                    if len(gaps) and int(gaps.max()) > 1:
                        gap = int(np.argmax(gaps))
                        extra = (int(indices[gap]) + int(indices[gap + 1])) // 2
                    else:
                        unused = np.setdiff1d(np.arange(length), indices)
                        extra = int(unused[0])
                    indices = np.sort(np.append(indices, extra))
        valid_length = len(indices)
        if valid_length < self.prompt_frames:
            indices = np.pad(indices, (0, self.prompt_frames - valid_length), mode="edge")
        assert indices.shape == (self.prompt_frames,)
        offset = float(row[f"videos/{RGB_KEY}/from_timestamp"])
        fps = float(repository.info["fps"])
        # Match official evaluate_processor's int(from_timestamp * fps)
        # absolute-frame calculation in formal mode, including its rounding.
        timestamps = (((int(offset * fps) + indices.astype(np.float64)) / fps)
                      if official_eval else (offset + indices.astype(np.float64) / fps)).tolist()
        video_path = self._video_path(repository, row)
        rgb = self._read_rgb(video_path, timestamps)
        metadata = {"human_repo_id": human_id, "episode_index": int(episode_index),
                    "frame_indices": indices.tolist(), "video_path": str(video_path),
                    "timestamps": timestamps,
                    "absolute_frame_indices": [round(ts * fps) for ts in timestamps],
                    "mask": [i >= valid_length for i in range(self.prompt_frames)],
                    "draw_seed": int(draw_seed),
                    "sampling": "official uniform_jitter(max_jitter=3)" if official_eval else "adapter training jitter" if self.training else "fixed uniform"}
        return rgb, metadata

    def normalize_state(self, qpos: torch.Tensor | np.ndarray,
                        sim_repo_id: str) -> torch.Tensor:
        qpos = torch.as_tensor(qpos, dtype=torch.float32)
        assert qpos.shape[-1] == 18 and torch.isfinite(qpos).all()
        return self.normalizer.normalize_state(qpos, self.repo_indices[sim_repo_id],
                                               method="bounds_q99")

    def denormalize_action(self, action: torch.Tensor | np.ndarray,
                           sim_repo_id: str) -> torch.Tensor:
        action = torch.as_tensor(action, dtype=torch.float32)
        assert action.shape[-1] == 16 and torch.isfinite(action).all()
        return self.normalizer.denormalize_action(action, self.repo_indices[sim_repo_id],
                                                  method="bounds_q99")

    def __getitem__(self, index: int | tuple[int, int]) -> dict[str, Any]:
        if isinstance(index, tuple):
            anchor_index, draw_seed = map(int, index)
        else:
            anchor_index, draw_seed = int(index), 0
        if anchor_index < 0:
            anchor_index += len(self)
        assert 0 <= anchor_index < len(self), anchor_index
        position = bisect.bisect_right(self._episode_starts, anchor_index) - 1
        episode = self.anchor_episodes[position]
        repository = self.repositories[episode.repository_index]
        row = repository.episodes[episode.episode_index]
        anchor = anchor_index - episode.global_start
        history, future = window_indices(episode.length, anchor, self.obs_horizon,
                                         self.pred_horizon)
        values = self._numeric(episode.repository_index)
        start = int(row["dataset_from_index"]) - values["index_start"]
        hi, fi = start + history, start + future
        assert np.all(values["episode_index"][hi] == episode.episode_index)
        assert np.all(values["episode_index"][fi] == episode.episode_index)
        assert np.array_equal(values["frame_index"][hi], history)
        offset = float(row[f"videos/{RGB_KEY}/from_timestamp"])
        timestamps = (offset + values["timestamp"][hi]).tolist()
        rgb = self._read_rgb(self._video_path(repository, row), timestamps)
        qpos = torch.from_numpy(values["qpos"][hi].copy()).float()
        action = torch.from_numpy(values["action"][fi].copy()).float()
        human_rgb, prompt_metadata = self.get_human_prompt(
            repository.repo_id, draw_seed=_stable_seed(anchor_index, draw_seed))
        sample = {
            "obs": {"rgb": rgb, "qpos": qpos,
                    "prompt": {"obs": {"rgb": human_rgb},
                               "metadata": {"mask": torch.tensor(
                                   prompt_metadata["mask"], dtype=torch.bool)}}},
            "action": action,
            "metadata": {"sim_repo_id": repository.repo_id,
                         "robot_episode_index": episode.episode_index,
                         "robot_frame_index": anchor, "anchor_index": anchor_index,
                         "draw_seed": draw_seed, "human_prompt": prompt_metadata},
        }
        assert qpos.shape == (self.obs_horizon, 18)
        assert action.shape == (self.pred_horizon, 16)
        assert torch.isfinite(qpos).all() and torch.isfinite(action).all()
        assert float(qpos.abs().max()) <= 1 and float(action.abs().max()) <= 1
        return sample

    def debug_boundary_checks(self) -> dict[str, int]:
        """Check every selected episode boundary without decoding any video."""
        for episode in self.anchor_episodes:
            history, _ = window_indices(episode.length, 0, self.obs_horizon, self.pred_horizon)
            _, future = window_indices(episode.length, episode.length - 1,
                                        self.obs_horizon, self.pred_horizon)
            assert np.all(history == 0) and np.all(future == episode.length - 1)
        return {"checked_episodes": len(self.anchor_episodes), "frame_anchors": len(self)}

    def close(self) -> None:
        if self._decoder_cache is not None:
            self._decoder_cache.clear()
        self._clear_process_caches()


def collate_bpp(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collate fixed-length RGB prompts, keeping trace metadata out of the model."""
    assert samples
    batch = {
        "obs": {"rgb": torch.stack([s["obs"]["rgb"] for s in samples]),
                "qpos": torch.stack([s["obs"]["qpos"] for s in samples]),
                "prompt": {"obs": {"rgb": torch.stack([
                    s["obs"]["prompt"]["obs"]["rgb"] for s in samples])},
                    "metadata": {"mask": torch.stack([
                        s["obs"]["prompt"]["metadata"]["mask"] for s in samples])}}},
        "action": torch.stack([s["action"] for s in samples]),
        "metadata": [s.get("metadata", {}) for s in samples],
    }
    assert batch["obs"]["rgb"].ndim == 5 and batch["obs"]["rgb"].shape[2] == 3
    assert batch["obs"]["qpos"].shape[-1] == 18 and batch["action"].shape[-1] == 16
    assert batch["obs"]["prompt"]["metadata"]["mask"].dtype == torch.bool
    for tensor in [batch["obs"]["rgb"], batch["obs"]["qpos"], batch["action"],
                   batch["obs"]["prompt"]["obs"]["rgb"]]:
        assert torch.isfinite(tensor).all()
    return batch


def build(data_config: dict[str, Any]) -> TIGBPPDataset:
    return TIGBPPDataset(data_config)


def _self_test() -> None:
    """Synthetic boundary and shape checks; no data download or video decode."""
    for length in [1, 2, 17]:
        for anchor in [0, length - 1]:
            history, future = window_indices(length, anchor, 2, 16)
            assert len(history) == 2 and len(future) == 16
            if anchor == 0:
                assert np.all(history == 0)
            if anchor == length - 1:
                assert np.all(future == length - 1)
    sample = {"obs": {"rgb": torch.zeros(2, 3, 8, 8), "qpos": torch.zeros(2, 18),
                      "prompt": {"obs": {"rgb": torch.ones(10, 3, 8, 8)},
                                 "metadata": {"mask": torch.zeros(10, dtype=torch.bool)}}},
              "action": torch.zeros(16, 16), "metadata": {"synthetic": True}}
    for batch_size in [1, 2]:
        batch = collate_bpp([sample] * batch_size)
        assert batch["obs"]["rgb"].shape == (batch_size, 2, 3, 8, 8)
        assert batch["action"].shape == (batch_size, 16, 16)
        assert batch["obs"]["prompt"]["metadata"]["mask"].shape == (batch_size, 10)
    assert _stable_seed(7, 5) == _stable_seed(7, 5) != _stable_seed(7, 6)
    assert _episode_ids("0:3", [0, 1, 2, 3]) == [0, 1, 2]
    from types import SimpleNamespace
    from unittest.mock import patch
    import random
    import sys
    class FakeCache:
        def __init__(self) -> None:
            self.values = {}
        def get_decoder(self, path: str) -> Any:
            if path not in self.values:
                self.values[path] = self._create_decoder(path)
            return self.values[path][0]
        def invalidate(self, path: str) -> None:
            self.values.pop(path, None)
        def clear(self) -> None:
            self.values.clear()
    calls = []
    def decoder(path: str, *, seek_mode: str) -> Any:
        calls.append((path, seek_mode))
        return object()
    with patch.dict(sys.modules, {"torchcodec.decoders": SimpleNamespace(VideoDecoder=decoder)}):
        cache = _BoundedDecoders(SimpleNamespace(VideoDecoderCache=FakeCache), 1)
        first = cache.get_decoder("a.mp4")
        assert cache.get_decoder("a.mp4") is first
        cache.get_decoder("b.mp4")
        assert list(cache.base.values) == ["b.mp4"]
        assert calls == [("a.mp4", "exact"), ("b.mp4", "exact")]
        cache.clear()
        assert not cache.base.values
    sample_dataset = TIGBPPDataset.__new__(TIGBPPDataset)
    sample_dataset.seed, sample_dataset.prompt_frames, sample_dataset.training = 13, 10, False
    sample_dataset._sim_to_human = {"sim": "human"}
    row = {"length": 20, f"videos/{RGB_KEY}/from_timestamp": 0.,
           f"videos/{RGB_KEY}/chunk_index": 0, f"videos/{RGB_KEY}/file_index": 0}
    sample_dataset.human_repositories = {"human": _Repository("human", Path("/fixture"),
        {"fps": 30, "video_path": "{chunk_index}_{file_index}.mp4"}, {0: row})}
    sample_dataset._read_rgb = lambda path, timestamps: torch.zeros(len(timestamps), 3, 8, 8)
    class Sampler:
        @staticmethod
        def _sample_frames_uniform_jitter(obj: Any, start: int, end: int, count: int) -> list[int]:
            assert obj.config.max_jitter == 3 and (start, end, count) == (0, 19, 10)
            return list(range(10))
    state_np, state_py = np.random.get_state(), random.getstate()
    with patch.dict(sys.modules, {"examples.baselines.lerobot_dataset.lerobot_human_dataset":
                                 SimpleNamespace(HumanVideoDataset=Sampler)}):
        rgb, metadata = sample_dataset.get_human_prompt("sim", draw_seed=5, official_eval=True)
    assert rgb.shape == (10, 3, 8, 8) and metadata["frame_indices"] == list(range(10))
    assert np.array_equal(np.random.get_state()[1], state_np[1]) and random.getstate() == state_py
    assert not torch.cuda.is_initialized()
    print("PASS: episode boundaries; B=1/2 RGB/q18/action16/prompt10; deterministic sampling; exact-only bounded decoder; official sampler delegation; RNG isolation; no CUDA")


if __name__ == "__main__":
    _self_test()
