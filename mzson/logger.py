from __future__ import annotations

import os
import json
import pickle
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np


class MZSONLogger:
    """Logger for mzson experiments, mirroring useful features from goatbench logger.

    - Tracks success and SPL statistics
    - Persists intermediate aggregated results
    - Manages episode/subtask directories
    - Adds mzson-specific hooks to record frontier candidates, descriptors, and selections
    """

    def __init__(
        self,
        output_dir: str,
        start_ratio: float,
        end_ratio: float,
        split: int,
        voxel_size: float,
    ) -> None:
        self.output_dir = output_dir
        self.voxel_size = voxel_size
        os.makedirs(self.output_dir, exist_ok=True)

        # Statistics stores
        def _load_or_default(path: str, default):
            if os.path.exists(path):
                with open(path, "rb" if path.endswith(".pkl") else "r") as f:
                    return pickle.load(f) if path.endswith(".pkl") else json.load(f)
            return default

        self.success_by_snapshot = _load_or_default(
            os.path.join(self.output_dir, f"success_by_snapshot_{start_ratio}_{end_ratio}_{split}.pkl"),
            {},
        )
        self.success_by_distance = _load_or_default(
            os.path.join(self.output_dir, f"success_by_distance_{start_ratio}_{end_ratio}_{split}.pkl"),
            {},
        )
        self.spl_by_snapshot = _load_or_default(
            os.path.join(self.output_dir, f"spl_by_snapshot_{start_ratio}_{end_ratio}_{split}.pkl"),
            {},
        )
        self.spl_by_distance = _load_or_default(
            os.path.join(self.output_dir, f"spl_by_distance_{start_ratio}_{end_ratio}_{split}.pkl"),
            {},
        )
        self.success_by_task = _load_or_default(
            os.path.join(self.output_dir, f"success_by_task_{start_ratio}_{end_ratio}_{split}.pkl"),
            defaultdict(list),
        )
        self.spl_by_task = _load_or_default(
            os.path.join(self.output_dir, f"spl_by_task_{start_ratio}_{end_ratio}_{split}.pkl"),
            defaultdict(list),
        )
        self.n_filtered_snapshots_list = _load_or_default(
            os.path.join(self.output_dir, f"n_filtered_snapshots_{start_ratio}_{end_ratio}_{split}.json"),
            {},
        )
        self.n_total_snapshots_list = _load_or_default(
            os.path.join(self.output_dir, f"n_total_snapshots_{start_ratio}_{end_ratio}_{split}.json"),
            {},
        )
        self.n_total_frames_list = _load_or_default(
            os.path.join(self.output_dir, f"n_total_frames_{start_ratio}_{end_ratio}_{split}.json"),
            {},
        )

        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.split = split

        # Episode/Subtask state
        self.episode_dir: str | None = None
        self.subtask_object_observe_dir: str | None = None
        self.pts_voxels = np.empty((0, 2))
        self.subtask_explore_dist = 0.0

    # ---------- persistence ----------
    def save_results(self) -> None:
        def _dump(obj, path: str):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb" if path.endswith(".pkl") else "w") as f:
                pickle.dump(obj, f) if path.endswith(".pkl") else json.dump(obj, f, indent=4)

        _dump(
            self.success_by_snapshot,
            os.path.join(self.output_dir, f"success_by_snapshot_{self.start_ratio}_{self.end_ratio}_{self.split}.pkl"),
        )
        _dump(
            self.success_by_distance,
            os.path.join(self.output_dir, f"success_by_distance_{self.start_ratio}_{self.end_ratio}_{self.split}.pkl"),
        )
        _dump(
            self.spl_by_snapshot,
            os.path.join(self.output_dir, f"spl_by_snapshot_{self.start_ratio}_{self.end_ratio}_{self.split}.pkl"),
        )
        _dump(
            self.spl_by_distance,
            os.path.join(self.output_dir, f"spl_by_distance_{self.start_ratio}_{self.end_ratio}_{self.split}.pkl"),
        )
        _dump(
            self.success_by_task,
            os.path.join(self.output_dir, f"success_by_task_{self.start_ratio}_{self.end_ratio}_{self.split}.pkl"),
        )
        _dump(
            self.spl_by_task,
            os.path.join(self.output_dir, f"spl_by_task_{self.start_ratio}_{self.end_ratio}_{self.split}.pkl"),
        )
        _dump(
            self.n_filtered_snapshots_list,
            os.path.join(self.output_dir, f"n_filtered_snapshots_{self.start_ratio}_{self.end_ratio}_{self.split}.json"),
        )
        _dump(
            self.n_total_snapshots_list,
            os.path.join(self.output_dir, f"n_total_snapshots_{self.start_ratio}_{self.end_ratio}_{self.split}.json"),
        )
        _dump(
            self.n_total_frames_list,
            os.path.join(self.output_dir, f"n_total_frames_{self.start_ratio}_{self.end_ratio}_{self.split}.json"),
        )

    # ---------- aggregation ----------
    def aggregate_results(self) -> None:
        # Placeholder: adopt the same aggregation pattern if needed.
        pass

    # ---------- episode/subtask management ----------
    def init_episode(self, episode_id: str) -> Tuple[str, str, str]:
        self.episode_dir = os.path.join(self.output_dir, episode_id)
        eps_frontier_dir = os.path.join(self.episode_dir, "frontier")
        eps_snapshot_dir = os.path.join(self.episode_dir, "snapshot")
        os.makedirs(self.episode_dir, exist_ok=True)
        os.makedirs(eps_frontier_dir, exist_ok=True)
        os.makedirs(eps_snapshot_dir, exist_ok=True)
        return self.episode_dir, eps_frontier_dir, eps_snapshot_dir

    def init_subtask(self, subtask_id: str) -> None:
        self.subtask_object_observe_dir = os.path.join(
            self.output_dir, subtask_id, "object_observations"
        )
        if os.path.exists(self.subtask_object_observe_dir):
            # clear any previous leftovers for this subtask id
            try:
                import shutil

                shutil.rmtree(self.subtask_object_observe_dir)
            except Exception:
                pass
        os.makedirs(self.subtask_object_observe_dir, exist_ok=True)
        self.pts_voxels = np.empty((0, 2))
        self.subtask_explore_dist = 0.0

    def log_step(self, pts_voxel: np.ndarray) -> None:
        self.pts_voxels = np.vstack([self.pts_voxels, pts_voxel])
        if len(self.pts_voxels) >= 2:
            self.subtask_explore_dist += (
                np.linalg.norm(self.pts_voxels[-1] - self.pts_voxels[-2]) * self.voxel_size
            )

    # ---------- subtask result ----------
    def log_subtask_result(
        self,
        success_by_snapshot: bool,
        success_by_distance: bool,
        subtask_id: str,
        gt_subtask_explore_dist: float,
        goal_type: str,
        n_filtered_snapshots: int,
        n_total_snapshots: int,
        n_total_frames: int,
    ) -> None:
        self.success_by_snapshot[subtask_id] = 1.0 if success_by_snapshot else 0.0
        self.success_by_distance[subtask_id] = 1.0 if success_by_distance else 0.0

        # SPL
        def _spl(success: float) -> float:
            denom = max(gt_subtask_explore_dist, self.subtask_explore_dist, 1e-6)
            return success * gt_subtask_explore_dist / denom

        self.spl_by_snapshot[subtask_id] = _spl(self.success_by_snapshot[subtask_id])
        self.spl_by_distance[subtask_id] = _spl(self.success_by_distance[subtask_id])
        self.success_by_task.setdefault(goal_type, []).append(self.success_by_distance[subtask_id])
        self.spl_by_task.setdefault(goal_type, []).append(self.spl_by_distance[subtask_id])

        # snapshot/frame counts
        self.n_filtered_snapshots_list[subtask_id] = int(n_filtered_snapshots)
        self.n_total_snapshots_list[subtask_id] = int(n_total_snapshots)
        self.n_total_frames_list[subtask_id] = int(n_total_frames)

        # reset per-subtask trackers
        self.subtask_object_observe_dir = None
        self.pts_voxels = np.empty((0, 2))
        self.subtask_explore_dist = 0.0

    # ---------- mzson specific hooks ----------
    def log_frontier_candidates(
        self,
        step: int,
        frontiers: List[Any],
        descriptors: List[Any],
        scores: List[float] | None = None,
    ) -> None:
        path = os.path.join(self.output_dir, f"candidates_{step:03d}.json")
        data = {
            "step": step,
            "num_frontiers": len(frontiers),
            "num_descriptors": len(descriptors),
            "scores": scores if scores is not None else [],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def log_selection(self, step: int, decision: Dict[str, Any], reason: str = "") -> None:
        path = os.path.join(self.output_dir, f"selection_{step:03d}.json")
        with open(path, "w") as f:
            json.dump({"step": step, "decision": decision, "reason": reason}, f, indent=2)



