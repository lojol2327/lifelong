from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np


@dataclass
class CandidateView:
    frontier_id: int
    view_direction: np.ndarray  # shape (2,) on ground plane (x,z) or (dx, dz)
    camera_tilt_rad: float
    meta: Dict[str, Any]


class FrontierManager:
    """Manage frontier extraction and candidate viewpoints per frontier.

    This class wraps interactions with the TSDF planner to keep run loop clean.
    """

    def __init__(self, num_candidates_per_frontier: int = 3, default_camera_tilt_deg: float = -25.0) -> None:
        self.num_candidates_per_frontier = num_candidates_per_frontier
        self.default_camera_tilt_rad = np.deg2rad(default_camera_tilt_deg)

    def update(
        self,
        scene,
        tsdf_planner,
        pts: np.ndarray,
        cfg_planner,
        step: int,
        save_frontier_image: bool,
        eps_frontier_dir: str,
        prompt_img_size: Tuple[int, int],
    ) -> List:
        """Update frontier map via TSDF planner and return current frontiers list.

        Returns the list of frontier objects owned by tsdf_planner (could be empty).
        """
        _ = tsdf_planner.update_frontier_map(
            pts=pts,
            cfg=cfg_planner,
            scene=scene,
            cnt_step=step,
            save_frontier_image=save_frontier_image,
            eps_frontier_dir=eps_frontier_dir,
            prompt_img_size=prompt_img_size,
        )
        return getattr(tsdf_planner, "frontiers", [])

    def get_candidate_viewpoints(self, frontier, k: int | None = None) -> List[CandidateView]:
        """Generate k candidate viewing directions for a given frontier.

        For now, use frontier.orientation (dx, dz) and add small angular offsets to diversify.
        """
        if k is None:
            k = self.num_candidates_per_frontier

        candidates: List[CandidateView] = []

        # orientation is assumed as a 2D ground-plane direction (dx, dz)
        base = np.asarray(frontier.orientation, dtype=float).reshape(-1)
        if base.size == 2 and np.linalg.norm(base) > 0:
            base = base / np.linalg.norm(base)
        else:
            base = np.array([1.0, 0.0], dtype=float)

        # small angular spread around base
        angles = np.linspace(-np.deg2rad(20), np.deg2rad(20), k)
        for a in angles:
            ca, sa = np.cos(a), np.sin(a)
            rot = np.array([[ca, -sa], [sa, ca]], dtype=float)
            direction = (rot @ base).astype(float)
            candidates.append(
                CandidateView(
                    frontier_id=frontier.frontier_id,
                    view_direction=direction,
                    camera_tilt_rad=self.default_camera_tilt_rad,
                    meta={"angle_offset_rad": float(a)},
                )
            )
        return candidates

    def plan_candidate_views(
        self,
        scene,
        tsdf_planner,
        current_pts: np.ndarray,
        frontier,
        k: int,
        standoff_m: float = 0.6,
        camera_tilt_rad: float | None = None,
    ) -> List[Dict[str, Any]]:
        """Plan candidate camera poses around a frontier with simple standoff.

        Returns a list of dicts: {"target_pts": np.ndarray, "view_dir": np.ndarray, "meta": {...}}
        """
        if camera_tilt_rad is None:
            camera_tilt_rad = self.default_camera_tilt_rad

        cand_dirs = self.get_candidate_viewpoints(frontier, k=k)
        out: List[Dict[str, Any]] = []
        frontier_hab = tsdf_planner.voxel2habitat(frontier.position)
        for cd in cand_dirs:
            view_dir_3d = np.array([cd.view_direction[0], 0.0, cd.view_direction[1]], dtype=float)
            view_dir_3d = view_dir_3d / (np.linalg.norm(view_dir_3d) + 1e-8)
            raw_target = frontier_hab - view_dir_3d * float(standoff_m)

            # snap to navigable
            try:
                nav_pt = scene.get_navigable_point_to(
                    target_position=raw_target,
                    min_dist=max(0.1, standoff_m * 0.3),
                    max_dist=max(1.0, standoff_m * 1.5),
                    prev_start_positions=[current_pts],
                )
            except Exception:
                nav_pt = raw_target

            out.append({
                "target_pts": nav_pt,
                "view_dir": view_dir_3d,
                "camera_tilt_rad": float(camera_tilt_rad),
                "meta": {"frontier_id": frontier.frontier_id, **cd.meta},
            })
        return out


