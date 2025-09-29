import numpy as np
from PIL import Image
import habitat_sim
from habitat_sim.utils.common import quat_to_angle_axis, quat_from_coeffs
import quaternion
from typing import Optional, Tuple, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from mzson.tsdf_planner import TSDFPlanner
    from mzson.data_models import Frontier

from mzson.habitat import pos_habitat_to_normal


def get_frontier_crop_box(
    frontier: "Frontier",
    tsdf_planner: "TSDFPlanner",
    cam_intr: np.ndarray,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
) -> Optional[Tuple[int, int, int, int]]:
    """
    프론티어 관측 이미지에서 실제 미탐색 영역에 해당하는 부분을 찾아
    Crop할 바운딩 박스(x_min, y_min, x_max, y_max)를 반환합니다.

    Args:
        frontier: 프론티어 객체. observation_image, depth, cam_pose를 포함해야 함.
        tsdf_planner: TSDF 플래너. unexplored 맵 정보를 사용.
        cam_intr: 카메라 내부 파라미터.
        min_depth: 유효 깊이 최소값.
        max_depth: 유효 깊이 최대값.

    Returns:
        Crop할 바운딩 박스 (x_min, y_min, x_max, y_max) 또는 None.
    """
    # 필수 데이터 확인
    if (
        frontier.observation_image is None
        or frontier.observation_depth is None
        or frontier.observation_cam_pose is None
    ):
        return None

    depth = frontier.observation_depth
    cam_pose_habitat = frontier.observation_cam_pose

    # 이미지 크기
    H, W = depth.shape

    # 1. 픽셀 좌표 그리드 생성
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    pixels_2d = np.stack([xs, ys], axis=-1).reshape(-1, 2)  # (H*W, 2)

    # 2. 유효 깊이 값을 가진 픽셀만 필터링
    valid_depth_mask = (depth > min_depth) & (depth < max_depth)
    pixels_2d_valid = pixels_2d[valid_depth_mask.reshape(-1)]
    depth_valid = depth[valid_depth_mask]

    if len(depth_valid) == 0:
        return None

    # 3. 3D 공간으로 역투영
    cam_fx, cam_fy = cam_intr[0, 0], cam_intr[1, 1]
    cam_cx, cam_cy = cam_intr[0, 2], cam_intr[1, 2]

    x_cam = (pixels_2d_valid[:, 0] - cam_cx) * depth_valid / cam_fx
    y_cam = (pixels_2d_valid[:, 1] - cam_cy) * depth_valid / cam_fy
    z_cam = -depth_valid # Habitat 좌표계는 -Z가 정면

    # 월드 좌표로 변환
    points_cam = np.vstack([x_cam, y_cam, z_cam, np.ones_like(z_cam)])
    points_world_habitat = (cam_pose_habitat @ points_cam)[:3, :].T

    # 4. 미탐색 영역 필터링
    points_world_normal = pos_habitat_to_normal(points_world_habitat)
    points_voxel = tsdf_planner.normal2voxel(points_world_normal)

    # 유효한 복셀 인덱스 마스크 생성 (TSDF 맵 경계 체크)
    tsdf_h, tsdf_w = tsdf_planner.frontier_map.shape
    valid_voxel_indices_mask = (
        (points_voxel[:, 0] >= 0)
        & (points_voxel[:, 0] < tsdf_h)
        & (points_voxel[:, 1] >= 0)
        & (points_voxel[:, 1] < tsdf_w)
    )

    # TSDF 맵의 경계를 벗어나는 유효하지 않은 복셀 좌표들을 필터링합니다.
    points_voxel_valid = points_voxel[valid_voxel_indices_mask]
    pixels_2d_final = pixels_2d_valid[valid_voxel_indices_mask]

    # --- 여기가 핵심 수정 사항 ---
    # 이전 로직: 단순히 '미탐색(unexplored)'인지 여부만 확인하여 너무 광범위했음
    # unexplored_flags = tsdf_planner.unexplored[points_voxel_valid[:, 0], points_voxel_valid[:, 1]]
    # pixels_in_unexplored = pixels_2d_final[unexplored_flags]

    # 새 로직: 해당 복셀이 '특정 프론티어 ID'에 속하는지 직접 확인하여 정확도를 높임
    voxel_frontier_ids = tsdf_planner.frontier_map[
        points_voxel_valid[:, 0], points_voxel_valid[:, 1]
    ]
    correct_frontier_mask = voxel_frontier_ids == frontier.frontier_id
    pixels_in_unexplored = pixels_2d_final[correct_frontier_mask]
    # --- 수정 끝 ---

    # 유효한 픽셀이 너무 적으면 Bounding Box를 생성하지 않습니다.
    if len(pixels_in_unexplored) < 100:
        logging.warning(
            f"Frontier ID {frontier.frontier_id} has too few valid pixels for cropping. "
            f"Found {len(pixels_in_unexplored)} pixels, expected at least 100."
        )
        return None

    # 이 픽셀들을 감싸는 바운딩 박스를 계산합니다.
    x_min = np.min(pixels_in_unexplored[:, 0])
    y_min = np.min(pixels_in_unexplored[:, 1])
    x_max = np.max(pixels_in_unexplored[:, 0])
    y_max = np.max(pixels_in_unexplored[:, 1])

    # 바운딩 박스가 너무 크거나 작으면 무효 처리
    box_w = x_max - x_min
    box_h = y_max - y_min

    # 바운딩 박스 유효성 검사
    if x_max <= x_min or y_max <= y_min:
        return None

    return int(x_min), int(y_min), int(x_max), int(y_max)


def resize_image(image, target_h, target_w):
    # image: np.array, h, w, c
    image = Image.fromarray(image)
    image = image.resize((target_w, target_h))
    return np.array(image)


def rgba2rgb(rgba, background=(255, 255, 255)):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, "RGBA image has 4 channels."

    rgb = np.zeros((row, col, 3), dtype="float32")
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

    a = np.asarray(a, dtype="float32") / 255.0

    R, G, B = background

    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype="uint8")


def get_pts_angle_aeqa(init_pts, init_quat):
    pts = np.asarray(init_pts)

    init_quat = quaternion.quaternion(*init_quat)
    angle, axis = quat_to_angle_axis(init_quat)
    angle = angle * axis[1] / np.abs(axis[1])

    return pts, angle


def get_pts_angle_goatbench(init_pos, init_rot):
    pts = np.asarray(init_pos)

    init_quat = quat_from_coeffs(init_rot)
    angle, axis = quat_to_angle_axis(init_quat)
    angle = angle * axis[1] / np.abs(axis[1])

    return pts, angle


def calc_agent_subtask_distance(curr_pts, viewpoints, pathfinder):
    # calculate the distance to the nearest view point
    all_distances = []
    for viewpoint in viewpoints:
        path = habitat_sim.ShortestPath()
        path.requested_start = curr_pts
        path.requested_end = viewpoint
        found_path = pathfinder.find_path(path)
        if not found_path:
            all_distances.append(np.inf)
        else:
            all_distances.append(path.geodesic_distance)
    return min(all_distances)


def get_point_from_mask(
    mask: np.ndarray,
    depth: np.ndarray,
    cam_intr: np.ndarray,
    cam_pose: np.ndarray,
    use_mask_center: bool = False
) -> Optional[np.ndarray]:
    """
    마스크와 깊이 정보로부터 3D 월드 좌표를 계산합니다.
    """
    if mask.sum() == 0:
        return None

    ys, xs = np.where(mask)
    
    if use_mask_center:
        center_x, center_y = int(xs.mean()), int(ys.mean())
    else:
        # 깊이 값이 유효한 픽셀들만 필터링
        valid_depth_mask = (depth[ys, xs] > 0.1) & (depth[ys, xs] < 5.0)
        if not np.any(valid_depth_mask):
            return None # 유효 깊이 없음
        
        ys, xs = ys[valid_depth_mask], xs[valid_depth_mask]
        
        # 유효 깊이 픽셀들의 중앙점 사용
        center_x, center_y = int(xs.mean()), int(ys.mean())

    point_depth = depth[center_y, center_x]
    if np.isnan(point_depth) or point_depth <= 0.1:
        return None

    # 2D pixel to 3D point in camera frame
    cam_fx, cam_fy = cam_intr[0, 0], cam_intr[1, 1]
    cam_cx, cam_cy = cam_intr[0, 2], cam_intr[1, 2]
    
    x_cam = (center_x - cam_cx) * point_depth / cam_fx
    y_cam = (center_y - cam_cy) * point_depth / cam_fy
    z_cam = point_depth
    
    # Camera frame (-Z forward) to Habitat frame (-Z forward, Y up)
    # The coordinate systems are aligned, but Habitat uses homogeneous coordinates.
    p_cam_h = np.array([x_cam, y_cam, z_cam, 1.0])

    # Transform to world coordinate system
    p_world = (cam_pose @ p_cam_h)[:3]
    
    return p_world
