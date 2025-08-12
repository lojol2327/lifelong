import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
from omegaconf import OmegaConf
import random
import numpy as np
import torch
import math
import time
import json
import logging
import matplotlib.pyplot as plt

import open_clip
from ultralytics import SAM, YOLOWorld

from mzson.habitat import pose_habitat_to_tsdf
from mzson.geom import get_cam_intr, get_scene_bnds
from mzson.tsdf_planner import TSDFPlanner, Frontier, SnapShot
from mzson.scene_goatbench import Scene
from mzson.utils import resize_image, calc_agent_subtask_distance, get_pts_angle_goatbench
from mzson.goatbench_utils import prepare_goatbench_navigation_goals
from src.query_vlm_goatbench import query_vlm_for_response
from mzson.logger import MZSONLogger
from mzson.siglip_itm import SigLipITM
from mzson.frontier import FrontierManager
from mzson.descriptor import DescriptorManager
from mzson.gpt_descriptor import generate_text_descriptors, generate_text_descriptors_chain


def main(cfg, start_ratio=0.0, end_ratio=1.0, split=1):
    """
    새로운 VLM 방법론을 사용한 GoatBench 평가 메인 함수
    
    Args:
        cfg: 설정 객체
        start_ratio: 시작 비율 (0.0-1.0)
        end_ratio: 종료 비율 (0.0-1.0) 
        split: 에피소드 분할 번호
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load the default concept graph config
    cfg_cg = OmegaConf.load(cfg.concept_graph_config_path)
    OmegaConf.resolve(cfg_cg)

    img_height = cfg.img_height
    img_width = cfg.img_width
    cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)

    # Depth sensor range configuration (optional)
    min_depth = None
    max_depth = None
    if hasattr(cfg, "min_depth"):
        min_depth = cfg.min_depth
    if hasattr(cfg, "max_depth"):
        max_depth = cfg.max_depth

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load dataset
    scene_data_list = os.listdir(cfg.test_data_dir)
    num_scene = len(scene_data_list)
    random.shuffle(scene_data_list)

    # split the test data by scene
    scene_data_list = scene_data_list[
        int(start_ratio * num_scene) : int(end_ratio * num_scene)
    ]
    num_episode = 0
    for scene_data_file in scene_data_list:
        with open(os.path.join(cfg.test_data_dir, scene_data_file), "r") as f:
            num_episode += len(json.load(f)["episodes"])
    logging.info(
        f"Total number of episodes: {num_episode}; Selected episodes: {len(scene_data_list)}"
    )
    logging.info(f"Total number of scenes: {len(scene_data_list)}")

    all_scene_ids = os.listdir(cfg.scene_data_path + "/train") + os.listdir(
        cfg.scene_data_path + "/val"
    )

    # load detection and segmentation models
    detection_model = YOLOWorld(cfg.yolo_model_name)
    logging.info(f"Load YOLO model {cfg.yolo_model_name} successful!")

    sam_predictor = SAM(cfg.sam_model_name)  # UltraLytics SAM
    logging.info(f"Load SAM model {cfg.sam_model_name} successful!")

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", "laion2b_s34b_b79k"  # "ViT-H-14", "laion2b_s32b_b79k"
    )
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    logging.info(f"Load CLIP model successful!")

    # Initialize the logger
    logger = MZSONLogger(
        cfg.output_dir, start_ratio, end_ratio, split, voxel_size=cfg.tsdf_grid_size
    )

    # SigLIP ITM 초기화 (메인 파이프라인에서 직접 관리)
    itm = SigLipITM(
        device=device,
        model_name=getattr(cfg, "siglip_model_name", "siglip2_base_patch16_384"),
        pretrained=getattr(cfg, "siglip_pretrained", ""),
        backend=getattr(cfg, "siglip_backend", "timm"),
    )
    frontier_mgr = FrontierManager(num_candidates_per_frontier=getattr(cfg, "num_candidates_per_frontier", 3),
                                   default_camera_tilt_deg=getattr(cfg, "camera_tilt_deg", -25.0))
    desc_mgr = DescriptorManager(similarity_threshold=getattr(cfg, "descriptor_similarity_threshold", 0.85))

    for scene_data_file in scene_data_list:
        # load goatbench data
        scene_name = scene_data_file.split(".")[0]
        scene_id = [scene_id for scene_id in all_scene_ids if scene_name in scene_id][0]
        scene_data = json.load(
            open(os.path.join(cfg.test_data_dir, scene_data_file), "r")
        )

        # select the episodes according to the split
        scene_data["episodes"] = scene_data["episodes"][split - 1 : split]
        total_episodes = len(scene_data["episodes"])

        all_navigation_goals = scene_data[
            "goals"
        ]  # obj_id to obj_data, apply for all episodes in this scene

        for episode_idx, episode in enumerate(scene_data["episodes"]):
            logging.info(f"Episode {episode_idx + 1}/{total_episodes}")
            logging.info(f"Loading scene {scene_id}")
            episode_id = episode["episode_id"]

            all_subtask_goal_types, all_subtask_goals = (
                prepare_goatbench_navigation_goals(
                    scene_name=scene_name,
                    episode=episode,
                    all_navigation_goals=all_navigation_goals,
                )
            )

            # check whether this episode has been processed
            finished_subtask_ids = list(logger.success_by_snapshot.keys())
            finished_episode_subtask = [
                subtask_id
                for subtask_id in finished_subtask_ids
                if subtask_id.startswith(f"{scene_id}_{episode_id}_")
            ]
            if len(finished_episode_subtask) >= len(all_subtask_goals):
                logging.info(f"Scene {scene_id} Episode {episode_id} already done!")
                continue

            pts, angle = get_pts_angle_goatbench(
                episode["start_position"], episode["start_rotation"]
            )

            # load scene
            try:
                del scene
            except:
                pass
            scene = Scene(
                scene_id,
                cfg,
                cfg_cg,
                detection_model,
                sam_predictor,
                clip_model,
                clip_preprocess,
                clip_tokenizer,
                device=device
            )

            # initialize TSDF planner (풀 이식: orig와 동일)
            floor_height = pts[1]
            tsdf_bnds, scene_size = get_scene_bnds(scene.pathfinder, floor_height)
            max_steps = int(math.sqrt(scene_size) * cfg.max_step_room_size_ratio)
            max_steps = max(max_steps, 50)
            
            # TSDFPlanner 초기화 (orig와 완전히 동일)
            tsdf_planner = TSDFPlanner(
                vol_bnds=tsdf_bnds,
                voxel_size=cfg.tsdf_grid_size,
                floor_height=floor_height,
                floor_height_offset=0,
                pts_init=pts,
                init_clearance=cfg.init_clearance * 2,
                save_visualization=cfg.save_visualization,
            )
            
            # 에피소드 컨텍스트 구성 (간단 dict, 별도 함수 제거)
            episode_context = {
                "start_position": pts,
                "floor_height": floor_height,
                "tsdf_bounds": tsdf_bnds,
                "visited_positions": [],
                "observations_history": [],
                "step_count": 0,
                "tsdf_planner": tsdf_planner,
            }

            episode_dir, eps_frontier_dir, eps_snapshot_dir = logger.init_episode(
                episode_id=f"{scene_id}_ep_{episode_id}"
            )

            logging.info(f"\n\nScene {scene_id} initialization successful!")

            # run questions in the scene
            global_step = -1
            for subtask_idx, (goal_type, subtask_goal) in enumerate(
                zip(all_subtask_goal_types, all_subtask_goals)
            ):
                subtask_id = f"{scene_id}_{episode_id}_{subtask_idx}"
                logging.info(
                    f"\nScene {scene_id} Episode {episode_id} Subtask {subtask_idx + 1}/{len(all_subtask_goals)}"
                )

                subtask_metadata = logger.init_subtask(
                    subtask_id=subtask_id,
                    goal_type=goal_type,
                    subtask_goal=subtask_goal,
                    pts=pts,
                    scene=scene,
                    tsdf_planner=tsdf_planner,
                )

                task_result = run_subtask(
                    subtask_metadata=subtask_metadata,
                    scene=scene,
                    episode_context=episode_context,
                    pts=pts,
                    angle=angle,
                    max_steps=max_steps,
                    cam_intr=cam_intr,
                    min_depth=min_depth,
                    max_depth=max_depth,
                    cfg=cfg,
                    logger=logger,
                    global_step=global_step,
                    tsdf_planner=tsdf_planner,
                    eps_frontier_dir=eps_frontier_dir,
                    itm=itm,
                    frontier_mgr=frontier_mgr,
                    desc_mgr=desc_mgr,
                )

                global_step = task_result.get("final_global_step", global_step)

            # save the results at the end of each episode
            logger.save_results()

            logging.info(f"Episode {episode_id} finish")
            if not cfg.save_visualization:
                os.system(f"rm -r {episode_dir}")

    logger.save_results()
    # aggregate the results from different splits into a single file
    logger.aggregate_results()

    logging.info(f"All scenes finish")


def _keep_file_nonempty():
    return None


def _detect_and_verify_object(rgb: np.ndarray, goal_texts: list, detection_model, itm: SigLipITM,
                              det_conf: float = 0.1, itm_thresh: float = 0.35) -> bool:
    """간단한 Object Goal 검증: YOLO 박스 크롭 → ITM 검증 → 임계 초과 시 True.
    goal_texts가 비어있으면 항상 False.
    """
    try:
        if not goal_texts:
            return False
        results = detection_model.predict(rgb, conf=det_conf, verbose=False)
        if not results or results[0].boxes is None or results[0].boxes.xyxy is None:
            return False
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        if boxes.size == 0:
            return False
        best = 0.0
        for (x1, y1, x2, y2) in boxes:
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(rgb.shape[1], x2); y2 = min(rgb.shape[0], y2)
            if x2 - x1 <= 1 or y2 - y1 <= 1:
                continue
            crop = rgb[y1:y2, x1:x2]
            scores = itm.image_text_scores(crop, goal_texts)
            if scores.size > 0:
                best = max(best, float(scores.max()))
        return best >= float(itm_thresh)
    except Exception:
        return False


def run_subtask(
    subtask_metadata, scene, episode_context, pts, angle, max_steps,
    cam_intr, min_depth, max_depth, cfg, logger, global_step,
    tsdf_planner, eps_frontier_dir, itm,
    frontier_mgr: FrontierManager = None,
    desc_mgr: DescriptorManager = None,
):
    """
    Args:
        subtask_metadata: 서브태스크 메타데이터 (질문, 목표 등)
        scene: 씬 객체
        episode_context: 에피소드 컨텍스트
        pts: 현재 위치
        angle: 현재 방향
        max_steps: 최대 스텝 수
        cam_intr: 카메라 내재 파라미터
        min_depth, max_depth: depth 센서 범위
        cfg: 설정 객체
        logger: 로거 객체
        global_step: 글로벌 스텝 카운터
        tsdf_planner: TSDFPlanner 인스턴스
        eps_frontier_dir: frontier 이미지 저장 디렉토리
        
    Returns:
        task_result: 태스크 실행 결과
    """
    logging.info(f"Running subtask: {subtask_metadata['question']}")
    
    # 초기 상태 설정
    current_pts = pts.copy()
    current_angle = angle
    step_count = 0
    task_success = False
    
    # TODO: 여기에 새로운 방법론의 핵심 로직을 구현하세요
    planner_cfg = cfg.planner if hasattr(cfg, "planner") else cfg
    
    while step_count < max_steps:
        step_count += 1
        global_step += 1
        
        logging.info(f"Step {step_count}/{max_steps}, Global step: {global_step}")
        
        # (1) 현재 위치에서 관찰 수집
        obs, cam_pose = scene.get_observation(current_pts, angle=current_angle)
        rgb = obs["color_sensor"]
        depth_raw = obs["depth_sensor"]
        
        # Depth 필터링 (Stretch3 spec에 맞춤)
        if min_depth is not None and max_depth is not None:
            depth = np.where(
                (depth_raw >= min_depth) & (depth_raw <= max_depth),
                depth_raw,
                np.nan
            )
        else:
            depth = depth_raw
        
        # 관찰을 히스토리에 저장
        episode_context["observations_history"].append({
            "step": step_count,
            "position": current_pts.copy(),
            "angle": current_angle,
            "rgb": rgb,
            "depth": depth,
        })
        
        # TSDF 통합 (orig와 동일)
        tsdf_planner.integrate(
            color_im=rgb,
            depth_im=depth,
            cam_intr=cam_intr,
            cam_pose=pose_habitat_to_tsdf(cam_pose),
            obs_weight=1.0,
            margin_h=int(cfg.margin_h_ratio * cfg.img_height),
            margin_w=int(cfg.margin_w_ratio * cfg.img_width),
            explored_depth=cfg.explored_depth,
        )
        
        # Frontier Map 업데이트 (FrontierManager 경유)
        frontiers = frontier_mgr.update(
            scene=scene,
            tsdf_planner=tsdf_planner,
            pts=current_pts,
            cfg_planner=planner_cfg,
            step=step_count,
            save_frontier_image=cfg.save_visualization,
            eps_frontier_dir=eps_frontier_dir,
            prompt_img_size=(cfg.prompt_h, cfg.prompt_w),
        )
        if frontiers is None:
            logging.warning("Frontier map update failed!")
        else:
            logging.info(f"Frontier map updated successfully! Found {len(frontiers)} frontiers")
            for i, frontier in enumerate(frontiers[:3]):  # 처음 3개만 로깅
                # Frontier 객체의 실제 속성 사용
                orientation = frontier.orientation
                frontier_angle = np.arctan2(orientation[1], orientation[0])  # orientation에서 각도 계산
                region_size = np.sum(frontier.region) if frontier.region is not None else 0
                logging.info(
                    f"  Frontier {i}: angle={frontier_angle:.2f}rad ({frontier_angle*180/np.pi:.1f}°), area={region_size}, pos={frontier.position}, id={frontier.frontier_id}"
                )

            # Frontier-wise candidate views 생성 → descriptor 구성 → 다양성 선택 → ITM 스코어 계산
            goal_texts = subtask_metadata.get("goal_texts", [])
            for ft in frontiers:
                # 후보 뷰 생성 + 내비게이션 가능한 위치로 스냅
                plan = frontier_mgr.plan_candidate_views(
                    scene=scene,
                    tsdf_planner=tsdf_planner,
                    current_pts=current_pts,
                    frontier=ft,
                    k=getattr(cfg, "num_candidates_per_frontier", 3),
                    standoff_m=getattr(cfg, "frontier_standoff_m", 0.6),
                    camera_tilt_rad=np.deg2rad(getattr(cfg, "camera_tilt_deg", -25.0)),
                )
                views_payload = []
                # 관측 수집
                images_for_batch = []
                for p in plan:
                    obs = scene.get_frontier_observation(p["target_pts"], p["view_dir"], camera_tilt=p["camera_tilt_rad"])
                    rgb_cand = obs["color_sensor"]
                    depth_cand = obs.get("depth_sensor")
                    images_for_batch.append(rgb_cand)
                    # 임베딩(빠른 다양성 필터용): SigLIP 이미지 임베딩 저장
                    img_emb = itm.encode_image(rgb_cand)
                    views_payload.append({
                        "view_pose": None,
                        "image_rgb": rgb_cand,
                        "depth": depth_cand,
                        "geom": {"dist_voxel": float(np.linalg.norm(ft.position[:2]))},
                        "embeddings": {"siglip_image": img_emb},
                        "tags": ["frontier_cand"],
                        "meta": {"frontier_id": ft.frontier_id, **p["meta"]},
                    })
                # descriptor 생성 및 다양성 선택
                descs = desc_mgr.build_descriptors(frontier_id=ft.frontier_id, views=views_payload, goal_ctx=None, mode="nav")
                diverse = desc_mgr.select_diverse(descs, top_n=getattr(cfg, "descriptors_per_frontier", 3))
                # 각 descriptor에 대해 ITM 스코어 산출(텍스트 목표가 있을 때)
                if goal_texts:
                    # 배치 스코어링(효율 개선)
                    imgs = [d.image_rgb for d in diverse]
                    scores_mat = np.vstack([itm.image_text_scores(img, goal_texts) for img in imgs])
                    for d, row in zip(diverse, scores_mat):
                        d.meta["itm_scores"] = row.tolist()
                # GPT 기반 텍스트 descriptor 생성 (옵션: use_chain_descriptors)
                stage = "goal" if goal_texts else "rapid"
                use_chain = bool(getattr(cfg, "use_chain_descriptors", False))
                per_frontier_n = int(getattr(cfg, "descriptors_per_frontier", 3))
                gpt_model = getattr(cfg, "gpt_model", "gpt-4o-mini")
                for d in diverse:
                    try:
                        if use_chain:
                            texts = generate_text_descriptors_chain(
                                image_rgb=d.image_rgb,
                                stage=stage,
                                top_n=per_frontier_n,
                                model=gpt_model,
                            )
                        else:
                            texts = generate_text_descriptors(
                                image_rgb=d.image_rgb,
                                stage=stage,
                                top_n=per_frontier_n,
                                model=gpt_model,
                            )
                        d.meta["gpt_descs"] = texts
                    except Exception:
                        d.meta["gpt_descs"] = []
                # frontier 객체에 저장 가능한 간단 리스트로 투영(기존 로깅과 호환)
                if not hasattr(ft, "descriptor_list") or ft.descriptor_list is None:
                    ft.descriptor_list = []
                for d in diverse:
                    itm_max = max(d.meta.get("itm_scores", []) or [0.0])
                    region_size = int(np.sum(ft.region))
                    # 거리/가시성은 placeholder; TSDFPlanner 정보 사용 시 보강
                    rank_score = 1.5 * itm_max + 0.3 * (region_size ** 0.5)
                    ft.descriptor_list.append({
                        "itm_scores": d.meta.get("itm_scores", []),
                        "gpt_descs": d.meta.get("gpt_descs", []),
                        "rank_score": rank_score,
                        "timestamp": time.time(),
                    })
                # 로깅 훅: 후보 descriptor 요약
                try:
                    logger.log_frontier_candidates(step=step_count, frontiers=[ft], descriptors=ft.descriptor_list)
                except Exception:
                    pass
        
        # (2) Frontier 기반 다음 행동 결정: 정책 스위치(ITM 기반) + tie-break
        if (tsdf_planner.max_point is None) and (tsdf_planner.target_point is None):
            mode, global_itm_max = decide_navigation_mode(frontiers, cfg)
            chosen_frontier = select_frontier(frontiers, tsdf_planner, current_pts, mode, cfg)
            if chosen_frontier is not None:
                set_ok = tsdf_planner.set_next_navigation_point(
                    choice=chosen_frontier,
                    pts=current_pts,
                    objects=scene.objects,
                    cfg=planner_cfg,
                    pathfinder=scene.pathfinder,
                )
                if not set_ok:
                    logging.info("set_next_navigation_point failed; continue exploring")
                else:
                    try:
                        logger.log_selection(frontier_id=chosen_frontier.frontier_id, mode=mode, itm_max=float(global_itm_max))
                    except Exception:
                        pass
            else:
                logging.info("No frontier available; breaking")
                break

        # (3) 한 스텝 전진
        step_vals = tsdf_planner.agent_step(
            pts=current_pts,
            angle=current_angle,
            objects=scene.objects,
            snapshots=scene.snapshots,
            pathfinder=scene.pathfinder,
            cfg=planner_cfg,
            path_points=None,
            save_visualization=cfg.save_visualization,
        )
        if step_vals[0] is None:
            logging.info("agent_step failed; breaking")
            break
        current_pts, current_angle, pts_voxel, fig, _, target_arrived = step_vals
        logger.log_step(pts_voxel=pts_voxel)
        episode_context["visited_positions"].append(current_pts.copy())
        logging.info(f"Current position: {current_pts}")

        # (4) 목표 달성 여부: viewpoint까지 거리 기반 판정
        agent_subtask_distance = calc_agent_subtask_distance(
            current_pts, subtask_metadata.get("viewpoints", []), scene.pathfinder
        )
        # Object Goal 검증(있을 경우) + 거리 조건 병행
        object_goal_hit = False
        if subtask_metadata.get("goal_type", "") == "object":
            object_goal_hit = _detect_and_verify_object(
                rgb=rgb,
                goal_texts=subtask_metadata.get("goal_texts", []),
                detection_model=scene.detection_model,
                itm=itm,
                det_conf=float(getattr(cfg, "obj_detect_conf", 0.15)),
                itm_thresh=float(getattr(cfg, "obj_verify_itm", 0.35)),
            )
        if agent_subtask_distance < cfg.success_distance or object_goal_hit:
            task_success = True
            logging.info(
                f"Success: distance {agent_subtask_distance:.2f} or object verified={object_goal_hit}!"
            )
            # 관찰 이미지 저장 (옵션)
            if cfg.save_visualization:
                obs_fin, _ = scene.get_observation(current_pts, angle=current_angle)
                rgb_fin = obs_fin["color_sensor"]
                plt.imsave(
                    os.path.join(logger.subtask_object_observe_dir, f"target.png"),
                    rgb_fin,
                )
            break
    
    # 결과 로깅
    if task_success:
        success_by_distance = True
        success_by_snapshot = False  # 본 방법론은 스냅샷 기반이 아님
        logging.info(f"Task completed successfully in {step_count} steps!")
    else:
        success_by_distance = False
        success_by_snapshot = False
        logging.info(f"Task failed after {step_count} steps")
    
    # 로거에 결과 기록
    logger.log_subtask_result(
        success_by_snapshot=success_by_snapshot,
        success_by_distance=success_by_distance,
        subtask_id=subtask_metadata.get("question_id", "unknown_subtask"),
        gt_subtask_explore_dist=subtask_metadata.get("gt_subtask_explore_dist", 0),
        goal_type=subtask_metadata.get("goal_type", "unknown"),
        n_filtered_snapshots=0,  # 새로운 방법론에서는 snapshot 사용 안함
        n_total_snapshots=0,
        n_total_frames=step_count,
    )
    
    return {
        "success": task_success,
        "final_position": current_pts,
        "final_angle": current_angle,
        "steps_taken": step_count,
        "final_global_step": global_step,
    }


def choose_best_frontier(tsdf_planner: TSDFPlanner, pts_habitat: np.ndarray):
    """
    정보 이득 기반 간단 휴리스틱으로 프론티어 선택.
    - 큰 프론티어 선호(+)
    - 가까운 프론티어 선호(+)
    """
    if not tsdf_planner.frontiers:
        return None

    # 현재 위치를 보xel 좌표로 변환
    pts_normal = tsdf_planner.pos_habitat_to_normal(pts_habitat)
    cur_voxel = tsdf_planner.normal2voxel(pts_normal)

    best_score = -np.inf
    best_ft = None
    for ft in tsdf_planner.frontiers:
        region_size = float(np.sum(ft.region))
        dist = np.linalg.norm(ft.position[:2] - cur_voxel[:2]) + 1e-6
        score = region_size / dist  # 간단한 점수: 면적 대비 거리
        if score > best_score:
            best_score = score
            best_ft = ft
    return best_ft


def decide_navigation_mode(frontiers, cfg):
    """간단 ITM 기반 모드 결정. frontier.descriptor_list의 itm_scores 최대값으로 판단.
    - Goal-Oriented: max_itm >= theta_high (default 0.35)
    - Rapid Exploration: max_itm < theta_low (default 0.15)
    - 중간은 탐욕 점수 혼합
    """
    theta_low = float(getattr(cfg, "itm_theta_low", 0.15))
    theta_high = float(getattr(cfg, "itm_theta_high", 0.35))
    global_max = 0.0
    for ft in frontiers or []:
        for d in getattr(ft, "descriptor_list", []) or []:
            scores = d.get("itm_scores", [])
            if scores:
                global_max = max(global_max, max(scores))
    if global_max >= theta_high:
        return "goal_oriented", global_max
    if global_max < theta_low:
        return "rapid_explore", global_max
    return "hybrid", global_max


def select_frontier(frontiers, tsdf_planner: TSDFPlanner, pts_habitat: np.ndarray, mode: str, cfg) -> Frontier | None:
    """모드에 따른 프론티어 선택. 동점시는 거리/면적 tie-break.
    - goal_oriented: descriptor_list의 itm_scores 최대가 큰 프론티어 우선
    - rapid_explore: 면적/거리 휴리스틱(score = area/dist)
    - hybrid: 0.5*goal + 0.5*explore 혼합
    """
    if not frontiers:
        return None

    pts_normal = tsdf_planner.pos_habitat_to_normal(pts_habitat)
    cur_voxel = tsdf_planner.normal2voxel(pts_normal)

    def area(ft):
        return float(np.sum(ft.region)) if getattr(ft, "region", None) is not None else 0.0

    def dist(ft):
        return float(np.linalg.norm(ft.position[:2] - cur_voxel[:2]) + 1e-6)

    def itm_max(ft):
        m = 0.0
        for d in getattr(ft, "descriptor_list", []) or []:
            scores = d.get("itm_scores", [])
            if scores:
                m = max(m, max(scores))
        return m

    best, best_score = None, -1e9
    for ft in frontiers:
        a = area(ft)
        d = dist(ft)
        g = itm_max(ft)
        explore_score = a / d
        goal_score = g
        if mode == "goal_oriented":
            s = goal_score
        elif mode == "rapid_explore":
            s = explore_score
        else:
            s = 0.5 * goal_score + 0.5 * explore_score
        # tie-break: 면적 우선, 다음 거리
        key = (s, a, -d)
        if key > (best_score, 0, 0):
            best, best_score = ft, s
    return best

def extract_frontier_descriptors(
    tsdf_planner: TSDFPlanner,
    scene: Scene,
    pts: np.ndarray,
    goal_texts: list,
    itm: SigLipITM,
    max_per_frontier: int = 3,
):
    """
    각 frontier마다 N개의 descriptor를 생성/업데이트.
    - Goal-oriented: SigLIP ITM(image x goal_texts)
    - Navigation-oriented: region size, distance, visibility
    중복을 줄이기 위해 frontier 이미지(feature) 내에서 k-중심 근사로 샘플링(간이: 상이한 crop/tilt 후보 중 상위 비상관 예시 선택) [placeholder].
    결과는 frontier 객체에 `descriptor_list`로 저장.
    """
    if not hasattr(tsdf_planner, "frontiers"):
        return
    for frontier in tsdf_planner.frontiers:
        # frontier.feature: resized rgb image (prompt size)
        rgb = frontier.feature
        if rgb is None:
            # 필요시 시점 정렬 후 관측 취득
            pos_habitat = tsdf_planner.voxel2habitat(frontier.position)
            view_frontier_direction = np.array([
                pos_habitat[0] - pts[0], 0.0, pos_habitat[2] - pts[2]
            ])
            obs = scene.get_frontier_observation(pts, view_frontier_direction)
            rgb = obs["color_sensor"]

        # Goal-oriented ITM scores
        itm_scores = itm.image_text_scores(rgb, goal_texts) if len(goal_texts) > 0 else np.array([])

        # Navigation-oriented metrics
        region_size = int(np.sum(frontier.region))
        # distance in voxel units from current pose
        pts_normal = tsdf_planner.pos_habitat_to_normal(pts)
        cur_voxel = tsdf_planner.normal2voxel(pts_normal)
        distance = float(np.linalg.norm(frontier.position[:2] - cur_voxel[:2]))
        visibility = float(np.sum(tsdf_planner.occupied_map_camera[frontier.region] == 0)) if tsdf_planner.occupied_map_camera is not None else 0.0

        descriptor = {
            "itm_scores": itm_scores.tolist() if itm_scores.size > 0 else [],
            "region_size": region_size,
            "distance": distance,
            "visibility": visibility,
            "timestamp": time.time(),
        }

        # Maintain a list and avoid heavy correlation: keep top-K by a simple score and deduplicate by score distance
        if not hasattr(frontier, "descriptor_list") or frontier.descriptor_list is None:
            frontier.descriptor_list = []

        # simple scalar for ranking: prefer higher itm (max), larger visibility/region, shorter distance
        itm_max = max(descriptor["itm_scores"]) if descriptor["itm_scores"] else 0.0
        descriptor["rank_score"] = (
            1.5 * itm_max + 0.5 * (visibility / (region_size + 1e-6)) + 0.3 * (region_size ** 0.5) - 0.2 * distance
        )

        # append and prune with diversity: ensure rank score differs by epsilon
        eps = 0.05
        exists_similar = any(abs(d["rank_score"] - descriptor["rank_score"]) < eps for d in frontier.descriptor_list)
        if not exists_similar:
            frontier.descriptor_list.append(descriptor)
            # keep top-K
            frontier.descriptor_list = sorted(frontier.descriptor_list, key=lambda d: d["rank_score"], reverse=True)[:max_per_frontier]


if __name__ == "__main__":
    # Get config path
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path", default="cfg/eval_stretch3_mzson.yaml", type=str)
    parser.add_argument("--start_ratio", help="start ratio", default=0.0, type=float)
    parser.add_argument("--end_ratio", help="end ratio", default=1.0, type=float)
    parser.add_argument("--split", help="which episode", default=1, type=int)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)

    # Set up logging
    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)  # recursive
    logging_path = os.path.join(
        str(cfg.output_dir),
        f"log_{args.start_ratio:.2f}_{args.end_ratio:.2f}_{args.split}.log",
    )

    os.system(f"cp {args.cfg_file} {cfg.output_dir}")

    class ElapsedTimeFormatter(logging.Formatter):
        def __init__(self, fmt=None, datefmt=None):
            super().__init__(fmt, datefmt)
            self.start_time = time.time()

        def formatTime(self, record, datefmt=None):
            elapsed_seconds = record.created - self.start_time
            hours, remainder = divmod(elapsed_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

    # Set up the logging format
    formatter = ElapsedTimeFormatter(fmt="%(asctime)s - %(message)s")

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    # Set the custom formatter
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)

    # run
    logging.info(f"***** Running {cfg.exp_name} with new VLM methodology *****")
    main(cfg, start_ratio=args.start_ratio, end_ratio=args.end_ratio, split=args.split) 