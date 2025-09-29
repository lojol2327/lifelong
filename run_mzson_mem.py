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
from PIL import Image, ImageDraw
from typing import Dict, Optional, Any, Tuple, List
from dataclasses import dataclass, field
import habitat_sim
import cv2 # Added for cv2.resize

from ultralytics import SAM, YOLOWorld

from mzson.habitat import (
    pose_habitat_to_tsdf, pos_habitat_to_normal, pos_normal_to_habitat
)
from mzson.geom import get_cam_intr, get_scene_bnds
from mzson.tsdf_planner import TSDFPlanner
from mzson.scene_goatbench import Scene
from mzson.utils import (
    resize_image, calc_agent_subtask_distance, get_pts_angle_goatbench,
    get_point_from_mask
)
from mzson.goatbench_utils import prepare_goatbench_navigation_goals
from mzson.logger import MZSONLogger
from mzson.siglip_itm import SigLipITM
from mzson.descriptor_extractor import DescriptorExtractor, DescriptorExtractorConfig
from mzson.data_models import Frontier


class ElapsedTimeFormatter(logging.Formatter):
    """Formats log time to be elapsed time from the start."""
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        self.start_time = time.time()

    def formatTime(self, record, datefmt=None):
        elapsed_seconds = record.created - self.start_time
        hours, remainder = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


@dataclass
class AnalyzedObservation:
    """VLM으로 분석된 관측 정보를 저장하는 데이터 클래스."""
    name: str
    map_pos: Tuple[int, int]
    angle: float
    descriptors: List[str]
    rgb: np.ndarray = field(repr=False) # For content similarity checks
    key_objects: List[str] = field(default_factory=list)
    best_itm_score: float = 0.0 # No longer a primary score, but kept for data structure consistency
    timestamp: float = field(default_factory=time.time)


def _get_memory_candidates_for_goal(
    app_state: "AppState",
    goal: Any,
    goal_type: str,
    current_pts: np.ndarray,
    tsdf_planner: TSDFPlanner,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Returns up to top_k candidates from long-term observation memory for warm-starting a subtask.
    Each candidate contains: {"map_pos": Tuple[int,int], "angle": float, "score": float}.
    Applies a similarity threshold and enforces a minimum spatial separation between candidates.
    """
    voxel_size = getattr(tsdf_planner, "_voxel_size", 0.1)
    min_sep_m = float(getattr(app_state.cfg, "memory_candidate_min_separation_m", 1.0))
    tau_text = float(getattr(app_state.cfg, "memory_select_text_threshold", 0.35))

    def _sep_ok(cands: List[Dict[str, Any]], new_pos: Tuple[int, int]) -> bool:
        for c in cands:
            d = np.linalg.norm(np.array(c["map_pos"]) - np.array(new_pos)) * voxel_size
            if d < min_sep_m:
                return False
        return True

    candidates: List[Dict[str, Any]] = []

    if goal_type == "image":
        # TODO: image-image retrieval (deferred)
        return candidates

    # Build target objects for object/description
    if goal_type == "object":
        target_objects: List[str] = goal if isinstance(goal, list) else [goal]
    elif goal_type == "description":
        # description은 이미 run_subtask에서 단일 main target으로 정제됨
        target_objects = goal if isinstance(goal, list) else [goal]
    else:
        target_objects = []

    # Normalize targets
    target_objects = [t for t in target_objects if isinstance(t, str) and len(t.strip()) > 0]
    if not target_objects:
        return candidates

    # Score each memory entry
    mem_scores: List[Tuple[float, bool, AnalyzedObservation]] = []
    for mem in app_state.observation_memory:
        # key_objects 우선 여부
        key_hit = False
        if mem.key_objects:
            inter = set([k.lower() for k in mem.key_objects]) & set([t.lower() for t in target_objects])
            key_hit = len(inter) > 0

        # ITM text-text 유사도 (타깃 텍스트 vs mem.descriptors)
        sim = 0.0
        try:
            if mem.descriptors:
                for t in target_objects:
                    scores = app_state.itm.text_text_scores(t, mem.descriptors)
                    if scores is not None and scores.size > 0:
                        sim = max(sim, float(scores.max()))
        except Exception:
            pass

        if sim >= tau_text:
            mem_scores.append((sim, key_hit, mem))

    # 우선순위: key_hit=True 먼저, 그 다음 sim 내림차순
    mem_scores.sort(key=lambda x: (not x[1], -x[0]))

    # Top-K with spatial separation
    picked: List[Dict[str, Any]] = []
    for sim, _key, mem in mem_scores:
        pos = tuple(int(v) for v in mem.map_pos)
        if _sep_ok(picked, pos):
            picked.append({"map_pos": pos, "angle": float(mem.angle), "score": sim})
        if len(picked) >= top_k:
            break

    return picked


@dataclass
class AppState:
    """모든 주요 컴포넌트와 설정을 담는 데이터 클래스."""
    cfg: OmegaConf
    device: torch.device
    logger: MZSONLogger
    itm: SigLipITM
    desc_extractor: DescriptorExtractor
    detection_model: YOLOWorld
    sam_predictor: SAM
    cam_intr: np.ndarray
    min_depth: Optional[float]
    max_depth: Optional[float]
    observation_memory: List[AnalyzedObservation] = field(default_factory=list)


def setup(cfg_file: str, start_ratio: float, end_ratio: float, split: int) -> AppState:
    """설정 로딩, 로깅, 모델 초기화 등 모든 준비 작업을 수행합니다."""
    cfg = OmegaConf.load(cfg_file)
    OmegaConf.resolve(cfg)
    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    os.makedirs(cfg.output_dir, exist_ok=True)
    if os.path.abspath(cfg_file) != os.path.abspath(os.path.join(cfg.output_dir, os.path.basename(cfg_file))):
        os.system(f"cp {cfg_file} {cfg.output_dir}")

    # 로깅 설정
    formatter = ElapsedTimeFormatter(fmt="%(asctime)s - %(message)s")
    root_logger = logging.getLogger()
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    root_logger.setLevel(logging.INFO)

    # 1. 콘솔 핸들러
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    # 2. 메인 로그 파일 핸들러 (초기 설정 및 최종 결과용)
    logging_path = os.path.join(str(cfg.output_dir), f"log_{start_ratio:.2f}_{end_ratio:.2f}_{split}.log")
    main_file_handler = logging.FileHandler(logging_path, mode="w")
    main_file_handler.setFormatter(formatter)
    root_logger.addHandler(main_file_handler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델 초기화
    detection_model = YOLOWorld(cfg.yolo_model_name)
    sam_predictor = SAM(cfg.sam_model_name)
    itm = SigLipITM(
        device=device,
        model_name=cfg.siglip_model_name,
        pretrained=cfg.siglip_pretrained,
        backend=cfg.siglip_backend,
    )
    desc_extractor_config = DescriptorExtractorConfig(
        use_chain_descriptors=cfg.use_chain_descriptors,
        gpt_model=cfg.gpt_model,
        n_descriptors=cfg.descriptors_per_frontier,
    )
    desc_extractor = DescriptorExtractor(itm, desc_extractor_config)
    logger = MZSONLogger(cfg.output_dir, start_ratio, end_ratio, split, voxel_size=cfg.tsdf_grid_size)

    cam_intr = get_cam_intr(cfg.hfov, cfg.img_height, cfg.img_width)
    min_depth = cfg.min_depth if hasattr(cfg, "min_depth") else None
    max_depth = cfg.max_depth if hasattr(cfg, "max_depth") else None

    logging.info("Setup complete.")
    
    return AppState(
        cfg=cfg, device=device, logger=logger, itm=itm,
        desc_extractor=desc_extractor, detection_model=detection_model,
        sam_predictor=sam_predictor, cam_intr=cam_intr, min_depth=min_depth, max_depth=max_depth,
        observation_memory=[], # Explicitly initialize long-term memory
    )


def run_evaluation(app_state: AppState, start_ratio: float, end_ratio: float, split: int, scene_id: Optional[str] = None):
    """준비된 상태를 바탕으로 데이터셋을 순회하며 평가를 실행합니다."""
    cfg = app_state.cfg
    cfg_cg = OmegaConf.load(cfg.concept_graph_config_path)
    OmegaConf.resolve(cfg_cg)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    scene_data_list = os.listdir(cfg.test_data_dir)
    
    if scene_id:
        target_scene_file = None
        # scene_id (e.g., '00871-VBzV5z6i1WS') contains the scene_name (e.g., 'VBzV5z6i1WS')
        for f in scene_data_list:
            scene_name_from_file = f.split(".")[0]
            if scene_name_from_file in scene_id:
                target_scene_file = f
                break
        
        if target_scene_file:
            scene_data_list = [target_scene_file]
            logging.info(f"Running for single scene: {scene_id}")
        else:
            logging.error(f"Scene ID {scene_id} not found in {cfg.test_data_dir}. Exiting.")
            return
    else:
        num_scene = len(scene_data_list)
        random.shuffle(scene_data_list)
        scene_data_list = scene_data_list[int(start_ratio * num_scene):int(end_ratio * num_scene)]
    
    num_episode = sum(len(json.load(open(os.path.join(cfg.test_data_dir, f), "r"))["episodes"]) for f in scene_data_list)
    logging.info(f"Total episodes: {num_episode}; Selected scenes: {len(scene_data_list)}")

    all_scene_ids = os.listdir(cfg.scene_data_path + "/train") + os.listdir(cfg.scene_data_path + "/val")

    formatter = ElapsedTimeFormatter(fmt="%(asctime)s - %(message)s")
    
    root_logger = logging.getLogger()
    # 메인 파일 핸들러를 찾아서 evaluation 중에는 잠시 제거했다가 끝나고 다시 추가합니다.
    main_file_handler = None
    for handler in root_logger.handlers:
        # 'log_'로 시작하는 파일명을 가진 핸들러를 메인 핸들러로 간주합니다.
        if isinstance(handler, logging.FileHandler) and "log_" in os.path.basename(handler.baseFilename):
             main_file_handler = handler
             break

    for scene_data_file in scene_data_list:
        scene_name = scene_data_file.split(".")[0]
        scene_id = [sid for sid in all_scene_ids if scene_name in sid][0]
        
        scene_file_handler = None
        try:
            # --- Set up scene-specific file logging ---
            if main_file_handler:
                root_logger.removeHandler(main_file_handler)

            # 씬 결과 폴더 내에 로그 파일 생성
            log_dir = os.path.join(app_state.cfg.output_dir, scene_id)
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"{scene_id}.log")
            scene_file_handler = logging.FileHandler(log_path, mode="w")
            scene_file_handler.setFormatter(formatter)
            root_logger.addHandler(scene_file_handler)

            scene_data = json.load(open(os.path.join(cfg.test_data_dir, scene_data_file), "r"))
            scene_data["episodes"] = scene_data["episodes"][split - 1 : split]
            all_navigation_goals = scene_data["goals"]

            scene = None # 루프 시작 전 scene 변수 초기화
            for episode_idx, episode in enumerate(scene_data["episodes"]):
                episode_id = episode["episode_id"]
                logging.info(f"Starting Episode {episode_idx + 1}/{len(scene_data['episodes'])} in scene {scene_id}")

                # 이전 scene 객체가 있다면 명시적으로 종료
                if scene is not None:
                    scene.close()
                    del scene

                all_subtask_goal_types, all_subtask_goals = prepare_goatbench_navigation_goals(
                    scene_name=scene_name, episode=episode, all_navigation_goals=all_navigation_goals
                )

                finished_subtask_ids = list(app_state.logger.success_by_snapshot.keys())
                finished_episode_subtask = [sid for sid in finished_subtask_ids if sid.startswith(f"{scene_id}_{episode_id}_")]
                if len(finished_episode_subtask) >= len(all_subtask_goals):
                    logging.info(f"Scene {scene_id} Episode {episode_id} already done!")
                    continue

                pts, angle = get_pts_angle_goatbench(episode["start_position"], episode["start_rotation"])

                scene = Scene(
                    scene_id, cfg, cfg_cg, app_state.detection_model, app_state.sam_predictor,
                    vlm_model=app_state.itm, device=app_state.device,
                )

                floor_height = pts[1]
                tsdf_bnds, scene_size = get_scene_bnds(scene.pathfinder, floor_height)
                max_steps = max(int(math.sqrt(scene_size) * cfg.max_step_room_size_ratio), 50)
                
                tsdf_planner = TSDFPlanner(
                    vol_bnds=tsdf_bnds, voxel_size=cfg.tsdf_grid_size, floor_height=floor_height,
                    floor_height_offset=0, pts_init=pts, init_clearance=cfg.init_clearance * 2,
                    save_visualization=cfg.save_visualization,
                )

                episode_context = {
                    "start_position": pts, "floor_height": floor_height, "tsdf_bounds": tsdf_bnds,
                    "visited_positions": [], "observations_history": [], "step_count": 0,
                    "tsdf_planner": tsdf_planner,
                    "observation_memory": [],
                    "subtask_observation_cache": {}, # Initialize short-term cache
                    "subtask_id": f"{scene_id}_{episode_id}", # Initialize subtask_id
                    "subtask_goal_str": "N/A", # Initialize subtask_goal_str
                    "target_objects": [], # Initialize target_objects
                }

                app_state.logger.init_episode(scene_id=scene_id, episode_id=episode_id)

                global_step = 0 # FIX: Changed from -1 to 0 to align with 1-based cnt_step
                for subtask_idx, (goal_type, subtask_goal) in enumerate(
                    zip(all_subtask_goal_types, all_subtask_goals)
                ):
                    subtask_id = f"{scene_id}_{episode_id}_{subtask_idx}"

                    # 이미 처리된 서브태스크는 건너뜁니다.
                    if subtask_id in app_state.logger.success_by_snapshot:
                        logging.info(f"Subtask {subtask_idx + 1}/{len(all_subtask_goals)} already done!")
                        continue

                    # subtask_goal 파싱 및 메타데이터 생성은 logger.init_subtask가 담당.
                    # 베이스라인 코드와 동일한 구조.
                    subtask_metadata = app_state.logger.init_subtask(
                        subtask_id=subtask_id,
                        goal_type=goal_type,
                        subtask_goal=subtask_goal,
                        pts=pts,
                        scene=scene,
                        tsdf_planner=tsdf_planner,
                    )

                    question = subtask_metadata.get("question", "N/A")
                    logging.info(f"\nSubtask {subtask_idx + 1}/{len(all_subtask_goals)}: {question}")
                    
                    task_result = run_subtask(
                        app_state=app_state,
                        subtask_id=subtask_id,
                        subtask_metadata=subtask_metadata,
                        scene=scene,
                        episode_context=episode_context,
                        pts=pts,
                        angle=angle,
                        max_steps=max_steps,
                        global_step=global_step,
                        tsdf_planner=tsdf_planner,
                    )
                    global_step = task_result.get("final_global_step", global_step)
                    
                app_state.logger.save_results()
                if not cfg.save_visualization:
                    os.system(f"rm -r {app_state.logger.get_episode_dir()}")

            # 마지막 에피소드가 끝난 후 scene 정리
            if scene is not None:
                scene.close()
                del scene

            # Scene 종료 시 observation memory 초기화
            logging.info(f"Scene {scene_id} finished. Clearing observation memory.")
            app_state.observation_memory.clear()

        finally:
            # --- 씬 로깅 핸들러 정리 및 메인 핸들러 복원 ---
            if scene_file_handler:
                scene_file_handler.close()
                root_logger.removeHandler(scene_file_handler)
            if main_file_handler:
                root_logger.addHandler(main_file_handler)

    app_state.logger.save_results()
    app_state.logger.aggregate_results()


def _detect_and_verify_goal(
    rgb: np.ndarray,
    depth: np.ndarray,
    goal: Any,
    goal_type: str,
    detection_model: YOLOWorld,
    sam_predictor: SAM,
    itm: SigLipITM,
    tsdf_planner: TSDFPlanner,
    cam_intr: np.ndarray,
    cam_pose: np.ndarray,
    cfg: OmegaConf,
) -> tuple[bool, Optional[tuple[int, int]]]:
    """
    Checks if the goal is visible in the current observation (rgb).
    This function only handles object goals.
    """
    if goal_type != "object":
        return False, None

    # --- PERFORMANCE OPTIMIZATION ---
    obj_det_conf = float(getattr(cfg, "direct_goal_detect_conf", 0.4))
    verification_thresh = float(getattr(cfg, "goal_verification_threshold", 0.45))
    
    resized_rgb = resize_image(rgb, target_h=640, target_w=640)
    # No need to resize depth here as it's only used with the upscaled mask at the end

    # 1. Detect candidate objects with YOLO
    try:
        if not goal: return False, None
        
        # FIX: Handle different goal types (text vs image)
        if hasattr(goal, 'size'):  # PIL Image object
            # For image goals, we can't use YOLO's set_classes with image
            # Instead, we'll use a generic object detection approach
            logging.debug("Image goal detected, using generic object detection")
            # Don't set specific classes for image goals
            results = detection_model.predict(resized_rgb, conf=obj_det_conf, verbose=False)
        else:
            # For text goals, set the specific classes
            detection_model.set_classes(goal)
            results = detection_model.predict(resized_rgb, conf=obj_det_conf, verbose=False)
            
        if not results or results[0].boxes is None or results[0].boxes.xyxy is None:
            return False, None
        boxes = results[0].boxes.xyxy.cpu().numpy()

        # 2. Verify with SAM and ITM
        best_score = 0.0
        best_mask = None
        for box in boxes:
            masks = sam_predictor.predict(resized_rgb, bboxes=box, verbose=False)[0].masks.data
            if masks.shape[0] == 0: continue
            mask = masks[0].cpu().numpy().astype(bool)
            
            masked_rgb_np = resized_rgb.copy()
            masked_rgb_np[~mask] = 0
            masked_rgb = Image.fromarray(masked_rgb_np)

            current_goal = goal[0] if isinstance(goal, list) else goal
            
            # FIX: Call the correct ITM scoring function based on the goal's type.
            if hasattr(current_goal, 'size'):  # Heuristic to check for a PIL Image object
                scores = itm.image_image_scores(masked_rgb, [current_goal])
            else:
                scores = itm.image_text_scores(masked_rgb, current_goal)

            if scores is not None and scores.size > 0 and scores.max() > best_score:
                best_score = float(scores.max())
                best_mask = mask
        
        # 3. Get 3D point from the mask and return
        if best_score >= verification_thresh and best_mask is not None:
            original_sized_mask = cv2.resize(best_mask.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            world_pos = get_point_from_mask(original_sized_mask, depth, cam_intr, cam_pose)
            
            if world_pos is None: return False, None
            
            voxel_pos = tsdf_planner.normal2voxel(pos_habitat_to_normal(world_pos))
            goal_desc = f"image goal" if hasattr(goal, 'size') else f"'{goal[0] if isinstance(goal, list) else goal}'"
            logging.info(f"Object goal {goal_desc} verified with score {best_score:.3f}, target: {voxel_pos}")
            return True, voxel_pos

    except Exception as e:
        logging.error(f"Error in _detect_and_verify_goal for goal '{goal}': {e}", exc_info=True)
        
    return False, None


def run_subtask(
    app_state: AppState,
    subtask_id: str,
    subtask_metadata, scene, episode_context, pts, angle, max_steps,
    global_step,
    tsdf_planner: TSDFPlanner,
):
    """
    하나의 서브태스크(예: '소파로 이동')를 자율적으로 수행합니다.
    내부적으로 동적 관측 메모리를 활용한 계층적 탐색 전략을 사용합니다.
    """
    # Initialize the observation history for this specific subtask
    episode_context["subtask_full_observation_history"] = {}
    episode_context["frontier_score_cache"] = {} # Initialize score cache for this subtask

    # Ensure no carry-over target from previous subtask
    try:
        tsdf_planner.max_point = None
        tsdf_planner.target_point = None
    except Exception:
        pass

    # Optionally clear memory at the start of each subtask
    # episode_context["observation_memory"].clear() 

    goal = subtask_metadata.get("goal")
    goal_type = subtask_metadata.get("goal_type")
    original_goal_str = subtask_metadata.get("question", subtask_id)

    # --- 목표 분석 및 target_objects 추출 (run_mzson_nomem.py의 검증된 로직 사용) ---
    target_objects = []
    if goal_type == "object":
        target_objects = goal if isinstance(goal, list) else [goal]
    elif goal_type == "description":
        logging.info(f"Goal is a description. Extracting target objects...")
        main_targets, _context_objects = app_state.desc_extractor.extract_target_objects_from_description(goal)
        target_objects = main_targets
    elif goal_type == "image":
        logging.info("Goal is an image. Extracting keywords for ITM scoring...")
        target_objects = app_state.desc_extractor.extract_keywords_from_image(goal)
        logging.info(f"Extracted keywords from image goal: {target_objects}")
    
    logging.info(f"Refined Target Objects: {target_objects}")

    # Store refined targets and original goal string
    episode_context["target_objects"] = target_objects if isinstance(target_objects, list) else []
    episode_context["subtask_goal_str"] = original_goal_str

    # 초기 상태 설정
    current_pts = pts.copy()
    current_angle = angle
    success_by_distance = False # task_success 대신 이 변수를 사용해 최종 성공 여부를 기록합니다.

    # --- Phase 1: Memory warm-start (Top-k candidates) ---
    top_k = int(getattr(app_state.cfg, "memory_candidates_k", 3))
    # For image goals, use extracted keywords as text targets for memory lookup
    mem_goal_type = "object" if goal_type == "image" else goal_type
    mem_goal = target_objects if mem_goal_type in ["object", "description"] else target_objects
    mem_candidates = _get_memory_candidates_for_goal(
        app_state=app_state,
        goal=mem_goal,
        goal_type=mem_goal_type,
        current_pts=current_pts,
        tsdf_planner=tsdf_planner,
        top_k=top_k,
    )

    # --- 단계별 탐색 전략을 위한 상태 변수 ---
    # 'memory_driven': Phase 1 - 기억 기반 우선 탐색
    # 'exploration': Phase 2 - 의미론적 통합 탐색 (Fallback)
    navigation_mode = "exploration"  
    
    if mem_candidates and mem_candidates[0]["score"] >= app_state.cfg.memory_navigation_threshold:
        best_mem_candidate = mem_candidates[0]
        logging.info(
            f"Phase 1 START: High-confidence memory found (score: {best_mem_candidate['score']:.3f}). "
            f"Attempting direct navigation to {best_mem_candidate['map_pos']}."
        )
        set_ok = tsdf_planner.set_target_point_from_observation(best_mem_candidate["map_pos"])
        if set_ok:
            navigation_mode = "memory_driven"
            episode_context["warmstart_angle"] = best_mem_candidate["angle"]
        else:
            logging.warning("Phase 1 FAILED: Could not set path to memory candidate. Switching to Phase 2.")
    else:
        logging.info("Phase 1 SKIP: No high-confidence memory found. Starting with Phase 2 (Exploration).")


    tracked_goal_path = None

    # run steps
    cnt_step = 0
    while cnt_step < max_steps:
        cnt_step += 1
        global_step += 1
        
        candidates_info = None # 시각화 정보를 위해 루프 시작 시 초기화

        # --- Phase 2: 계층적 전략 수행 ---
        if navigation_mode == "exploration":
            # [REVISED LOGIC]
            # 목표 지속성(Goal Persistence)을 적용합니다.
            # 장기 목표(max_point)가 없을 때만 새로운 목표를 선택합니다.
            if tsdf_planner.max_point is None:
                logging.info("No long-term target. Stop, Scan, and Select.")

                # --- 1. 항상 관측 및 맵 업데이트 수행 ---
                current_step_observations = _observe_and_update_maps(
                    scene=scene, tsdf_planner=tsdf_planner, current_pts=current_pts,
                    current_angle=current_angle, cnt_step=cnt_step, cfg=app_state.cfg,
                    cam_intr=app_state.cam_intr, min_depth=app_state.min_depth,
                    max_depth=app_state.max_depth, eps_frontier_dir=app_state.logger.get_frontier_dir(),
                )
                
                for obs in current_step_observations:
                    episode_context["subtask_full_observation_history"][obs['name']] = obs

                for frontier in tsdf_planner.frontiers:
                    if frontier.source_observation_name is None:
                        relevant_obs_list = get_relevant_observations(
                            frontier, current_step_observations,
                            app_state.cfg.relevant_view_angle_threshold_deg
                        )
                        if len(relevant_obs_list) > 0:
                            frontier.source_observation_name = relevant_obs_list[0]['name']
                            frontier.cam_pose = relevant_obs_list[0]['cam_pose']
                            frontier.depth = relevant_obs_list[0]['depth']

                # --- 2. 장기 기억 업데이트 (메모리 버전의 핵심 로직) ---
                frontier_source_obs_names = {f.source_observation_name for f in tsdf_planner.frontiers if f.source_observation_name}
                observations_to_analyze = [
                    obs for obs in current_step_observations 
                    if obs['name'] in frontier_source_obs_names
                ]
                logging.info(f"Analyzing {len(observations_to_analyze)} new frontier source images for semantic memory.")
                _analyze_and_store_semantic_memory(
                    observations_to_analyze, episode_context, app_state
                )

                # --- 3. 캐시 업데이트 ---
                active_frontier_ids = {f.frontier_id for f in tsdf_planner.frontiers}
                required_obs_names = {f.source_observation_name for f in tsdf_planner.frontiers if f.source_observation_name}
                episode_context["subtask_observation_cache"] = {
                    name: obs for name, obs in episode_context["subtask_full_observation_history"].items() if name in required_obs_names
                }
                current_cache = episode_context.get("frontier_score_cache", {})
                episode_context["frontier_score_cache"] = {
                    fid: score_data for fid, score_data in current_cache.items() if fid in active_frontier_ids
                }

                # Fallback: force-link frontier to nearest observation if none linked
                if not required_obs_names and tsdf_planner.frontiers:
                    if episode_context["subtask_full_observation_history"]:
                        any_obs = next(iter(episode_context["subtask_full_observation_history"].values()))
                        for frontier in tsdf_planner.frontiers:
                            if frontier.source_observation_name is None:
                                frontier.source_observation_name = any_obs['name']
                                frontier.cam_pose = any_obs['cam_pose']
                                frontier.depth = any_obs['depth']
                        episode_context["subtask_observation_cache"][any_obs['name']] = any_obs

                # --- 4. 새로운 목표 선택 (통합된 단일 호출) ---
                logging.info("Using unified target selection (frontiers + memory)...")
                selection_dir = os.path.join(app_state.logger.subtask_dir, "selection")
                chosen_target, candidates_info = _select_next_target(
                    app_state=app_state, episode_context=episode_context,
                    current_pts=current_pts,
                    goal=goal,
                    goal_type=goal_type,
                    original_goal_str=original_goal_str,
                    tsdf_planner=tsdf_planner,
                    selection_dir=selection_dir,
                    target_objects=target_objects,
                    mem_candidates=mem_candidates, # Pass remaining memory candidates
                )
                if not chosen_target:
                    logging.warning("Failed to select a new target. Will re-scan next step.")
                # 새로운 목표가 선택되면 설정
                if chosen_target:
                    # The returned choice can be a Frontier or an AnalyzedObservation (from memory)
                    if isinstance(chosen_target, Frontier):
                        set_ok = tsdf_planner.set_next_navigation_point(
                            choice=chosen_target,
                            pts=current_pts,
                            objects=scene.objects,
                            cfg=app_state.cfg.planner,
                            pathfinder=scene.pathfinder
                        )
                    elif isinstance(chosen_target, AnalyzedObservation):
                            # For memory candidates, we directly set the target point
                        set_ok = tsdf_planner.set_target_point_from_observation(chosen_target.map_pos)
                        if set_ok:
                            episode_context["warmstart_angle"] = chosen_target.angle
                    else:
                        set_ok = False
                        logging.error(f"Unknown target type from _select_next_target: {type(chosen_target)}")

                    if not set_ok:
                        logging.warning("Failed to set navigation point for newly chosen target. Will retry next step.")
                        tsdf_planner.max_point = None
                        tsdf_planner.target_point = None
            
            # 장기 목표는 있지만 단기 길목이 없다면 (예: 중간 길목에 도착한 후),
            # 장기 목표를 향한 다음 단기 길목을 계산합니다.
            elif tsdf_planner.target_point is None:
                logging.info(f"Committed target exists. Calculating next waypoint.")
                set_ok = tsdf_planner.set_next_navigation_point(
                    choice=tsdf_planner.max_point,
                    pts=current_pts,
                    objects=scene.objects,
                    cfg=app_state.cfg.planner,
                    pathfinder=scene.pathfinder
                )
                # 경로 계산 실패 시, 장기 목표를 포기하고 다음 스텝에서 재평가합니다.
                if not set_ok:
                    logging.warning("Failed to find path to committed target. Clearing for re-evaluation.")
                    tsdf_planner.max_point = None
                    tsdf_planner.target_point = None
            
            else: # 장기/단기 목표가 모두 있으면 계속 진행합니다.
                logging.info(f"Continuing towards current target: {tsdf_planner.target_point}")


        elif navigation_mode == "memory_driven":
            # In Phase 1, we just navigate. If we arrive, we will re-evaluate.
            logging.info(f"Phase 1 In Progress: Navigating towards memory candidate at {tsdf_planner.max_point.position}")

        mode_info = f"Mode: {navigation_mode.upper()}"
        target_info = f"| Target: {tsdf_planner.target_point}" if tsdf_planner.target_point is not None else ""
        logging.info(f"\nStep {cnt_step}/{max_steps}, Global step: {global_step} | {mode_info} {target_info}")

        # --- 4. 에이전트 한 스텝 이동 ---
        step_vals = tsdf_planner.agent_step(
            pts=current_pts,
            angle=current_angle,
            objects=scene.objects,
            snapshots=scene.snapshots,
            pathfinder=scene.pathfinder,
            cfg=app_state.cfg.planner if hasattr(app_state.cfg, "planner") else app_state.cfg,
            path_points=tracked_goal_path, # 목표 추적 모드일 때 경로 전달
            save_visualization=app_state.cfg.save_visualization,
        )

        # 경로 탐색 실패 시, 목표를 초기화하고 다음 스텝에서 재평가하도록 합니다.
        if step_vals[0] is None:
            logging.warning("Agent step failed. Clearing targets to force re-evaluation.")
            tsdf_planner.max_point = None
            tsdf_planner.target_point = None
            continue

        current_pts, current_angle, _, fig, _, waypoint_arrived = step_vals
        app_state.logger.log_step(pts_voxel=tsdf_planner.habitat2voxel(current_pts)[:2])
        episode_context["visited_positions"].append(current_pts.copy())
        
        # [REVISED ARRIVAL LOGIC]
        # 중간 길목(waypoint)에 도착했다면, 단기 목표만 초기화합니다.
        if waypoint_arrived:
            logging.info("Intermediate waypoint reached.")
            tsdf_planner.target_point = None

            # 최종 장기 목표(max_point)에 충분히 가까워졌는지 확인합니다.
            if tsdf_planner.max_point is not None:
                current_voxel_pos = tsdf_planner.normal2voxel(pos_habitat_to_normal(current_pts))
                # 0.5m 이내로 근접했다면 최종 목표에 도달한 것으로 간주합니다.
                dist_to_max_point_m = np.linalg.norm(current_voxel_pos[:2] - tsdf_planner.max_point.position) * tsdf_planner._voxel_size
                
                if navigation_mode == "memory_driven" and dist_to_max_point_m < 0.5:
                    logging.info(f"Phase 1 COMPLETE: Arrived at memory candidate location.")
                    # 도착 후 주변을 스캔하고, 목표가 보이지 않으면 Phase 2로 전환해야 함.
                    # 우선은 도착 시 바로 Phase 2로 전환하여 재평가하도록 구현.
                    tsdf_planner.max_point = None # 장기 목표 초기화
                    navigation_mode = "exploration"
                    logging.info("Switching to Phase 2 (Exploration) for re-evaluation.")

                elif navigation_mode == "exploration" and dist_to_max_point_m < 0.5: 
                    logging.info("Vicinity of long-term target reached. Clearing for full re-evaluation.")
                    tsdf_planner.max_point = None # 장기 목표를 초기화하여 다음 스텝에서 재평가하도록 합니다.

        # 목표 추적/정보 수집 단계에서 목적지에 도착했을 때 처리
        if waypoint_arrived:
            if navigation_mode == "goal_tracking": # This mode seems unused in the original file, but keeping the logic
                logging.info("Arrived at tracked goal destination! Reverting to EXPLORATION mode.")
                navigation_mode = "exploration"
                tracked_goal_path = None
            
        # --- Visualization (optional) ---
        if app_state.cfg.save_visualization and fig is not None:
            # Prepare visualization info from the selection process if it happened in this step
            vis_selection_info = None
            if candidates_info: # candidates_info is only populated when _select_next_target is called
                selection_dir = os.path.join(app_state.logger.subtask_dir, "selection")
                scored_candidates = candidates_info["scored_candidates"]
                best_candidate_data = candidates_info["chosen_candidate_data"]
                chosen_candidate = best_candidate_data["candidate"]

                candidates_for_log = []
                scores_for_log = {}
                for sc in scored_candidates:
                    cand = sc["candidate"]
                    name = f"{cand['type']}_{cand['data'].frontier_id if cand['type'] == 'frontier' else cand['data'].name}"
                    candidates_for_log.append({"rgb": cand["rgb"], "name": name})
                    scores_for_log[name] = sc["semantic_score"]
                
                chosen_name_for_log = f"{chosen_candidate['type']}_{chosen_candidate['data'].frontier_id if chosen_candidate['type'] == 'frontier' else chosen_candidate['data'].name}"

                caption = (
                    f"Goal: {original_goal_str}\n"
                    f"Chosen: {chosen_name_for_log} (Score: {best_candidate_data['semantic_score']:.2f})\n"
                    f"ITM: {best_candidate_data['score_itm']:.2f}, VLM: {best_candidate_data['vlm_likelihood']:.2f}, Expl: {best_candidate_data['exploration_score']:.2f}"
                )
                
                vis_selection_info = {
                    "save_dir": selection_dir,
                    "candidates": candidates_for_log,
                    "chosen_name": chosen_name_for_log,
                    "scores": scores_for_log,
                    "caption": caption,
                    "filename_prefix": ""
                }

            app_state.logger.log_step_visualizations(
                global_step=global_step,
                subtask_id=subtask_metadata["subtask_id"],
                subtask_metadata=subtask_metadata,
                fig=fig,
                candidates_info=vis_selection_info # Use the processed info
            )

        # Apply warm-start orientation hint once when we arrive near the target
        if waypoint_arrived and "warmstart_angle" in episode_context:
            current_angle = episode_context["warmstart_angle"]
            del episode_context["warmstart_angle"]

        # --- 공통 실행: 최종 성공 여부 판정 ---
        agent_subtask_distance = calc_agent_subtask_distance(
            current_pts, subtask_metadata["viewpoints"], scene.pathfinder
        )
        success_by_distance = agent_subtask_distance < app_state.cfg.success_distance
        
        # [FIX] 매 스텝마다 성공 여부를 체크하여 조기 종료.
        # 기존에는 goal_tracking 모드에서만 성공을 체크했기 때문에,
        # description/image goal에서는 성공할 수 없는 문제가 있었음.
        if success_by_distance:
            logging.info(
                f"SUCCESS: Distance condition met at step {global_step} ({agent_subtask_distance:.2f}m < 1.0m). Stopping."
            )
            break

        # If the goal is found and we are close enough, call STOP
        if navigation_mode == "goal_tracking" and waypoint_arrived:
            # We have reached the destination planned by _detect_and_verify_goal
            # Now check if we are actually close to the goal viewpoints
            # 위에서 이미 success_by_distance로 체크하고 break 하므로, 이 블록은 사실상 도달하기 어려움.
            # 하지만 혹시 모를 상황을 위해 유지.
            logging.info(
                f"INFO: Goal tracking arrived at destination. Distance to goal is {agent_subtask_distance:.2f}m."
            )
            # 이미 위에서 break 되었으므로, 이곳의 break는 제거하거나 주석처리해도 무방.
            # break
            
        # Check for max steps
        if cnt_step >= max_steps:
            logging.warning(f"Max steps reached ({max_steps}). Ending subtask.")
            break
    
    # 결과 로깅
    # 루프가 종료된 시점의 success_by_distance 값을 사용합니다.
    success_by_snapshot = False  # 본 방법론은 스냅샷 기반이 아님
    if success_by_distance:
        logging.info(f"Task completed successfully in {cnt_step} steps!")
    else:
        logging.info(f"Task failed after {cnt_step} steps")
    
    # 로거에 결과 기록
    logging.info(f"Logging result for subtask_id: {subtask_id}")
    app_state.logger.log_subtask_result(
        success_by_snapshot=success_by_snapshot,
        success_by_distance=success_by_distance,
        subtask_id=subtask_id, # <-- Use the direct subtask_id argument
        gt_subtask_explore_dist=subtask_metadata.get("gt_subtask_explore_dist", 0),
        goal_type=subtask_metadata.get("goal_type", "unknown"),
        n_filtered_snapshots=0,  # 새로운 방법론에서는 snapshot 사용 안함
        n_total_snapshots=0,
        n_total_frames=cnt_step,
    )
    
    return {
        "success": success_by_distance, # task_success 대신 정확한 값을 반환합니다.
        "final_position": current_pts,
        "final_angle": current_angle,
        "steps_taken": cnt_step,
        "final_global_step": global_step,
    }

def _observe_and_update_maps(
    scene, tsdf_planner, current_pts, current_angle, cnt_step,
    cfg, cam_intr, min_depth, max_depth, eps_frontier_dir
) -> List[Dict[str, Any]]:
    """
    Performs a full 360-degree scan to observe the surroundings, then updates
    the TSDF and frontier maps. This logic is consistent for every step.
    """
    # YAML 설정 파일에서 스캔 횟수를 읽어옵니다.
    num_scan_views = cfg.num_scan_views
    angle_increment = np.deg2rad(360 / num_scan_views)
    
    all_angles = [
        current_angle + (i * angle_increment)
        for i in range(num_scan_views)
    ]

    # 현재 위치의 맵 좌표(보xel) 저장
    current_map_pos = tsdf_planner.normal2voxel(pos_habitat_to_normal(current_pts))

    current_step_observations = []
    for view_idx, ang in enumerate(all_angles):
        obs, cam_pose = scene.get_observation(current_pts, angle=ang)
        rgb = obs["color_sensor"]
        depth_unfiltered = obs["depth_sensor"]

        obs_file_name = f"{cnt_step}-{view_idx}.png"
        obs_data = {
            "rgb": rgb, "angle": ang, "cam_pose": cam_pose, "name": obs_file_name, "depth": depth_unfiltered,
            "map_pos": (int(current_map_pos[0]), int(current_map_pos[1]))
        }
        # FIX: scene.all_observations에 더 이상 저장하지 않습니다.
        current_step_observations.append(obs_data)

        if min_depth is not None and max_depth is not None:
            depth = np.where(
                (depth_unfiltered >= min_depth) & (depth_unfiltered <= max_depth),
                depth_unfiltered, np.nan,
            )
        else:
            depth = depth_unfiltered

        tsdf_planner.integrate(
            color_im=rgb, depth_im=depth, cam_intr=cam_intr,
            cam_pose=pose_habitat_to_tsdf(cam_pose), obs_weight=1.0,
            margin_h=int(cfg.margin_h_ratio * cfg.img_height),
            margin_w=int(cfg.margin_w_ratio * cfg.img_width),
            explored_depth=cfg.explored_depth,
        )

    # (2) Update frontier map
    tsdf_planner.update_frontier_map(
        pts=current_pts,
        cfg=cfg.planner if hasattr(cfg, "planner") else cfg,
        scene=scene,
        cnt_step=cnt_step,
        save_frontier_image=cfg.save_visualization,
        eps_frontier_dir=eps_frontier_dir,
    )

    return current_step_observations


def _detect_and_verify_goal_in_views(
    current_step_observations: List[Dict[str, Any]],
    goal: Any,
    scene: Scene,
    tsdf_planner: TSDFPlanner,
    app_state: AppState,
) -> tuple[bool, Optional[tuple[int, int]]]:
    """
    Iterates through observations to find the highest-confidence goal.
    """
    is_goal_visible = False
    target_voxel = None

    for obs_data in current_step_observations:
        is_visible, voxel = _detect_and_verify_goal(
            rgb=obs_data["rgb"],
            depth=obs_data["depth"],
            goal=goal,
            goal_type="object", # Assuming object goal for this function
            detection_model=scene.detection_model,
            sam_predictor=scene.sam_predictor,
            itm=app_state.itm,
            tsdf_planner=tsdf_planner,
            cam_intr=app_state.cam_intr,
            cam_pose=obs_data["cam_pose"],
            cfg=app_state.cfg,
        )
        if is_visible:
            is_goal_visible = True
            target_voxel = voxel
            logging.info(f"Goal detected in view {obs_data['name']}!")
            break 

    return is_goal_visible, target_voxel


def _select_next_target(
    app_state: AppState,
    episode_context: Dict[str, Any],
    current_pts: np.ndarray,
    goal: Any,
    goal_type: str,
    original_goal_str: str, # Added for logging
    tsdf_planner: TSDFPlanner,
    selection_dir: str,
    target_objects: List[str], # Receive refined targets
    mem_candidates: List[Dict[str, Any]], # Receive memory candidates
) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]: # Return type changed to Any
    """
    Selects the next target from a unified pool of geometric frontiers and semantic memories.
    """
    cfg = app_state.cfg
    score_cache = episode_context.get("frontier_score_cache", {})
    
    # --- 1. 통합 후보군 생성 (Unified Candidate Pool) ---
    all_candidates = []

    # 1a. 기하학적 프론티어 추가
    subtask_obs_cache = episode_context.get("subtask_observation_cache", {})
    for f in tsdf_planner.frontiers:
        if f.source_observation_name and f.source_observation_name in subtask_obs_cache:
            source_obs = subtask_obs_cache[f.source_observation_name]
            all_candidates.append({
                "type": "frontier",
                "data": f,
                "rgb": source_obs["rgb"],
                "key_objects": [],
                "descriptors": None, # Will be generated on-the-fly
            })

    # 1b. 의미론적 메모리 추가 (long-term memory에서 가져온 AnalyzedObservation 객체)
    # Re-fetch full memory objects for scoring
    full_mem_candidates = [
        mem for mem in app_state.observation_memory 
        if any(c["map_pos"] == mem.map_pos for c in mem_candidates)
    ]

    for mem in full_mem_candidates:
        all_candidates.append({
            "type": "memory",
            "data": mem,
            "rgb": mem.rgb,
            "key_objects": mem.key_objects,
            "descriptors": mem.descriptors,
        })

    if not all_candidates:
        logging.info("No candidates (frontiers or memories) available for scoring.")
        return None, None

    # --- 2. 통합 후보군 평가 ---
    scored_candidates = []
    cur_voxel = tsdf_planner.normal2voxel(pos_habitat_to_normal(current_pts))

    for cand in all_candidates:
        score_itm = 0.0
        vlm_likelihood_score = 0.0
        
        # --- 점수 계산 ---
        # Cache key 생성
        cache_key = ""
        if cand["type"] == "frontier":
            cache_key = f"frontier_{cand['data'].frontier_id}"
        elif cand["type"] == "memory":
            # Assuming mem.name is unique enough for a step, or use a combination of name and timestamp
            cache_key = f"memory_{cand['data'].name}_{cand['data'].timestamp}"

        if cache_key not in score_cache:
            logging.info(f"Cache MISS for {cache_key}. Performing VLM analysis.")
            
            descriptors = cand["descriptors"]
            if descriptors is None: # For frontiers, generate descriptors
                descriptors, vlm_likelihood_score, parsed_objects = app_state.desc_extractor.analyze_scene_for_goal(
                    rgb=cand["rgb"], goal=goal, key_objects=cand["key_objects"]
                )
                if parsed_objects: cand["key_objects"] = parsed_objects
            else: # For memories, VLM likelihood is re-evaluated for the current goal
                _, vlm_likelihood_score, _ = app_state.desc_extractor.analyze_scene_for_goal(
                    rgb=cand["rgb"], goal=goal, key_objects=cand["key_objects"]
                )

            # ITM score
            if descriptors and target_objects:
                all_itm_scores = [
                    float(s.max()) for t in target_objects 
                    if (s := app_state.itm.text_text_scores(t, descriptors)) is not None and s.size > 0
                ]
                score_itm = max(all_itm_scores) if all_itm_scores else 0.0

            score_cache[cache_key] = {"score_itm": score_itm, "vlm_likelihood": vlm_likelihood_score}
        else:
            logging.info(f"Cache HIT for {cache_key}.")
            cached_scores = score_cache[cache_key]
            score_itm = cached_scores.get("score_itm", 0.0)
            vlm_likelihood_score = cached_scores.get("vlm_likelihood", 0.0)
        
        # --- 최종 점수 집계 ---
        # Semantic Score
        semantic_score = (cfg.w_itm_score * score_itm) + (cfg.w_vlm_score * vlm_likelihood_score)

        # Exploration Score (only for frontiers)
        exploration_score = 0.0
        if cand["type"] == "frontier":
            f = cand["data"]
            unexplored_volume = tsdf_planner.get_unexplored_volume_from_frontier(f)
            normalized_volume_score = np.tanh(unexplored_volume / 1000.0)
            distance = np.linalg.norm(cur_voxel[:2] - f.position) * tsdf_planner._voxel_size
            exploration_score = 0.25 * (normalized_volume_score + (1.0 / (1.0 + distance)))

        scored_candidates.append({
            "candidate": cand,
            "semantic_score": semantic_score,
            "score_itm": score_itm,
            "vlm_likelihood": vlm_likelihood_score,
            "exploration_score": exploration_score
        })

    if not scored_candidates:
        return None, None
        
    # --- 3. 최적 후보 선택 로직 ---
    scored_candidates.sort(key=lambda x: x["semantic_score"], reverse=True)
    best_candidate_data = scored_candidates[0]
    
    # Condition 1: All scores are low, resort to pure exploration
    if best_candidate_data["semantic_score"] < cfg.low_score_threshold:
        logging.info(f"All candidate scores are below threshold {cfg.low_score_threshold}. "
                     f"Switching to exploration mode.")
        scored_candidates.sort(key=lambda x: x["exploration_score"], reverse=True)
        best_candidate_data = scored_candidates[0]
        logging.info(f"Best Candidate (by Exploration): {best_candidate_data['candidate']['data'].frontier_id if best_candidate_data['candidate']['type'] == 'frontier' else best_candidate_data['candidate']['data'].name} "
                     f"with Exploration Score: {best_candidate_data['exploration_score']:.3f}")

    # Condition 2: Tie-breaking for top two candidates
    else:
        top_score = scored_candidates[0]["semantic_score"]
        tie_candidates = [
            f for f in scored_candidates 
            if top_score - f["semantic_score"] < cfg.tie_breaking_threshold
        ]
        
        if len(tie_candidates) > 1:
            logging.info(f"{len(tie_candidates)} candidates are in a tie-breaking contention. "
                         f"Using exploration score to resolve.")
            
            # Find the best among the candidates using the combined score
            best_in_tie = max(
                tie_candidates, 
                key=lambda x: x["semantic_score"] + x["exploration_score"]
            )
            
            # Log if the winner changed
            if best_in_tie['candidate']['data'].frontier_id != best_candidate_data['candidate']['data'].frontier_id:
                original_winner = best_candidate_data
                logging.info(f"Tie-breaker chose Candidate {best_in_tie['candidate']['data'].frontier_id if best_in_tie['candidate']['type'] == 'frontier' else best_in_tie['candidate']['data'].name} "
                             f"(Combined: {best_in_tie['semantic_score'] + best_in_tie['exploration_score']:.3f}) "
                             f"over Candidate {original_winner['candidate']['data'].frontier_id if original_winner['candidate']['type'] == 'frontier' else original_winner['candidate']['data'].name} "
                             f"(Combined: {original_winner['semantic_score'] + original_winner['exploration_score']:.3f}).")
            
            best_candidate_data = best_in_tie
    
    chosen_candidate = best_candidate_data["candidate"]
    
    logging.info(f"Best Candidate (type: {chosen_candidate['type']}): "
                 f"Semantic Score: {best_candidate_data['semantic_score']:.3f} "
                 f"(ITM: {best_candidate_data['score_itm']:.3f}, VLM: {best_candidate_data['vlm_likelihood']:.3f}, "
                 f"Exploration: {best_candidate_data['exploration_score']:.3f})")

    # --- 4. 시각화 정보 준비 ---
    candidates_info = {
        "scored_candidates": scored_candidates,
        "chosen_candidate_data": best_candidate_data,
    }

    return chosen_candidate["data"], candidates_info
    

def _analyze_and_store_semantic_memory(
    current_step_observations: List[Dict[str, Any]],
    episode_context: Dict[str, Any],
    app_state: AppState,
):
    """
    Analyzes new observations, identifies GEOMETRICALLY and SEMANTICALLY novel ones,
    and updates the long-term (app_state) observation memory.
    """
    cfg = app_state.cfg
    itm = app_state.itm
    long_term_memory = app_state.observation_memory # Use the global, persistent memory
    
    # --- 1. Filter for Geometrically Novel Observations ---
    diversity_dist_m = cfg.observation_diversity_distance_m
    diversity_angle_deg = cfg.observation_diversity_angle_deg
    diversity_angle_rad = np.deg2rad(diversity_angle_deg)
    
    geometrically_novel_observations = []
    for obs in current_step_observations:
        is_geometrically_novel = True
        # Voxel size needs to be accessed from a planner instance.
        tsdf_planner = episode_context.get("tsdf_planner")
        if not tsdf_planner: continue
        diversity_dist_vox = diversity_dist_m / tsdf_planner._voxel_size
        
        for mem in long_term_memory:
            dist = np.linalg.norm(np.array(obs["map_pos"]) - np.array(mem.map_pos))
            if dist <= diversity_dist_vox:
                angle_diff = abs(obs["angle"] - mem.angle)
                if angle_diff > np.pi: angle_diff = 2 * np.pi - angle_diff
                if angle_diff <= diversity_angle_rad:
                    is_geometrically_novel = False
                    break
        if is_geometrically_novel:
            geometrically_novel_observations.append(obs)

    # --- 2. Filter for Semantically Novel Observations ---
    if not geometrically_novel_observations:
        return
        
    semantic_novelty_threshold = cfg.semantic_novelty_threshold
    
    for obs_data in geometrically_novel_observations:
        # Generate descriptors for the new view. The goal string here is for broad context.
        refined_goal_str = ", ".join(episode_context.get("target_objects", []))
        if not refined_goal_str:
             refined_goal_str = episode_context.get("subtask_goal_str", "indoor scene")

        descriptors, _, objects_from_desc = app_state.desc_extractor.analyze_scene_for_goal(
            rgb=obs_data["rgb"],
            goal=refined_goal_str,
            key_objects=[],
        )
        if not descriptors:
            continue

        # Check for semantic novelty against long-term memory
        is_semantically_novel = True
        if long_term_memory:
            memory_descriptors = [desc for mem in long_term_memory for desc in mem.descriptors]
            if memory_descriptors:
                scores = itm.text_text_scores(descriptors, memory_descriptors)
                max_similarity = float(scores.max()) if scores is not None and scores.size > 0 else 0.0
                if max_similarity >= semantic_novelty_threshold:
                    is_semantically_novel = False
        
        if is_semantically_novel:
            # Prefer objects parsed by analyzer; fallback to descriptor-as-object if empty
            if objects_from_desc:
                key_objects = list({o.strip() for o in objects_from_desc if isinstance(o, str) and len(o.strip()) > 0})
            else:
                key_objects = list({d.strip() for d in descriptors if isinstance(d, str) and len(d.strip()) > 0})
            logging.info(f"Found semantically novel observation {obs_data['name']}. Storing in long-term memory.")
            analyzed_obs = AnalyzedObservation(
                name=obs_data["name"],
                map_pos=obs_data["map_pos"],
                angle=obs_data["angle"],
                descriptors=descriptors,
                key_objects=key_objects if key_objects else [],
                rgb=obs_data["rgb"],
            )
            # Add to both for visualization and persistence
            episode_context["observation_memory"].append(analyzed_obs)
            app_state.observation_memory.append(analyzed_obs)


def get_relevant_observations(
    frontier: Frontier,
    observations: List[Dict[str, Any]],
    angle_threshold_deg: float,
) -> List[Dict[str, Any]]:
    """
    주어진 관측 리스트 내에서 특정 프론티어와 관련된 모든 관측들을 찾습니다.
    """
    relevant_obs = []
    frontier_angle = np.arctan2(frontier.orientation[1], frontier.orientation[0])
    angle_threshold_rad = np.deg2rad(angle_threshold_deg)

    for obs_data in observations:
        obs_angle = obs_data.get("angle")
        if obs_angle is None:
            continue
        
        # 각도 차이 계산 (0~π 범위로 정규화)
        angle_diff = abs(frontier_angle - obs_angle)
        if angle_diff > np.pi:
            angle_diff = 2 * np.pi - angle_diff

        # 각도 차이가 임계값 이내이면 관련 관측으로 추가
        if angle_diff <= angle_threshold_rad:
            relevant_obs.append(obs_data)

    logging.debug(f"Frontier {frontier.frontier_id}: Found {len(relevant_obs)} relevant observations. "
                 f"Frontier angle: {frontier_angle:.1f}°")
    return relevant_obs


def main(cfg_file: str, start_ratio: float, end_ratio: float, split: int, scene_id: Optional[str] = None):
    """메인 실행 함수: 준비 -> 평가"""
    app_state = setup(cfg_file, start_ratio, end_ratio, split)
    run_evaluation(app_state, start_ratio, end_ratio, split, scene_id)
    logging.info("All scenes finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path", default="cfg/eval_stretch3_mzson.yaml", type=str)
    parser.add_argument("--start_ratio", help="start ratio", default=0.0, type=float)
    parser.add_argument("--end_ratio", help="end ratio", default=1.0, type=float)
    parser.add_argument("--split", help="which episode", default=1, type=int)
    parser.add_argument("--scene_id", help="Run evaluation for a specific scene ID", default=None, type=str)
    args = parser.parse_args()
    
    # If scene_id is an empty string or the unsubstituted variable from launch.json, treat it as None
    if args.scene_id == "" or args.scene_id == "${input:sceneId}":
        args.scene_id = None

    # logging.info(f"***** Running {OmegaConf.load(args.cfg_file).exp_name} with VLM methodology *****")
    main(args.cfg_file, args.start_ratio, args.end_ratio, args.split, args.scene_id) 